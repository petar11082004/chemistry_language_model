from dataclasses import dataclass, asdict
from typing import Tuple, Dict
import math
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# 1) Column definitions
#--------------------------
COLS_GAP = ["log_delta_e", "inv_delta_e"] # we willprepend a constant 1
COL_T1 = ["t1"]

COLS_X_OCC = [
    "occ_maglz_expect",
    "occ_charges_0", "occ_charges_1", "occ_charges_2",
    "occ_inv_R_01", "occ_inv_R_02", "occ_inv_R_12",
    "occ_mo_energies",
]
COLS_X_VIR = [
    "vir_maglz_expect",
    "vir_charges_0", "vir_charges_1", "vir_charges_2",
    "vir_inv_R_01", "vir_inv_R_02", "vir_inv_R_12",
    "vir_mo_energies",
]

# -------------------------
# 2) Utils
#--------------------------
class ShiftedSoftplus(nn.Module):
    """ρ(x) = (1/β) * log(0.5 + 0.5 * exp(β x)) which is softplus shifted to be ~0 at x=0."""
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        z = torch.clamp(self.beta * x, -50, 50)
        return (1.0/ self.beta)* torch.log(0.5 + 0.5*torch.exp(z))
    
def _to_tensor(x:np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype = torch.float32, device = device)

# -------------------------
# 3) Dataset
# -------------------------

class T1Dataset(Dataset):
    """Packs all inputs so each batch is (X_occ, X_vir, gap_phi, y)."""
    def __init__(self, df: pd.DataFrame):
        # Core blocks
        x_occ = df[COLS_X_OCC].to_numpy(dtype = np.float32)
        x_vir = df[COLS_X_VIR].to_numpy(dtype = np.float32)
        
        # φ(Δε) = [1, log Δε, (Δε)^-1] 
        gap_raw = df[COLS_GAP].to_numpy(dtype = np.float32) # columns: [logΔε, invΔε]
        ones = np.ones((gap_raw.shape[0], 1), dtype = np.float32)
        gap_phi = np.concatenate([ones, gap_raw], axis = 1)
        y = df[COL_T1].to_numpy(dtype = np.float32)

        self.x_occ = x_occ
        self.x_vir = x_vir
        self.gap_phi = gap_phi
        self.y = y

    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        return(
            self.x_occ[idx],
            self.x_vir[idx],
            self.gap_phi[idx],
            self.y[idx]
        )

# -------------------------
# 4) Model
# -------------------------

@dataclass
class T1Config:
    d_occ: int = 8
    d_vir: int = 8
    d_gap_phi: int = 3 # [1, logΔε, (Δε)^-1]
    hidden_dim: int = 64
    huber_delta: float = 1.0
    lambda_sign: float = 1.0
    lambda_aux: float = 1.0
    lambda_mono: float = 1.0
    weight_decay: float = 1e-4
    lr: float = 1e-4
    monotonic_eps: float = 1e-2 # finite-difference ε for Δε
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class T1Model(nn.Module):
    """
    Inputs:
        X_occ: (B, 8)
        X_vir: (B, 8)
        gap_phi: (B, 3) = [1, logΔε, (Δε)^-1]

    Returns:
        t_hat, s_hat, log_amp, g(all (B, ))
    """

    def __init__(self, cfg: T1Config):
        super().__init__()
        self.cfg = cfg

        act = ShiftedSoftplus()
        self.occ_net = nn.Sequential(
            nn.Linear(cfg.d_occ, cfg.hidden_dim),
            act,
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            act,
            nn.Linear(cfg.hidden_dim, cfg.d_occ),
            act,
        )

        self.vir_net = nn.Sequential(
            nn.Linear(cfg.d_vir, cfg.hidden_dim),
            act,
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            act,
            nn.Linear(cfg.hidden_dim, cfg.d_vir),
            act,
        )

        # interaction weights (all trainable)
        self.M = nn.Parameter(torch.randn(cfg.d_occ, cfg.d_vir) * 0.01) #(8, 8)
        self.u = nn.Parameter(torch.randn(cfg.d_occ) * 0.01)            # (8,)
        self.wg = nn.Parameter(torch.randn(cfg.d_gap_phi + cfg.d_occ + cfg.d_vir) * 0.01) #(19,)
        self.cg = nn.Parameter(torch.zeros(1))

        self.vs = nn.Parameter(torch.randn(cfg.d_gap_phi + cfg.d_occ + cfg.d_vir) * 0.01) #(19,)
        self.alpha_s = nn.Parameter(torch.zeros(1))
        self.bs = nn.Parameter(torch.zeros(1))

        self.vm = nn.Parameter(torch.randn(cfg.d_gap_phi + cfg.d_occ + cfg.d_vir) * 0.01) #(19,)
        self.alpha_m = nn.Parameter(torch.zeros(1))
        self.bm = nn.Parameter(torch.zeros(1))

    def forward (self, X_occ: torch.Tensor, X_vir: torch.Tensor, gap_phi: torch.Tensor):
        """
        gap_phi[:, 0] = 1
        gap_phi[:, 1] = logΔε
        gap_phi[:, 2] = (Δε)^-1
        """

        hi = self.occ_net(X_occ) #(B, 8)
        ha = self.vir_net(X_vir) #(B, 8)

        # ξ = [h_i, h_a, 1, logΔε, (Δε)^-1,]  -> (B,19)
        xi = torch.cat([hi, ha, gap_phi], dim = -1)

        # g= h_i @ M @ h_a^T + u @ (h_i ⊙ h_a)^T + w_g @ ξ^T + c_g
        g_bilinear = torch.einsum('bi,ij,bj->b', hi, self.M, ha)
        g_hadamard = torch.einsum('i,bi->b', self.u, hi * ha)
        g_linear = torch.einsum('j,bj->b', self.wg, xi) + self.cg

        if torch.isnan(hi).any() or torch.isinf(hi).any():
            print("⚠️ NaN/Inf in hi (occ_net output)")
        if torch.isnan(ha).any() or torch.isinf(ha).any():
            print("⚠️ NaN/Inf in ha (vir_net output)")

        if torch.isnan(g_bilinear).any() or torch.isinf(g_bilinear).any():
            print("⚠️ NaN/Inf in g_bilinear")
        if torch.isnan(g_hadamard).any() or torch.isinf(g_hadamard).any():
            print("⚠️ NaN/Inf in g_hadamard")
        if torch.isnan(g_linear).any() or torch.isinf(g_linear).any():
            print("⚠️ NaN/Inf in g_linear")


        g = g_bilinear + g_hadamard + g_linear
        g = torch.clamp(g, -1e6, 1e6)

        # ŝ and log|t|
        s_hat = torch.tanh(self.alpha_s * g + torch.einsum('j,bj -> b', self.vs, xi) + self.bs) #(B,)
        log_amp = self.alpha_m*g + torch.einsum('j, bj-> b', self.vm, xi) + self.bm        #(B,)
        log_amp = torch.clamp(log_amp, -20, 20)

        inv_delta_e = torch.clamp(gap_phi[:, 2], min=1e-3, max=1e3) # (B,)
        t_hat = s_hat * torch.exp(log_amp) * inv_delta_e
        t_hat = torch.clamp(t_hat, -1e6, 1e6)

        return t_hat, s_hat, log_amp, g
    
# -------------------------
# 5) Loss
# -------------------------

class T1Loss(nn.Module):
    """
    J = Huber(t, t̂) + λ_sign * BCE(sign) + λ_aux * Huber(g, Δε t) + λ_mono * R_gap
    Weight decay is done by the optimizer (Adam(..., weight_decay=...)).
    """
    def __init__(self, cfg:T1Config, model:T1Model):
        super().__init__()
        self.cfg = cfg
        self.model = model
    
    def forward(
        self, 
        outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        X_occ: torch.Tensor,
        X_vir: torch.Tensor,
        gap_phi: torch.Tensor,
    ) -> torch.Tensor:
        t_hat, s_hat, _, g = outputs
        print(f"shape of g:{g.shape}")
        t_true = targets.squeeze(-1) # (B,)
        print(f"shape of t_true:{t_true.shape}")

        #1) Amplitude fit (Huber)
        loss_amp = F.huber_loss(t_hat, t_true, delta =self.cfg.huber_delta)

        # --- Debugging checks ---
        if torch.isnan(t_hat).any():
            print("⚠️ NaN detected in t_hat")
        
        if torch.isinf(t_hat).any():
            print("⚠️ Inf detected in t_hat")

        #2) Sign fit (BCE) with p = (1 + ŝ)/2 and label y_sign = 1[t_true >= 0]
        p = (1.0 + s_hat)*0.5
        p = torch.clamp(p, 1e-6, 1.0 - 1e-6)

        y_sign = (t_true >= 0).float()
        y_sign = torch.clamp(y_sign, 0.0, 1.0)
        y_sign = y_sign.view_as(p)

        # --- Debug ---
        if not torch.all((y_sign == 0) | (y_sign == 1)):
            print("⚠️ y_sign contains invalid values:", y_sign.unique())

        loss_sign = F.binary_cross_entropy(p, y_sign)*self.cfg.lambda_sign

        # --- Debugging checks ---
        if torch.isnan(p).any() or torch.isinf(p).any():
            print("⚠️ NaN/Inf detected in p")
            print("p sample:", p[:5].detach().cpu().numpy())
            print("y_sign sample:", y_sign[:5].detach().cpu().numpy())
            print("p shape:", p.shape, "y_sign shape:", y_sign.shape)

        #3) Auxilary stabiliser: g ≈ Δε * t_true
        inv_delta_e = gap_phi[:, 2]
        delta_e = 1.0/torch.clamp(inv_delta_e, min = 1e-6, max = 1e6)
        y_aux = delta_e * t_true
        corr = torch.corrcoef(torch.stack([g.flatten(), y_aux.flatten()]))[0,1] 
        loss_aux = (1 - corr) * self.cfg.lambda_aux

        # 4) Monotonicity prior: encourage |t| to decrease with Δε
        #    R_gap = mean( relu( (|t(Δε+dε)| - |t(Δε)|) / dε ) )
        d_eps = self.cfg.monotonic_eps
        delta_e_plus = delta_e + d_eps
        gap_phi_plus = gap_phi.clone()
        gap_phi_plus[:, 1] = torch.log(delta_e_plus) # logΔε
        gap_phi_plus[:, 2] = 1/ delta_e_plus  # (Δε)^-1

        with torch.enable_grad():
            # Important: recompute with perturbed Δε (weights get gradients)
            t_hat_plus, _, _, _ = self.model(X_occ, X_vir, gap_phi_plus)

        d_abs = (torch.abs(t_hat_plus) - torch.abs(t_hat))/d_eps
        r_gap = torch.relu(d_abs).mean()*self.cfg.lambda_mono

        total = loss_amp + loss_sign + loss_aux + r_gap

        '''
        # ---- Debug print ----
        if torch.isnan(total).any() == False:
            print(f"amp={loss_amp.item():.3e}, "
                  f"sign={loss_sign.item():.3e}, "
                  f"aux={loss_aux.item():.3e}, "
                  f"mono={r_gap.item():.3e}, "
                  f"total={total.item():.3e}")
        '''

        return total


# -------------------------
# 6) Trainer / Wrapper
# -------------------------

class T1Regressor:
    """
    Thin wrapper that:
        - holds the fitted T1Dataprocessor
        - trains the neural model
        - saves/loads everything
        - predicts on raw DataFrames
    """
    def __init__(self, cfg:T1Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = T1Model(cfg).to(self.device)
        self.processor = None # filled in fit() or load()

    # ---------- Fit ----------
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        epochs: int = 200,
        batch_size: int = 64,
        processor = None, # optionally pass a pre-fitted processor
    ):
        
        # 1) processor
        if processor is None:
            from t1_data_processor import T1DataProcessor
            processor = T1DataProcessor()
            processor.fit(train_df)
        self.processor = processor

        df_proc = self.processor.transform(train_df)
        ds = T1Dataset(df_proc)
        train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last = False)

        # Validation loader if provided
        val_loader = None
        if val_df is not None:
            df_val = self.processor.transform(val_df)
            ds_val = T1Dataset(df_val)
            val_loader = DataLoader(ds_val, batch_size= batch_size, shuffle=False)

        # 2) optimizer & loss
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr = self.cfg.lr,
            weight_decay = self.cfg.weight_decay,
        )
        criterion = T1Loss(self.cfg, self.model)
        # 3) training loop
        self.model.train()
        for epoch in range(1, epochs + 1):
            running = 0.0
            for X_occ, X_vir, gap_phi, y in train_loader:
                X_occ = _to_tensor(X_occ, self.device)
                X_vir = _to_tensor(X_vir, self.device)
                gap_phi = _to_tensor(gap_phi, self.device)
                y = _to_tensor(y, self.device)

                opt.zero_grad(set_to_none = True)
                outputs = self.model(X_occ, X_vir, gap_phi)
                loss = criterion(outputs, y, X_occ, X_vir, gap_phi)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step()
                running += loss.item()

            avg_train = running/max(1, len(train_loader))

            # --- Validation ---
            avg_val = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_occ, X_vir, gap_phi, y in val_loader:
                        X_occ = _to_tensor(X_occ, self.device)
                        X_vir = _to_tensor(X_vir, self.device)
                        gap_phi = _to_tensor(gap_phi, self.device)
                        y = _to_tensor(y, self.device)

                        outputs = self.model(X_occ, X_vir, gap_phi)
                        loss = criterion(outputs, y, X_occ, X_vir, gap_phi)
                        val_loss += loss.item()
                avg_val = val_loss/max(1, len(val_loader))
                self.model.train()

            # --- Logging ---
            if epoch %10 == 0 or epoch == 1:
                if avg_val is not None:
                    print(f"[Epoch {epoch}] train_loss = {avg_train:.6f}, val_loss = {avg_val:.6f}")
                else:
                    print(f"[Epoch {epoch}] train_loss = {avg_train:.6f}")

    # ---------- Predict ----------
    @torch.no_grad()
    def predict(self, df_raw: pd.DataFrame) -> np.ndarray:
        assert self.processor is not None, "Model not fitted or loaded (processor missing)."
        df = self.processor.transform(df_raw)
        ds = T1Dataset(df)
        loader = DataLoader(ds, batch_size=256, shuffle = False)
        self.model.eval()
        preds = []
        for X_occ, V_vir, gap_phi, _ in loader:
            X_occ = _to_tensor(X_occ, self.device)
            X_vir = _to_tensor(X_vir, self.device)
            gap_phi = _to_tensor(gap_phi, self.device)
            t_hat, _, _, _ = self.model(X_occ, X_vir, gap_phi)
            preds.append(t_hat.cpu().numpy())
        return np.concatenate(preds, axis = 0)
    
    # ---------- Save / Load ----------
    def save(self, ckpt_path:str, processor_path: str):
        """Saves model weights+config and the fitted processor."""
        torch.save(
            {"state_dict": self.model.state_dict(), "config": asdict(self.cfg)},
            ckpt_path
        )
        with open(processor_path, "wb") as f:
            pickle.dump(self.processor, f)
    
    @classmethod
    def load(cls, ckpt_path: str, processor_path: str) -> "T1Regressor":
        """Loads model and fitted processor"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location = device)
        cfg = T1Config(**ckpt["config"])
        reg = cls(cfg)
        reg.model.load_state_dict(ckpt["state_dict"])
        reg.model.to(reg.device).eval()
        with open(processor_path, "rb") as f:
            reg.processor = pickle.load(f)
        return reg
