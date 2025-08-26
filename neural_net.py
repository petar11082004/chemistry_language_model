'''
import pandas as pd
import torch
from t1_data_processor import T1DataProcessor
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.nn.functional import huber_loss, binary_cross_entropy

train_df = pd.read_parquet('t1_pairs_training_20250815_170401.parquet')
processor = T1DataProcessor()
processor.fit(train_df)
train_df = processor.transform(train_df)

cols_gap = ["log_delta_e", "inv_delta_e",]
col_t1 = ["t1"]
cols_X_occ = [
    "occ_maglz_expect",
    "occ_charges_0", "occ_charges_1", "occ_charges_2",
    "occ_inv_R_01", "occ_inv_R_02", "occ_inv_R_12",
    "occ_mo_energies"
]

cols_X_vir = [
    "vir_maglz_expect",
    "vir_charges_0", "vir_charges_1", "vir_charges_2",
    "vir_inv_R_01", "vir_inv_R_02", "vir_inv_R_12",
    "vir_mo_energies"
]
gap = train_df[cols_gap].copy()
t1 = train_df[col_t1].copy()
ones = np.ones(gap.shape[0], 1)
gap["1"] = ones
X_occ = train_df[cols_X_occ].copy()
X_vir = train_df[cols_X_vir].copy()

def ssp(x, beta = 1): # shifted softplus activation function
            return (1/beta) * torch.log(0.5+0.5*torch.exp(beta * x))

class T1Model(nn.Module):

    def __init__(self, d_occ = 8, d_vir = 8, d_gap = 2, hidden_dim = 64):
        """
        d_occ :number of occupied features
        d_vir :number of virtual features
        d_gap :number of gap features
        hidden_dim : internal embedding size
        """
        super().__init__()
        
        self.activation = ssp
        self.occ_net = nn.Sequential(
              nn.Linear(d_occ, hidden_dim)
              self.activation()
              nn.Linear(hidden_dim, hidden_dim)
              self.activation()
              nn.Linear(hidden_dim, d_occ)
              self.activation()
            )

        self.vir_net = nn.Sequential(
              nn.Linear(d_vir, hidden_dim)
              self.activation()
              nn.Linear(hidden_dim, hidden_dim)
              self.activation()
              nn.Linear(hidden_dim, d_occ)
              self.activation()
        )

        self.M = torch.randn(8, 8)
        self.u = torch.randn(8)
        self.wg = torch.randn(19)
        self.cg = torch.rand(1)
        self.vs = torch.rand(19)
        self.alpha_s = torch.rand(1)
        self.bs = torch.rand(1)
        self.alpha_m = torch.rand(1)
        self.vm = torch.rand(19)
        self.bm = torch.rand(1)

    def forward(self, X_occ, X_vir, gap):
        ha = self.vir_net(X_vir)
        hi = self.occ_net(X_occ)
        xi = torch.cat(hi, ha, gap)
        g = hi @ self.M @ ha.T + self.u @ (hi * ha).T + self.wg @xi.T +self.cg
        sign = torch.tanh(self.alpha_s @ g + self.vs @ xi.T + self.bs)
        log_magnitude = self.alpha_m @ g self.vm @ xi.T self.bm
        t1 = sign@torch.exp(log_magnitude)@gap[2]
        return t1.squeeze(1), sign.squeeze(1),  log_magnitude.squeeze(1), g.squeeze(1)


class T1Loss(nn.Module):
    def __init__(self, lambda_sign=1.0, lambda_aux=1.0, lambda_l2=1e-4, lambda_mono=1.0, delta_huber=1.0):
        super().__init__()
        self.lambda_sign = lambda_sign
        self.lambda_aux = lambda_aux
        self.lambda_l2 = lambda_l2
        self.lambda_mono = lambda_mono
        self.delta_huber = delta_huber

    def forward(self, inputs, targets, delta_e, model):

        t_true = targets
        t1, sign, log_magnitude, g = inputs

        loss_amplitude = huber_loss(t1, t_true, delta=self.delta_hueber)
        
        p = (1+sign)/2
        y_sign = ((t1>=0).float())
        sign_loss = -self.lambda_sign(y_sign * np.log(p) + (1-y_sign)*np.log(1-p))

        auxilary_stabalizer_loss = self.ambda_aux*huber_loss(g, t1/gap[2])

        l2_loss = 0
        for param in model.parameters():
             l2_loss += torch.sum(param**2)
        l2_loss*= self.lambda_l2

        abs_t = torch.abs(t1)
        d_abs_t = torch.autograd.grad(
             abs_t.sum(), delta_e, create_graph = True, retain_graph=True
        )[0]
        R_gap = self.lambda_mono * torch.clamp(d_abs_t, min =0.0).mean(0)

        total = (
             loss_amplitude
             + sign_loss
             + auxilary_stabalizer_loss
             + l2_loss
             R_gap

        )

        return total

model = T1DataProcessor()
X_occ_tensor = torch.tensor(X_occ.values)
X_occ_trainloader = DataLoader(X_occ_tensor, batch_size = 64, shuffle= True)
X_vir_tensor = torch.tensor(X_vir.values)
X_vir_trainloader = DataLoader(X_vir_tensor, batch_size = 64, shuffle= True)
gap_tensor = torch.tensor(gap.values)
gap_trainloader = DataLoader(gap_tensor, batch_size = 64, shuffle= True)
t1_tensor = torch.tensor(t1.values)
t1_trainloader = DataLoader(t1_tensor, batch_size = 64, shuffle= True)

criterion = T1Loss
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epochs = 5
for e in range(epochs):
    running_loss = 0

    for X_occ, X_vir, gap, t1 in zip(X_occ_trainloader, X_vir_trainloader, gap_trainloader, t1_trainloader):
        optimizer.zero_grad()
        output = model(X_occ, X_vir, gap)
        loss = criterion(output, model, 1/gap[2], model)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    else:
        print(f"Training loss: {running_loss/len(X_occ_trainloader)}")
'''

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
        
        return (1.0/ self.beta)* torch.log(0.5 + 0.5*torch.exp(self.beta*x))
    
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
    lr: float = 1e-3
    monotonic_eps: float = 1e-3 # finite-difference ε for Δε
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
        g = g_bilinear + g_hadamard + g_linear

        # ŝ and log|t|
        s_hat = torch.tanh(self.alpha_s * g + torch.einsum('j,bj -> b', self.vs, xi) + self.bs) #(B,)
        log_amp = self.alpha_m*g + torch.einsum('j, bj-> b', self.vm, xi) + self.bm        #(B,)

        inv_delta_e = gap_phi[:, 2] # (B,)
        t_hat = s_hat * torch.exp(torch.clamp(log_amp, -20, 20)) * inv_delta_e

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
        t_true = targets.squeeze(-1) # (B,)

        #1) Amplitude fit (Huber)
        loss_amp = F.huber_loss(t_hat, t_true, delta =self.cfg.huber_delta)

        #2) Sign fit (BCE) with p = (1 + ŝ)/2 and label y_sign = 1[t_true >= 0]
        p = (1.0 + s_hat)*0.5
        p = torch.clamp(p, 1e-6, 1.0 - 1e-6)
        y_sign = (t_true >= 0).float()
        loss_sign = F.binary_cross_entropy(p, y_sign)*self.cfg.lambda_sign

        # --- Debugging prints ---
        print("p shape:", p.shape, "y_sign shape:", y_sign.shape)
        print("p min/max:", p.min().item(), p.max().item())

        #3) Auxilary stabiliser: g ≈ Δε * t_true
        inv_delta_e = gap_phi[:, 2]
        delta_e = 1.0/torch.clamp(inv_delta_e, min = 1e-12)
        y_aux = delta_e * t_true
        loss_aux = F.huber_loss(g, y_aux, delta = self.cfg.huber_delta)*self.cfg.lambda_aux

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

        total = loss_amp + loss_sign + loss_aux+r_gap
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
