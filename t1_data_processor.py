import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

class T1DataProcessor:
    """
    Preprocess molecular orbital features for predicting t1 amplitudes.

    Decisions encoded:
    - occ/vir energies: standardized (z-score).
    - inverse distances (inv_R_*): masked standardization (zeros = padding).
    - maglz_expect: kept raw (not transformed).
    - charges_*: kept raw (categorical integers with PAD=0).
    - Δε block: adds inverse gap (1/Δε) and log gap (log Δε),
                then standardizes both.

    Follows a fit/transform API (like sklearn):
    - fit(df): compute means/stds on training data (store internally).
    - transform(df): apply those same statistics to any dataframe
                     (train/val/test) to avoid leakage.
    """

    def __init__(self, eps=1e-8):
        """
        eps: small number to avoid division by zero or log(0).
        """
        self.eps = eps
        self.means = {}
        self.stds = {}

    def fit(self, df: pd.DataFrame):
        """
        Compute and store column-wise means/stds on training dataframe.
        These statistics will be reused for all future transformations.
        """

        # === Energies: always standardized ===
        for col in ["occ_mo_energies", "vir_mo_energies"]:
            self.means[col] = df[col].mean()
            self.stds[col] = df[col].std() + self.eps

        # === Inverse distances: masked standardization ===
        inv_cols = [c for c in df.columns if "inv_R" in c]
        for col in inv_cols:
            mask = df[col] != 0  # 0 means "absent" → skip
            self.means[col] = df.loc[mask, col].mean()
            self.stds[col] = df.loc[mask, col].std() + self.eps

        # === Gap features (Δε = ε_vir - ε_occ) ===
        delta_e = df["vir_mo_energies"] - df["occ_mo_energies"]
        inv_delta_e = 1.0 / (delta_e + self.eps)
        log_delta_e = np.log(delta_e + self.eps)

        self.means["inv_delta_e"] = inv_delta_e.mean()
        self.stds["inv_delta_e"]  = inv_delta_e.std() + self.eps
        self.means["log_delta_e"] = log_delta_e.mean()
        self.stds["log_delta_e"]  = log_delta_e.std() + self.eps

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply stored normalization statistics to a dataframe.
        Returns a copy of df with standardized energies/distances/gap features.
        """
        out = df.copy()

        # === Standardize energies ===
        for col in ["occ_mo_energies", "vir_mo_energies"]:
            out[col] = (df[col] - self.means[col]) / self.stds[col]

        # === Masked standardization for inverse distances ===
        inv_cols = [c for c in df.columns if "inv_R" in c]
        for col in inv_cols:
            mask = df[col] != 0
            out.loc[mask, col] = (df.loc[mask, col] - self.means[col]) / self.stds[col]
            # keep zeros as-is (padding)

        # === Recompute Δε features, then standardize ===
        delta_e = df["vir_mo_energies"] - df["occ_mo_energies"]
        inv_delta_e = 1.0 / (delta_e + self.eps)
        log_delta_e = np.log(delta_e + self.eps)

        out["inv_delta_e"] = (inv_delta_e - self.means["inv_delta_e"]) / self.stds["inv_delta_e"]
        out["log_delta_e"] = (log_delta_e - self.means["log_delta_e"]) / self.stds["log_delta_e"]

        return out
