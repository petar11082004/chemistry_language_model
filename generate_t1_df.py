from pyscf import gto, scf, cc
from molecule_feature_extractor import MoleculeFeatureExtractor
import pandas as pd
import numpy as np

def generate_t1_df(mol: gto.Mole) -> pd.DataFrame:

    """
    Build a DataFrame where each row is an (occupied, virtual) MO pair.

    Assumptions:
      - Closed-shell RHF (mol.spin == 0). Energies in Hartree.
    Returns columns:
      occ_* and vir_* feature columns plus 't1' (CCSD t1 amplitude).
    """

    mf = scf.RHF(mol)
    mf.kernel()
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    n_occ = int((mo_occ>0).sum())

    mfe = MoleculeFeatureExtractor(mol)
    maglz_expect, charges_0, charges_1, charges_2, inv_R_01, inv_R_02, inv_R_12,mo_energies = mfe.extract_molecule_features()

    df_mo = pd.DataFrame({
        "maglz_expect": maglz_expect,
        "charges_0":    charges_0,
        "charges_1":    charges_1,
        "charges_2":    charges_2,
        "inv_R_01":     inv_R_01,
        "inv_R_02":     inv_R_02,
        "inv_R_12":     inv_R_12,
        "mo_energies":  mo_energies,
    })

    occ_df = df_mo.iloc[:n_occ].copy()
    vir_df = df_mo.iloc[n_occ:].copy()

    occ_df.insert(0, "occ_idx", np.arange(len(occ_df)))
    vir_df.insert(0, "vir_idx", np.arange(len(vir_df)))

    occ_df = occ_df.add_prefix("occ_")
    vir_df = vir_df.add_prefix("vir_")

    occ_df["_key"] = 1
    vir_df["_key"] = 1
    pairs = occ_df.merge(vir_df, on = '_key').drop(columns = '_key')

    desired = [
        "occ_maglz_expect","occ_charges_0","occ_charges_1","occ_charges_2",
        "occ_inv_R_01","occ_inv_R_02","occ_inv_R_12","occ_mo_energies",
        "vir_maglz_expect","vir_charges_0","vir_charges_1","vir_charges_2",
        "vir_inv_R_01","vir_inv_R_02","vir_inv_R_12","vir_mo_energies",
    ]

    # keep indices too (first), if you want them; drop if you don't
    front = [c for c in ("occ_idx", "vir_idx") if c in pairs.columns]
    pairs = pairs[front + desired]

    mo_occ = mf.mo_occ
    C_loc, _ = MoleculeFeatureExtractor.localize_orbitals_separately(mol, mo_coeff, mo_occ)
    mycc = cc.ccsd.CCSD(mf, mo_coeff=C_loc).run()
    t1 = mycc.t1.flatten()
    pairs['t1'] = t1



    return pairs
