from pyscf import gto
from generate_t1_df import generate_t1_df
import pandas as pd
from datetime import datetime

f2_molecules= [
    """
    F	0.0000	0.0000	0.0000
    F	0.0000	0.0000	1.5531
    """,
    """
    F	0.0000	0.0000	0.0000
    F	0.0000	0.0000	1.5178
    """,
    """
    F	0.0000	0.0000	0.0000
    F	0.0000	0.0000	1.4825
    """,
    """
    F	0.0000	0.0000	0.0000
    F	0.0000	0.0000	1.4472
    """,
    """
    F	0.0000	0.0000	0.0000
    F	0.0000	0.0000	1.4119
    """,
    """
    F	0.0000	0.0000	0.0000
    F	0.0000	0.0000	1.3766
    """,
    """
    F	0.0000	0.0000	0.0000
    F	0.0000	0.0000	1.3413
    """,
    """
    F	0.0000	0.0000	0.0000
    F	0.0000	0.0000	1.3060
    """,
    """
    F	0.0000	0.0000	0.0000
    F	0.0000	0.0000	1.2707
    """,
]
co2_molecules = [
    """
    C1	0.0000	0.0000	0.0000
    O2	0.0000	0.0000	1.2783
    O3	0.0000	0.0000	-1.1621
    """,
    """
    C1	0.0000	0.0000	0.0000
    O2	0.0000	0.0000	1.2493
    O3	0.0000	0.0000	-1.1621
    """,
    """
    C1	0.0000	0.0000	0.0000
    O2	0.0000	0.0000	1.2202
    O3	0.0000	0.0000	-1.1621
    """,
    """
    C1	0.0000	0.0000	0.0000
    O2	0.0000	0.0000	1.1916
    O3	0.0000	0.0000	-1.1621
    """,
    """
    C1	0.0000	0.0000	0.0000
    O2	0.0000	0.0000	1.1621
    O3	0.0000	0.0000	-1.1621
    """,
    """
    C1	0.0000	0.0000	0.0000
    O2	0.0000	0.0000	1.1330
    O3	0.0000	0.0000	-1.1621
    """,
    """
    C1	0.0000	0.0000	0.0000
    O2	0.0000	0.0000	1.1040
    O3	0.0000	0.0000	-1.1621
    """,
    """
    C1	0.0000	0.0000	0.0000
    O2	0.0000	0.0000	1.0749
    O3	0.0000	0.0000	-1.1621
    """,
    """
    C1	0.0000	0.0000	0.0000
    O2	0.0000	0.0000	1.0459
    O3	0.0000	0.0000	-1.1621
    """,
]

hf_molecules = [
    """
    F1	0.0000	0.0000	0.0000
    H2	0.0000	0.0000	1.0085
    """,
    """
    F1	0.0000	0.0000	0.0000
    H2	0.0000	0.0000	0.9856
    """,
    """
    F1	0.0000	0.0000	0.0000
    H2	0.0000	0.0000	0.9626
    """,
    """
    F1	0.0000	0.0000	0.0000
    H2	0.0000	0.0000	0.9397
    """,
    """
    F1	0.0000	0.0000	0.0000
    H2	0.0000	0.0000	0.9168
    """,
    """
    F1	0.0000	0.0000	0.0000
    H2	0.0000	0.0000	0.8939
    """,
    """
    F1	0.0000	0.0000	0.0000
    H2	0.0000	0.0000	0.8710
    """,
    """
    F1	0.0000	0.0000	0.0000
    H2	0.0000	0.0000	0.8480
    """,
    """
    F1	0.0000	0.0000	0.0000
    H2	0.0000	0.0000	0.8251
    """,

]


def build_mol(geom: str, *, charge: int = 0, spin: int | None = None, basis: str = "sto-3g", unit: str = "Angstrom") -> gto.Mole:

    mol = gto.Mole()
    mol.atom = geom
    mol.unit = unit
    mol.basis = basis
    mol.charge = charge
    if spin is not None:
        mol.spin = spin
    mol.build()
    return mol

def dfs_from_geoms(geoms: list[str], *, 
                   charges: list[int]| None = None,
                   spins: list[int] | None = None,
                   basis: str = 'sto-3g',
                   unit: str = "Angstrom") -> pd.DataFrame:
    
    dfs = []
    for i, geom in enumerate(geoms):
        q = 0 
        s = 0
        mol = build_mol(geom, charge = q, spin = s, basis = basis, unit = unit)
        df = generate_t1_df(mol)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index = True)

df = dfs_from_geoms(f2_molecules)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
f2_parquet = f"./t1_pairs_f2_{stamp}.parquet"

try:
    df.to_parquet(f2_parquet, index = False)
    print(f"Saved Parquet:\n {f2_parquet}\n ")
except Exception as e:
     print(f"[warn] Parquet save failed ({e}). Install 'pyarrow' or 'fastparquet' to enable Parquet export.")