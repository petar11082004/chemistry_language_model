import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('molecule_feature_extractor.py')
sys.path.append('/home/pp583/revqcmagic')
from qcmagic.core.sspace.statespace import positivify_coeffmats
from molecule_feature_extractor import MoleculeFeatureExtractor
df = pd.read_parquet('t1_trends/t1_pairs_hf_20250829_152147.parquet')

bond_lengths = [1.1, 1.075, 1.05, 1.025, 1.00, 0.975, 0.95, 0.925, 0.9]
t1 = np.abs(df['t1'])
t1 = np.array(t1).reshape(9, int(len(t1)/9))

''' 
plt.figure(figsize = (8,6))
for i in range(t1.shape[1]):
    plt.plot(bond_lengths, t1[:, i])


plt.title(r"$F_2$"+ " second plot")
plt.xlabel("bond length multiplier")
plt.ylabel("t1")
plt.show()
'''

import numpy as np
from pyscf import gto, scf, cc
from pyscf.scf import addons  # for project_dm_nr2nr
import matplotlib.pyplot as plt

def make_diatomic(atom1, atom2, R, basis="sto-3g"):
    return gto.M(
        atom = f"""
            {atom1} 0.0 0.0 0.0
            {atom2} 0.0 0.0 {R}
        """,
        unit = "Angstrom",
        basis = basis,
    )

R0 = 1.4119
bond_lengths = np.linspace(0.9*R0, 1.1*R0, 9)

t1_list = []
C_list, L_list, U_list = [], [], [] 

for R in bond_lengths:
    mol = make_diatomic("F", "F", R)
    mf = scf.RHF(mol).run()

    # CCSD and |t1|
    mycc = cc.ccsd.CCSD(mf).run()
    t1 = mycc.t1

    L, U = MoleculeFeatureExtractor.localize_orbitals_separately(mol, mf.mo_coeff, mf.mo_occ)
    n_occ = sum(mf.mo_occ > 0)
    L_df = pd.DataFrame(L)
    print('########### L ############')
    print(L_df)
    print('\n')
    # --- Store results ---
    C = mf.mo_coeff.copy()
    positivify_coeffmats([C])
    C_list.append(C.flatten())  
    L_list.append(L.flatten())            
    U_list.append(U.flatten())            

    # |t1|
    t1 = np.abs(t1.flatten())
    t1_list.append(t1)

# Convert to arrays for plotting
t1_list = np.array(t1_list)
C_list = np.array(C_list)   # shape (n_bonds, n_AO*n_MO)
L_list = np.array(L_list)
U_list = np.array(U_list)

# --- Plot a few representative entries ---
plt.figure(figsize=(10,6))
for i in range(C_list.shape[1]-5, C_list.shape[1]):   # plot first 5 elements
    plt.plot(bond_lengths, C_list[:, i], label=f"C[{i}]")
plt.xlabel("bond length")
plt.ylabel("C coefficients")
plt.title("Selected canonical MO coefficients vs bond length")
plt.legend()
plt.show()

my_list = np.arange(18,98,10)
plt.figure(figsize=(10,6))
for i in my_list:   # plot first 5 elements
    plt.plot(bond_lengths, np.abs(L_list[:, i]), label=f"L[{i}]")
plt.xlabel("bond length")
plt.ylabel("L coefficients")
plt.title("Selected localized MO coefficients vs bond length")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(min(5, U_list.shape[1])):   # plot first 5 entries of U
    plt.plot(bond_lengths, U_list[:, i], label=f"U[{i}]")
plt.xlabel("bond length")
plt.ylabel("U entries")
plt.title("Selected orbital rotation matrix entries vs bond length")
plt.legend()
plt.show()