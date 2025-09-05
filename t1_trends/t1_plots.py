import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('molecule_feature_extractor.py')
sys.path.append('/home/pp583/revqcmagic')
from qcmagic.core.sspace.statespace import positivify_coeffmats
from molecule_feature_extractor import MoleculeFeatureExtractor
df = pd.read_parquet('t1_trends/t1_pairs_hf_20250829_152147.parquet')
import numpy as np
from pyscf import gto, scf, cc
from pyscf.scf import addons  # for project_dm_nr2nr
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def permute_orbitals(L_prev,S, L):
    O = L_prev.T @ S @ L
    r,c = linear_sum_assignment(-np.abs(O))
    perm = np.empty_like(c)
    perm[r] = c
    L_aln = L[:, perm].copy()
    return L_aln

def make_fiveatomic(atom1, atom2, atom3, atom4, atom5, R, basis="sto-3g"):
    return gto.M(
        atom = f"""
            {atom1} 0.0 0.0 0.0
            {atom2} 0.0 0.0 {R}
            {atom3} 1.0250   0.0000  -0.3623
            {atom4} -0.5125   0.8877  -0.3623
            {atom5} -0.5125  -0.8877  -0.3623
        """,
        unit = "Angstrom",
        basis = basis,
    )

def make_hf(R, basis = "sto-3g"):
    return gto.M(
        atom = f"""
        F1	0.0000	0.0000	0.0000
        H2	0.0000	0.0000	{R}
        """,
        unit = "Angstrom",
        basis = basis,
    )

def make_f2(R, basis = "sto-3g"):
    return gto.M(
        atom = f"""
        F1	0.0000	0.0000	0.0000
        F2	0.0000	0.0000	{R}
        """,
        unit = "Angstrom",
        basis = basis,
    )

def make_hcn(R, basis = "sto-3g"):
    return gto.M(
        atom = f"""
        C1	0.0000	0.0000	0.0000
        H2	0.0000	0.0000	{R}
        N3	0.0000	0.0000	-1.1560
        """,
        unit = "Angstrom",
        basis = basis,
    )

R0 = 1.0640
bond_lengths = np.linspace(0.9*R0, 1.1*R0, 9)

t1_list = []
C_list, L_list, U_list = [], [], [] 

L_prev = None
for R in bond_lengths:
    mol = make_fiveatomic('C', 'H1', 'H2', 'H3', 'H4', R)
    mol = make_hcn(R)
    mf = scf.RHF(mol).run()

    L, U = MoleculeFeatureExtractor.localize_orbitals_separately(mol, mf.mo_coeff, mf.mo_occ, L_prev)
    L_prev = L

    n_occ = sum(mf.mo_occ > 0)

    U_occ = U[:n_occ, :n_occ].copy()
    U_vir = U[n_occ:, n_occ:].copy()

    mycc = cc.ccsd.CCSD(mf).run()
    t1 = mycc.t1

    t1 = U_occ.T @ t1 @ U_vir
    L_df = pd.DataFrame(L)
    U_df = pd.DataFrame(U)
    
    ####################################################################################################
    mfe = MoleculeFeatureExtractor(mol)
    maglz_expect, atoms_0, atoms_1, atoms_2, charges_0, charges_1, charges_2, inv_R_01, inv_R_02, inv_R_12,mo_energies = mfe.extract_molecule_features()
    #####################################################################################################

    df_mo = pd.DataFrame({
        "maglz_expect": maglz_expect,
        "atoms_0": atoms_0,
        "atoms_1":atoms_1,
        "atoms_2": atoms_2,
        "charges_0":    charges_0,
        "charges_1":    charges_1,
        "charges_2":    charges_2,
        "inv_R_01":     inv_R_01,
        "inv_R_02":     inv_R_02,
        "inv_R_12":     inv_R_12,
        "mo_energies":  mo_energies,
    })
    print('\n')
    print('##########  MULTIPLIER  ########## \n')
    print(R/R0)
    print('\n')
    print(df_mo)

    #print(L_df)
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



plt.figure(figsize=(10,6))
for i in range(min(5, t1_list.shape[1])):   # plot first 5 entries of U
    plt.plot(bond_lengths/R0, np.abs(t1_list[:, i]), label=f"t1[{i}]")
plt.xlabel("bond length")
plt.ylabel("t1 entries")
plt.title("Selected t1 amplitude entries vs bond length")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(C_list.shape[1]-5, C_list.shape[1]):   # plot first 5 elements
    plt.plot(bond_lengths/R0, C_list[:, i], label=f"C[{i}]")
plt.xlabel("bond length")
plt.ylabel("C coefficients")
plt.title("Selected canonical MO coefficients vs bond length")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(min(5, L_list.shape[1])):  # plot first 5 elements
    plt.plot(bond_lengths/R0, np.abs(L_list[:, i]), label=f"L[{i}]")
plt.xlabel("bond length")
plt.ylabel("L coefficients")
plt.title("Selected localized MO coefficients vs bond length")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for i in range(min(5, U_list.shape[1])):   # plot first 5 entries of U
    plt.plot(bond_lengths/R0, U_list[:, i], label=f"U[{i}]")
plt.xlabel("bond length")
plt.ylabel("U entries")
plt.title("Selected orbital rotation matrix entries vs bond length")
plt.legend()
plt.show()