#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pyscf')


# In[1]:


def loc_orbitals_separately(mol, mo_coeff, mo_occ):

    """
    Localize occupied and virtual molecular orbitals separately using the Boys method.

    Args:
        mol: pyscf.gto.Mole
            Molecule object.
        mf: pyscf.scf.hf.SCF
            Mean-field (SCF) calculation result.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: 
            Localized occupied and virtual orbital coefficients.
    """
    
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    
    # Create boolean masks
    occ_idx = mo_occ > 0
    vir_idx = mo_occ == 0
    
    # Use masks to select occupied and virtual orbitals
    occupied_orbitals_coeffs = mo_coeff[:, occ_idx]
    virtual_orbitals_coeffs = mo_coeff[:, vir_idx]
    
    localized_occupied_orbitals_method = lo.Boys(mol, occupied_orbitals_coeffs)
    localized_virtual_orbitals_method = lo.Boys(mol,  virtual_orbitals_coeffs)
    
    localized_occupied_orbitals_coeffs = localized_occupied_orbitals_method.kernel()
    localized_virtual_orbitals_coeffs = localized_virtual_orbitals_method.kernel()

    print(localized_occupied_orbitals_coeffs.shape)
    print(localized_virtual_orbitals_coeffs.shape)

    return localized_occupied_orbitals_coeffs, localized_virtual_orbitals_coeffs


# In[2]:


from pyscf import gto, scf, cc
import numpy as np

mol = gto.Mole()
mol.atom = '''
H  0.0000  0.0000  -2.261
C  0.0000  0.0000  -1.203
C  0.0000  0.0000   1.203
H  0.0000  0.0000   2.261
'''
mol.basis = 'sto-3g'
mol.charge = 0
mol.spin = 0
mol.build()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

# Step 2: Perform RHF
mf = scf.RHF(mol)
mf.kernel()

# Step 3: Run CCSD
mycc = cc.CCSD(mf)
mycc.kernel()

# Step 4: Access amplitudes
t1 = mycc.t1  # Shape: (nocc, nvir)
t2 = mycc.t2  # Shape: (nocc, nocc, nvir, nvir)

print(t1.shape)
print("T2 amplitudes:")
print(t2.shape)

# Shape: (n_basis, n_mo)
mo_coeff = mf.mo_coeff
mo_occ = mf.mo_occ

# Create boolean masks
occ_idx = mo_occ > 0
vir_idx = mo_occ == 0

# Use masks to select occupied and virtual orbitals
occupied_orbitals_coeffs = mo_coeff[:, occ_idx]
virtual_orbitals_coeffs = mo_coeff[:, vir_idx]

print("Occupied orbitals shape:", occupied_orbitals_coeffs.shape)
print("Virtual orbitals shape:", virtual_orbitals_coeffs.shape)

from pyscf import lo
localized_occupied_orbitals_method = lo.Boys(mol, occupied_orbitals_coeffs)
localized_virtual_orbitals_method = lo.Boys(mol, virtual_orbitals_coeffs)

localized_occupied_orbitals_coeffs = localized_occupied_orbitals_method.kernel()
localized_virtual_orbitals_coeffs = localized_virtual_orbitals_method.kernel()
localized_occupied_orbitals_coeffs.shape

loc_occ_coeffs, loc_vir_coeffs = loc_orbitals_separately(mol, mf.mo_coeff, mf.mo_occ)


# In[3]:


loc_mo_coeffs = np.hstack([loc_occ_coeffs, loc_vir_coeffs])


# In[4]:


occupancy = np.sum(loc_mo_coeffs**2, axis = 0)


# In[5]:


"""
import numpy as np
from pyscf import gto, scf, lo

# Setup molecule
mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "sto-3g"
mol.build()

mf = scf.RHF(mol).run()

# Localize occupied orbitals
C_occ = mf.mo_coeff[:, mf.mo_occ > 0]
C_loc = lo.Boys(mol, C_occ).kernel()

# Get AO overlap matrix
S = mf.get_ovlp()

# Get number of AOs and atoms
n_aos = mol.nao
n_atoms = mol.natm

print(n_aos)

# For each localized orbital, compute population per atom
for i in range(C_loc.shape[1]):
    c = C_loc[:, i]  # i-th localized MO
    pop_per_atom = np.zeros(n_atoms)

    for A in range(n_atoms):
        ao_indices = mol.aoslice_by_atom()[A]
        p0, p1 = ao_indices[2], ao_indices[3]
        for mu in range(p0, p1):
            for nu in range(n_aos):
                pop_per_atom[A] += c[mu] * c[nu] * S[mu, nu]

    print(f"Localized orbital {i}:")
    for A in range(n_atoms):
        symb = mol.atom_symbol(A)
        print(f"  Atom {A} ({symb}): {pop_per_atom[A]:.4f} electrons")
"""
import numpy as np
from pyscf import gto, scf, lo

# Define CO2 molecule aligned along z-axis
mol = gto.Mole()
mol.atom = '''
O1 0.0 0.0 -1.16
C1 0.0 0.0  0.00
O2 0.0 0.0  1.16
'''
mol.unit = 'Bohr'
mol.basis = "sto-3g"
mol.build()

# Run RHF calculation
mf = scf.RHF(mol).run()

# Use all MOs (not just occupied)
C_all = mf.mo_coeff

# Optionally localize all MOs (not just occupied ones)
# Note: localization of virtuals is less well-defined
C_loc = lo.Boys(mol, C_all).kernel()

# Get overlap matrix
S = mf.get_ovlp()

# Atom and AO info
n_aos = mol.nao
n_atoms = mol.natm

print(f"Number of AOs: {n_aos}")
print(f"Number of atoms: {n_atoms}")
print(f"Number of MOs (localized): {C_loc.shape[1]}\n")


atoms_1 = []
atoms_2 = []
distances = []
# Population analysis per atom for each localized MO
for i in range(C_loc.shape[1]):
            c = C_loc[:, i] # i-th localized orbital
            pop_per_atom = np.zeros(n_atoms)

            for A in range(n_atoms):
                ao_slice = mol.aoslice_by_atom()[A]
                p0, p1 = ao_slice[2], ao_slice[3]

                for mu in range(p0, p1):
                    for nu in range(n_aos):
                        pop_per_atom[A] += c[mu] * c[nu] * S[mu, nu]

            indices = pop_per_atom.argsort()[-2:][::-1]
            symb_1 = mol.atom_symbol(indices[0])
            symb_2 = mol.atom_symbol(indices[1])
            coord_0 = mol.atom_coord(0)  # e.g., O
            coord_1 = mol.atom_coord(1)
            print(coord_0)
            atoms_1.append(symb_1)
            atoms_2.append(symb_2)
            distances.append(np.linalg.norm(coord_1 - coord_0))

print(atoms_1)
print(atoms_2)
print(distances)


# In[7]:


# Compute angular momentum operator integrals: r × p
lz_3comp = gto.moleintor.getints('int1e_cg_irxp_sph',
                             mol._atm, mol._bas, mol._env,
                             comp=3)

# Extract only the z-component (component 2 of 3)
lz_matrix = lz_3comp[2]
print(mol.nao)


# In[8]:


l_2 = lz_matrix @ lz_matrix

import pandas as pd
df = pd.DataFrame(l_2)

df = pd.DataFrame(lz_matrix)


# In[9]:


Lz_mo = loc_mo_coeffs.conj().T @ l_2 @ loc_mo_coeffs  # (n_mo, n_mo)
lz_expect= np.diag(Lz_mo).real

# 6. Classify orbitals
print("Localized Orbital Angular Momentum Classification:\n")
for i, lz in enumerate(lz_expect):
    if abs(lz) < 0.1:
        label = 'sigma (σ)'
    elif abs(lz - 1.0) < 0.1 or abs(lz + 1.0) < 0.1:
        label = 'pi (π)'
    elif abs(lz - 4.0) < 0.1 or abs(lz + 4.0) < 0.1:
        label = 'delta (δ)'
    else:
        label = 'other / mixed'
    print(f"LMO {i + 1:2d}: <Lz> = {lz:+.3f} -> {label}")


# In[ ]:


from pyscf import gto
import numpy as np

# Build a molecule
mol = gto.Mole()
mol.atom = '''
O 0.0 0.0 0.0
C 0.0 0.0 1.16
'''
mol.unit = 'Bohr'  # Optional: use Angstroms instead of Bohr
mol.build()

# Get coordinates of atom 0 and atom 1
coord_0 = mol.atom_coord(0)  # e.g., O
coord_1 = mol.atom_coord(1)  # e.g., C
print(coord_1)
# Compute distance (Euclidean norm)
distance = np.linalg.norm(coord_0 - coord_1)

print(f"Distance between atom 0 ({mol.atom_symbol(0)}) and atom 1 ({mol.atom_symbol(1)}): {distance:.4f} {mol.unit}")


# In[ ]:





# In[ ]:




