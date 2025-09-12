from pyscf import gto, scf
from molecule_feature_extractor import MoleculeFeatureExtractor

# Define the methane molecule
mol = gto.Mole()
mol.atom = '''
C1    0.000   0.000   0.000
H2    0.000   0.000   1.090
H3    1.029   0.000  -0.363
H4   -0.514   0.891  -0.363
H5   -0.514  -0.891  -0.363
'''
mol.basis = 'sto-3g'   # you can also use '6-31g', 'cc-pvdz', etc.
mol.unit = 'Angstrom'
mol.build()

# Run Hartree-Fock
mf = scf.RHF(mol)
mf.kernel()
mo_occ = mf.mo_occ

C_loc, _ = MoleculeFeatureExtractor.localize_orbitals_separately(mol, mf.mo_coeff, mo_occ)
rot_C_loc = MoleculeFeatureExtractor.rotate_orbitals(mol, C_loc, mf)
