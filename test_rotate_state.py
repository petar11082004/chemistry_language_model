from pyscf import gto, scf
from molecule_feature_extractor import MoleculeFeatureExtractor

# Define the methane molecule
mol = gto.Mole()
mol.atom = '''
C1	0.0000	0.0000	0.0000
H2	0.6276	0.6276	0.6276
H3	0.6276	-0.6276	-0.6276
H4	-0.6276	0.6276	-0.6276
H5	-0.6276	-0.6276	0.6276
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
