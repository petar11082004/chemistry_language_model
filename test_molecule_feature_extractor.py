import unittest
import numpy as np
from pyscf import gto, scf
from language_model import MoleculeFeatureExtractor
from types import SimpleNamespace
import sys
sys.path.append('/home/pp583/revqcmagic')
from pyscf.scf import addons
from qcmagic.auxiliary.linearalgebra3d import Vector3D
from qcmagic.core.drivers.statetools.rotate import rotate_state
from qcmagic.interfaces.converters.pyscf import scf_to_state, configuration_to_mol

class TestMoleculeFeatureExtractor(unittest.TestCase):

    def setUp(self):
        """Setup a test molecule and run SCF."""
        self.mol = gto.Mole()
        self.mol.atom = '''
        C2 0.0000 0.0000 0.0000
        O3 0.0000 0.0000 1.1621
        O1 0.0000 0.0000 -1.1621
        '''
        self.mol.unit = 'Angstrom'
        self.mol.basis = 'sto-3g'
        self.mol.charge = 0
        self.mol.spin = 0
        self.mol.build()

        self.mf = scf.RHF(self.mol)
        self.e_pyscf = self.mf.kernel()
        self.mo_coeff = self.mf.mo_coeff
        self.mo_occ = self.mf.mo_occ

        self.C_loc, self.U = MoleculeFeatureExtractor.localize_orbitals_separately(self.mol, self.mo_coeff, self.mo_occ)
    
    def test_rotate_orbitals(self):

        rot_C_loc = MoleculeFeatureExtractor.rotate_orbitals(self.mol, self.C_loc, self.mf)

        fro_norm_before = np.linalg.norm(self.C_loc, ord='fro')
        fro_norm_after = np.linalg.norm(rot_C_loc, ord='fro')
        assert np.isclose(fro_norm_before, fro_norm_after, atol=1e-8)

    def test_weighted_sum_energy(self):
        """Test energy calculation for a known linear combination of orbitals"""

        # Mock mean-field object with mo_energy as an attribute
        mf = SimpleNamespace(mo_energy = np.array([1.0, 3.0]))

        # Construct a unitary matrix with a known linear combination
        c1 = np.sqrt(0.25)
        c2 = np.sqrt(0.75)

        # a 2x2 unitary transformation mixing two orbitals
        U = np.array([
            [c1, -c2],
            [c2, c1]
        ])

        #Expected energies after transformation
        expected_0 = c1**2 * mf.mo_energy[0] + c2**2 * mf.mo_energy[1] # = 2.5
        expected_1 = c2**2 * mf.mo_energy[0] + c1**2 * mf.mo_energy[1] # = 1.5

        # Run the function
        energies = MoleculeFeatureExtractor.calculate_energy(mf, U)

        # Assertions
        assert len(energies) == 2
        np.testing.assert_almost_equal(energies[0], expected_0, decimal = 6)
        np.testing.assert_almost_equal(energies[1], expected_1, decimal=6)



    def test_calculate_energy_dimensions(self):
        """Test if energy array length matches number of localized orbitals."""
        energies = MoleculeFeatureExtractor.calculate_energy(self.mf, self.U)
        self.assertEqual(len(energies), self.C_loc.shape[1], "Energy list should match number of localized orbitals")
    
    def test_calculate_energy_type_values(self):
        '''Test if localized energies are real numbers'''
        energies = MoleculeFeatureExtractor.calculate_energy(self.mf, self.U)
        for e in energies:
            self.assertIsInstance(e, float)
        self.assertFalse(np.isnan(energies).any(), "Energies should not be NaN")
        self.assertFalse(np.iscomplexobj(energies), "Energies should be real_valued")

    def test_U_is_unitary(self):
        """Test if U is approximately unitary."""
        U = self.U
        identity = np.eye(U.shape[1])
        product = U.conj().T @ U
        np.testing.assert_allclose(product, identity, atol = 1e-6, err_msg = "U should be unitary")

if __name__ == '__main__':
    unittest.main()