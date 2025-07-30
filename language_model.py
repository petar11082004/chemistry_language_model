# import pyscf
from pyscf import gto, scf, cc
from pyscf import lo
from pyscf.tools import cubegen
import numpy as np
import scipy as sp
import py3Dmol
import os
    
class MoleculeFeatureExtractor:

    def __init__(self, mol):
        self.mol = mol

    @staticmethod
    def localize_orbitals_separately(mol, mo_coeff, mo_occ):

        """
        Localize occupied and virtual molecular orbitals separately using Boys method.

        Args:
            mol: pyscf.gto.Mole
                Molecule object.
            mf: pyscf.scf.hf.SCF
                Mean-field (SCF) calculation result.

        Returns:
            numpy.ndarray:
                stacked localized occupied and virtual orbitals.
        """

        #Create boolean masks
        occ_idx = mo_occ > 0
        vir_idx = mo_occ == 0

        # Use masks to select occupied and virtual orbitals
        occupied_orbitals_coeffs = mo_coeff[:, occ_idx]
        virtual_orbitals_coeffs = mo_coeff[:, vir_idx]

        #.pipek.PipekMezey, .edmiston.EdmistonRuedenberg

        localized_occupied_orbitals_method = lo.pipek.PipekMezey(mol, occupied_orbitals_coeffs)
        localized_virtual_orbitals_method = lo.pipek.PipekMezey(mol, virtual_orbitals_coeffs)        

        localized_occupied_orbitals_coeffs = localized_occupied_orbitals_method.kernel()
        localized_virtual_orbitals_coeffs = localized_virtual_orbitals_method.kernel()

        print(f"localized_occupied_orbitals_coeffs shape {localized_occupied_orbitals_coeffs.shape}")
        print(f"localized_virtual_orbitals_coeffs.shape {localized_virtual_orbitals_coeffs.shape}")
        print(f"np.linalg.pinv(occupied_orbitals_coeffs): {np.linalg.pinv(occupied_orbitals_coeffs).shape}")
        print(f"np.linalg.pinv(virtual_orbitals_coeffs){np.linalg.pinv(virtual_orbitals_coeffs).shape}")

        U_occ =  np.linalg.pinv(occupied_orbitals_coeffs) @ localized_occupied_orbitals_coeffs
        U_vir = np.linalg.pinv(virtual_orbitals_coeffs) @ localized_virtual_orbitals_coeffs
        U = sp.linalg.block_diag(U_occ, U_vir)

        loc_mo_coeffs = np.hstack([localized_occupied_orbitals_coeffs, localized_virtual_orbitals_coeffs])

        return loc_mo_coeffs, U

    @staticmethod
    def population_analysis(mol, C_loc, mf):

        """
        Perform Mulliken population analysis on molecular orbitals 
        to determine which atoms each orbital is centered on.
        Args:
            mol: pyscf.gto.Mole
                Molecule object.
            C_loc: numpy.ndarray
                localized molecular orbitals coefficients.
            mf: pyscf.scf.hf.SCF
                Mean-field (SCF) calculation result.

        Returns:
            Tuple([List[str], List[str], List[float]]):
                atoms_0: list of most-contributing atom symbols per orbital.
                atoms_1: list of second-most -contributing atom symbols per orb.
                distances: list of distances between those atom pairs.
        """

        # Get overlap matrix
        S = mf.get_ovlp()

        # Atom and AO info
        n_aos = mol.nao
        n_atoms = mol.natm

        atoms_0 = []
        atoms_1 = []
        distances = []
        
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
            idx0, idx1 = indices[0], indices[1]
            symb_0 = mol.atom_symbol(idx0)
            symb_1 = mol.atom_symbol(idx1)
            coord_0 = mol.atom_coord(idx0)  
            coord_1 = mol.atom_coord(idx1)
            
            atoms_0.append(symb_0)
            atoms_1.append(symb_1)
            distances.append(np.linalg.norm(coord_1 - coord_0))
            
        return atoms_0, atoms_1, distances

    @staticmethod
    def determine_orbital_type(mol, C_loc):

        """
        calculate the expectation values of Lz^2 for the molecular orbitals 
        to determine their character.

        Args:
            mol: pyscf.gto.Mole
                Molecule object.
            C_loc: numpy.ndarray
                localized molecular orbitals coefficients.
            
        Returns:
            list of str:
                labels: Labels of the type of MO (σ, π, δ, or mixed).
        """

        lz_3comp = gto.moleintor.getints('int1e_cg_irxp_sph', mol._atm, mol._bas, mol._env, comp = 3)
        lz_matrix = lz_3comp[2]

        evals, evecs = np.linalg.eigh(lz_matrix)
        maglz = evecs @ np.diag(np.abs(evals)) @ evecs.T

        #lz_squared = lz_matrix.conj().T @ lz_matrix 
        maglz_expect = np.diag(C_loc.conj().T @ maglz @ C_loc).real

        labels = []
        print(f"maglz_expect: {maglz_expect}")
        for maglz_val in maglz_expect:

            if abs(maglz_val) < 0.2:
                label = 'σ'
            elif abs(maglz_val - 1.0) < 0.2:
                label = 'π'
            elif abs(maglz_val - 2.0) < 0.2:
                label = 'δ'
            else:
                label = 'mixed'

            labels.append(label)
        return labels

    @staticmethod
    def calculate_energy(mf, U):

        """
        Calculate the energies of the localized molecular orbitals.

        Args:
            mf: pyscf.scf.hf.SCF
                Mean-field (SCF) calculation result.
            U: np.ndarray
               Unitary matrix used to localize the MOs.
            
        Returns:
            List[float]:
                loc_mo_energies: Energies of the localized MOs.
        """

        mo_energies = mf.mo_energy
        print(f"MO energies: {mo_energies}")
        loc_mo_energies = np.diag(U.conj().T @ np.diag(mo_energies) @ U).real
        return loc_mo_energies
    
    @staticmethod
    def generate_cube_files(C_loc):

        """
        generate cube files so that we can visualise the molecular orbitals

        Args:
            C_loc: np.ndarray
                   localized molecular orbitals' coefficients
        """
        # Create output directory if it doesn't exist
        os.makedirs("cube_files", exist_ok=True)
        for mo_index in range(C_loc.shape[1]):
            coeff_vector = C_loc[:, mo_index]

            cube_filename = os.path.join("cube_files", f'CO2mo{mo_index}.cube')

            cubegen.orbital(mol, cube_filename, coeff_vector, nx=80, margin=3.0)
        
    def extract_molecule_features(self):

        """
        Extract molecule features

        Args:

        Returns:
            Tuple([List[str], List[str], List[float], List[str], List[float]):
                atoms_0: list of most-contributing atom symbols per orbital.
                atoms_1: list of second-most -contributing atom symbols per orb.
                distances: list of distances between those atom pairs.
                labels: Labels of the type of MO (σ, π, δ, or mixed).
                mo_energies: Energies of the localized MOs.
        """
        mf = scf.RHF(self.mol)
        mf.kernel()
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        C_loc, U = MoleculeFeatureExtractor.localize_orbitals_separately(self.mol, mo_coeff, mo_occ)
        atoms_0, atoms_1, distances = MoleculeFeatureExtractor.population_analysis(self.mol, C_loc, mf)
        print(f"atoms 0: {atoms_0}")
        print(f"atoms 1: {atoms_1}")
        print(f"distances: {distances}")
        labels = MoleculeFeatureExtractor.determine_orbital_type(self.mol, C_loc)
        print(labels)
        mo_energies = MoleculeFeatureExtractor.calculate_energy(mf, U)
        MoleculeFeatureExtractor.generate_cube_files(C_loc)

        return atoms_0, atoms_1, distances, labels, mo_energies

mol = gto.Mole()
mol.atom ='''
C2	0.0000	0.0000	0.0000
O3	0.0000	0.0000	1.1621
O1	0.0000	0.0000	-1.1621
'''

mol.unit = 'Angstrom'
mol.basis = 'sto-3g'
mol.charge = 0
mol.spin = 0
mol.build()

c = MoleculeFeatureExtractor(mol)
_,_,_,_,_, = c.extract_molecule_features()
