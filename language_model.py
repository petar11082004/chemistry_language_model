# import pyscf
import sys
sys.path.append('/home/pp583/revqcmagic')
from pyscf import gto, scf, cc
from pyscf import lo
from pyscf.tools import cubegen
import numpy as np
import scipy as sp
import py3Dmol
import os
from qcmagic.core.drivers.statetools.rotate import rotate_coeffs
from qcmagic.auxiliary.linearalgebra3d import Vector3D
from qcmagic.core.drivers.statetools.rotate import rotate_state
from qcmagic.interfaces.converters.pyscf import scf_to_state, configuration_to_mol
from qcmagic.core.cspace.basis.basisshell import BasisType

    
class MoleculeFeatureExtractor:

    def __init__(self, mol):
        self.mol = mol
    
    @staticmethod
    def localize_orbitals_separately(mol, mo_coeff, mo_occ):

        """
        Localize occupied and virtual molecular orbitals separately using PipekMezey method.

        Args:
            mol: pyscf.gto.Mole
                Molecule object.
            mo_coeff: numpy.ndarray
                MO coefficients
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

        localized_occupied_orbitals_method = lo.edmiston.EdmistonRuedenberg(mol, occupied_orbitals_coeffs)
        localized_virtual_orbitals_method = lo.edmiston.EdmistonRuedenberg(mol, virtual_orbitals_coeffs)        

        localized_occupied_orbitals_coeffs = localized_occupied_orbitals_method.kernel()
        localized_virtual_orbitals_coeffs = localized_virtual_orbitals_method.kernel()

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
            List([List[int]):
                indices_list: list with the indexes of atoms on which the orbitals are mainly cenetered on
        """

        # Get overlap matrix
        S = mf.get_ovlp()

        # Atom and AO info
        n_aos = mol.nao
        n_atoms = mol.natm

        indices_list = []
        
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
            indices_list.append(indices)

            print('##### pop per atom #####')
            print(pop_per_atom)
            print('\n')

        print('##### indices #####')
        print(indices_list)
        print('\n')
        return indices_list

    @staticmethod
    def find_distances_and_atoms_on_which_MOs_are_centered(mol, indices_list):

        """
        Find the atoms (and their respective distances) on which the molecular orbitals are primarily centered.

        Args:
            indices_list (List[List[int]]): 
                A list of lists, where each sublist contains the indices of atoms that a 
                localized molecular orbital is centered on, sorted by population percentage.

        Returns:
            Tuple[List[str], List[str], List[float]]: 
                - atoms_0: List of atoms on which each MO is most strongly centered.
                - atoms_1: List of atoms on which each MO is second most strongly centered.
                - distances: Euclidean distance between each respective pair in atoms_0 and atoms_1.
        """
        
        atoms_0 = []
        atoms_1 = []
        distances =[]

        for indices in indices_list:
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
    def find_mo_orientation_vectors(indices_list):

        """
        Find the orientations (vectors) of the localized molecular orbitals

        Args:
            indices_list: List(List[int])
                The indexes of the atoms on which the localized molecular orbital is centered on, sorted by population percentage

        Returns:
            List[List[float]]:
                mo_orientation_vectors: vectors describing the orientation of the localized MOs
        """

        mo_orientation_vectors = []

        for indices in indices_list:
            idx0, idx1 = indices[0], indices[1]
            coord_0 = mol.atom_coord(idx0)  
            coord_1 = mol.atom_coord(idx1)
            mo_orientation_vectors.append(coord_1 - coord_0)
        
        return mo_orientation_vectors

    @staticmethod
    def find_mo_rotation_angles(mo_orientation_vectors):

        """
        Find angle of rotation for the MOs to be aligned with the z-axis

        Args:
            mo_orientation_vectors: List[List[float]]
                vectors describing the orientation of the localized MOs

        Returns:
            List[float]:
                angles: The rotation angles for each MO
        """

        angles = []

        for vector in mo_orientation_vectors:
            angle = np.arccos(np.dot(np.array([0,0,1]), vector)/np.linalg.norm(vector))
            angles.append(angle)
        
        return angles

    @staticmethod
    def rotate_orbitals(mol, C_loc, mf):

        """
        Rotate molecular orbital (MO) coefficients so that the localized orbitals align along the z-axis.

        Args:
            mol: pyscf.gto.Mole
                The molecule object containing atomic and basis set information.
            C_loc: numpy.ndarray
                Localized molecular orbital coefficient matrix.
            mf: pyscf.scf.hf.SCF
                Mean-field (SCF) calculation object containing orbital information.

        Returns:
            numpy.ndarray:
                rot_C_loc: The rotated molecular orbital coefficient matrix, aligned along the z-axis.
        """

        indices_list = MoleculeFeatureExtractor.population_analysis(mol, C_loc, mf)
        vectors = MoleculeFeatureExtractor.find_mo_orientation_vectors(indices_list)
        angles = MoleculeFeatureExtractor.find_mo_rotation_angles(vectors)

        rot_C_loc = []

        for i, angle in enumerate(angles):
            
            mf = scf.RHF(mol)
            mf.mo_coeff = C_loc
            state = scf_to_state(mf)
            config = state.configuration
            state_rot = rotate_state(state, angle, Vector3D([0,0,1]))
    
            cbs=config.get_subconfiguration("ConvolvedBasisSet")
            coeffs_rot_pyscf = cbs.convert_coefficient_matrices(
                                state_rot.coefficients, format_from=BasisType.BT_LIBINT, format_to=BasisType.BT_PYSCF)

            rot_C_loc.append(coeffs_rot_pyscf[0][:, i])
                
        return  np.array(rot_C_loc)

    @staticmethod
    def calculate_mag_lz(mol, C_loc):

        """
        calculate the expectation values of |Lz| for the molecular orbitals 

        Args:
            mol: pyscf.gto.Mole
                Molecule object.
            C_loc: numpy.ndarray
                localized molecular orbitals coefficients.
            
        Returns:
            List[float]:
                maglz_expect: the expectation values of |Lz|for each MO, when aligned with the z-axis
        """

        lz_3comp = gto.moleintor.getints('int1e_cg_irxp_sph', mol._atm, mol._bas, mol._env, comp = 3)
        lz_matrix = lz_3comp[2]

        evals, evecs = np.linalg.eigh(lz_matrix)
        maglz = evecs @ np.diag(np.abs(evals)) @ evecs.T

        #lz_squared = lz_matrix.conj().T @ lz_matrix 
        maglz_expect = np.diag(C_loc.conj().T @ maglz @ C_loc).real

        return maglz_expect

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
            Tuple([List[str], List[str], List[float], List[float], List[float]):
                atoms_0: list of most-contributing atom symbols per orbital.
                atoms_1: list of second-most -contributing atom symbols per orb.
                distances: list of distances between those atom pairs.
                maglz_expect: the expectation values of |Lz|for each MO, when aligned with the z-axis
                mo_energies: Energies of the localized MOs.
        """

        mf = scf.RHF(self.mol)
        mf.kernel()
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ


        C_loc, U = MoleculeFeatureExtractor.localize_orbitals_separately(self.mol, mo_coeff, mo_occ)


        indices_list = MoleculeFeatureExtractor.population_analysis(mol, C_loc, mf)
        atoms_0, atoms_1, distances = MoleculeFeatureExtractor.find_distances_and_atoms_on_which_MOs_are_centered(mol, indices_list)
        rot_C_loc = MoleculeFeatureExtractor.rotate_orbitals(self.mol, C_loc, mf)
        maglz_expect = MoleculeFeatureExtractor.calculate_mag_lz(self.mol, rot_C_loc)
        mo_energies = MoleculeFeatureExtractor.calculate_energy(mf, U)

        #MoleculeFeatureExtractor.generate_cube_files(C_loc)

        return atoms_0, atoms_1, distances, maglz_expect, mo_energies

mol = gto.Mole()
mol.atom ='''
C1	0.0000	0.0000	0.0000
H2	0.6276	0.6276	0.6276
H3	0.6276	-0.6276	-0.6276
H4	-0.6276	0.6276	-0.6276
H5	-0.6276	-0.6276	0.6276
'''

mol.unit = 'Angstrom'
mol.basis = 'sto-3g'
mol.charge = 0
mol.spin = 0
mol.build()

c = MoleculeFeatureExtractor(mol)
atoms_0, atoms_1, distances, maglz_expect, mo_energies = c.extract_molecule_features()

print("################### atoms_0 ##################")
print('\n')
print(atoms_0)
print('\n')
print("################### atoms_1 ##################")
print('\n')
print(atoms_1)
print('\n')
print("################### distances ##################")
print('\n')
print(distances)
print('\n')
print("################### maglz_expect ##################")
print('\n')
print(maglz_expect)
print('\n')
print("################### mo_energies ##################")
print('\n')
print(mo_energies)
print('\n')
