import pyscf
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
from qcmagic.core.sspace.statespace import positivify_coeffmats
from scipy.linalg import expm
import math
import pandas as pd
from scipy.optimize import linear_sum_assignment
    
class MoleculeFeatureExtractor:

    def __init__(self, mol):
        self.mol = mol

    @ staticmethod
    def permute_orbitals(C ,S, L):
        O = C.T @ S @ L
        r,c = linear_sum_assignment(-np.abs(O))
        perm = np.empty_like(c)
        perm[r] = c
        L_aln = L[:, perm].copy()
        return L_aln

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
        C_occ = mo_coeff[:, occ_idx]
        C_vir = mo_coeff[:, vir_idx]

        n_occ = C_occ.shape[1]
        n_vir = C_vir.shape[1]

        # Create small real antisymmetric matrices A_occ, A_vir
        np.random.seed(42)
        A_occ = np.random.randn(n_occ, n_occ)
        A_vir = np.random.randn(n_vir, n_vir)
        A_occ = A_occ - A_occ.T  # make it antisymmetric
        A_vir = A_vir - A_vir.T  # make it antisymmetric
        A_occ*= 0.5  # scale to make rotation small
        A_vir*= 0.5  # scale to make rotation small

        # Compute real orthogonal matrix Q = exp(A)
        Q_occ = expm(A_occ)  # real orthogonal
        Q_vir = expm(A_vir) #real orthogonal

        # Apply rotation: C' = C @ Q
        C_occ_rot = C_occ @ Q_occ
        C_vir_rot = C_vir @ Q_vir

        #.pipek.PipekMezey, .edmiston.EdmistonRuedenberg

        L_occ_method = lo.EdmistonRuedenberg(mol, C_occ_rot)
        L_vir_method = lo.EdmistonRuedenberg(mol, C_vir_rot)

        L_occ_method.init_guess = None
        L_vir_method.init_guess = None        

        L_occ = L_occ_method.kernel()
        L_vir = L_vir_method.kernel()

        S = mol.intor("int1e_ovlp")        
        L_occ = MoleculeFeatureExtractor.permute_orbitals(C_occ, S ,L_occ)
        L_vir = MoleculeFeatureExtractor.permute_orbitals(C_vir, S, L_vir)

        positivify_coeffmats([L_occ])
        positivify_coeffmats([L_vir])

        U_occ =  np.linalg.pinv(C_occ_rot) @ L_occ
        U_vir = np.linalg.pinv(C_vir_rot) @ L_vir
        U = sp.linalg.block_diag(U_occ, U_vir)

        L = np.hstack([L_occ, L_vir])

        return L, U
    
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

        C_dagger = C_loc.conj().T
        SC = S @ C_loc
        for i in range(C_loc.shape[1]):
            c_dagger = C_dagger[i, :] # i-th localized orbital
            sc = SC[:, i]
            pop_per_atom = np.zeros(n_atoms)

            for A in range(n_atoms):
                ao_slice = mol.aoslice_by_atom()[A]
                p0, p1 = ao_slice[2], ao_slice[3]

                pop_per_atom[A] = c_dagger[p0:p1] @ sc[p0:p1]

            counter = sum(pop_per_atom >= 0.15)
            counter = min(max(counter, 1), 3) 
            indices = pop_per_atom.argsort()[-counter:][::-1]
            
            indices_list.append(indices)

        return indices_list

    @staticmethod
    def find_inverse_distances_and_atoms_on_which_MOs_are_centered(mol, indices_list):

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
        
        charges_0 = []
        charges_1 = []
        charges_2 = []
        inv_R_01 =[]
        inv_R_12 = []
        inv_R_02 = []

        for indices in indices_list:

            if len(indices)== 1:
                idx0 = indices[0]
                charge_0 = mol.atom_charges()[idx0]
                charges_0.append(charge_0)
                charges_1.append(0)
                charges_2.append(0)
                inv_R_01.append(0)
                inv_R_12.append(0)
                inv_R_02.append(0)
            
            elif len(indices) == 2:
                idx0, idx1 = indices[0], indices[1]
                charge_0 = mol.atom_charges()[idx0]
                charge_1 = mol.atom_charges()[idx1]
                coord_0 = mol.atom_coord(idx0)  
                coord_1 = mol.atom_coord(idx1)
                
                charges_0.append(charge_0)
                charges_1.append(charge_1)
                charges_2.append(0)
                inv_R_01.append(1/np.linalg.norm(coord_1 - coord_0))
                inv_R_12.append(0)
                inv_R_02.append(0)
            
            elif len(indices) == 3:
                idx0, idx1, idx2 = indices[0], indices[1], indices[2]
                charge_0 = mol.atom_charges()[idx0]
                charge_1 = mol.atom_charges()[idx1]
                charge_2 = mol.atom_charges()[idx2]
                coord_0 = mol.atom_coord(idx0)  
                coord_1 = mol.atom_coord(idx1)
                coord_2 = mol.atom_coord(idx2)
                charges_0.append(charge_0)
                charges_1.append(charge_1)
                charges_2.append(charge_2)
                inv_R_01.append(1/np.linalg.norm(coord_1 - coord_0))
                inv_R_12.append(1/np.linalg.norm(coord_2 - coord_1))
                inv_R_02.append(1/np.linalg.norm(coord_2 - coord_0))
        
        return charges_0, charges_1, charges_2, inv_R_01, inv_R_02, inv_R_12

    @staticmethod
    def find_mo_orientation_vectors(indices_list, mol):

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
            if len(indices) == 1:
                mo_orientation_vectors.append(np.array([0, 0, 0]))
            if len(indices) == 2:
                idx0, idx1 = indices[0], indices[1]
                coord_0 = mol.atom_coord(idx0)  
                coord_1 = mol.atom_coord(idx1)
                mo_orientation_vectors.append(coord_1 - coord_0)
            
            elif len(indices) == 3:
                idx0, idx1, idx2 = indices[0], indices[1], indices[2]
                coord_0 = mol.atom_coord(idx0)  
                coord_1 = mol.atom_coord(idx1)
                coord_2 = mol.atom_coord(idx2)
                vecotr_01 = coord_1 - coord_0
                vector_02 = coord_2 - coord_0
                orientation_vector = np.cross(vecotr_01, vector_02)
                mo_orientation_vectors.append(orientation_vector)
        
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
            
            if not np.array_equal(vector, [0, 0, 0]):
                angle = np.arccos(np.dot(np.array([0,0,1]), vector)/np.linalg.norm(vector))
                angles.append(angle)
            else:
                angles.append(math.radians(0))
        
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
        vectors = MoleculeFeatureExtractor.find_mo_orientation_vectors(indices_list, mol)
        angles = MoleculeFeatureExtractor.find_mo_rotation_angles(vectors)

        rot_C_loc = []
        nAO, nMO = C_loc.shape
        rot_C_loc = np.zeros((nAO, nMO), dtype=C_loc.dtype)


        for i, angle in enumerate(angles):
            
            mf = scf.RHF(mol)
            mf.mo_coeff = C_loc
            state = scf_to_state(mf)
            config = state.configuration
            state_rot = rotate_state(state, angle, Vector3D([0,0,1]))
    
            cbs=config.get_subconfiguration("ConvolvedBasisSet")
            coeffs_rot_pyscf = cbs.convert_coefficient_matrices(
                                state_rot.coefficients, format_from=BasisType.BT_LIBINT, format_to=BasisType.BT_PYSCF)

            rot_C_loc[:, i] = coeffs_rot_pyscf[0][:, i]
                
        return  np.array(rot_C_loc)

    @staticmethod
    def calculate_mag_lz(mol, rot_C_loc):

        """
        calculate the expectation values of |Lz| for the molecular orbitals 

        Args:
            mol: pyscf.gto.Mole
                Molecule object.
            rot_C_loc: numpy.ndarray
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
        maglz_expect = np.diag(rot_C_loc.conj().T @ maglz @ rot_C_loc).real

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
        loc_mo_energies = np.diag(U.conj().T @ np.diag(mo_energies) @ U).real
        return loc_mo_energies
    
    @staticmethod
    def generate_cube_files(C_loc, mol):

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

            cube_filename = os.path.join("cube_files", f'mo{mo_index}.cube')

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

        #MoleculeFeatureExtractor.generate_cube_files(C_loc, self.mol)

        indices_list = MoleculeFeatureExtractor.population_analysis(self.mol, C_loc, mf)
        charges_0, charges_1, charges_2, inv_R_01, inv_R_02, inv_R_12 = MoleculeFeatureExtractor.find_inverse_distances_and_atoms_on_which_MOs_are_centered(self.mol, indices_list)
        rot_C_loc = MoleculeFeatureExtractor.rotate_orbitals(self.mol, C_loc, mf)
        maglz_expect = MoleculeFeatureExtractor.calculate_mag_lz(self.mol, rot_C_loc)
        mo_energies = MoleculeFeatureExtractor.calculate_energy(mf, U)


        return maglz_expect, charges_0, charges_1, charges_2, inv_R_01, inv_R_02, inv_R_12, mo_energies
    

