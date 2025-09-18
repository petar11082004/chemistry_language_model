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
from qcmagic.core.cspace.basis.basisset import ConvolvedBasisSet
from scipy.linalg import expm
import math
import pandas as pd
from scipy.optimize import linear_sum_assignment
from qcmagic.interfaces.converters.pyscf import configuration_to_mol

    
class MoleculeFeatureExtractor:

    def __init__(self, mol):
        self.mol = mol

    @staticmethod
    def permute_orbitals(C ,S, L):
        """
        Permute localised orbitals L to best match reference orbitals C by maximising orbital overlap

        Args:
            C (np.ndarray): reference orbital coefficients
            S (np.ndarray): AO overlap matrix
            L (np.ndarray): Localised orital coefficients.
        
        Returns:
            np.ndarray: Localised orbitals reordered to best align with C
        """

        # Overlap between reference and localised orbitals
        O = C.T @ S @ L

        # Solve assignment problem: maximise |O| using Hungarian algorithm 
        r,c = linear_sum_assignment(-np.abs(O))

        # Build permutation mapping
        perm = np.empty_like(c)
        perm[r] = c

        # Reorder localised orbitals
        L_aln = L[:, perm].copy()
        return L_aln

    @staticmethod
    def localize_orbitals_separately(mol, mo_coeff, mo_occ, L_prev =None):

        """
        Localize occupied and virtual molecular orbitals separately using the Pipek–Mezey method.

        Args:
            mol (pyscf.gto.Mole): Molecule object.
            mo_coeff (np.ndarray): Molecular orbital coefficients.
            mo_occ (np.ndarray): Molecular orbital occupations.
            L_prev (np.ndarray, optional): Reference localized orbitals for permutation.

        Returns:
            tuple:
                L (np.ndarray): Localized orbitals (occupied + virtual).
                U (np.ndarray): Transformation matrix from canonical to localized orbitals.
        """

        # --- Separate occupied and virtual orbitals ---
        occ_idx = mo_occ > 0
        vir_idx = mo_occ == 0

        C_occ = mo_coeff[:, occ_idx]
        C_vir = mo_coeff[:, vir_idx]

        n_occ = C_occ.shape[1]
        n_vir = C_vir.shape[1]

        # --- Apply small random orthogonal rotations ---
        np.random.seed(42)
        A_occ = np.random.randn(n_occ, n_occ)
        A_vir = np.random.randn(n_vir, n_vir)

        A_occ = 0.5 * (A_occ - A_occ.T)  # antisymmetric
        A_vir = 0.5 * (A_vir - A_vir.T)  # antisymmetric

        Q_occ = expm(A_occ)  # orthogonal rotation
        Q_vir = expm(A_vir)

        C_occ_rot = C_occ @ Q_occ
        C_vir_rot = C_vir @ Q_vir

        # --- Overlap matrix ---
        S = mol.intor("int1e_ovlp")        
        
        # --- Localize separately with Pipek–Mezey ---
        L_occ_method = lo.pipek.PipekMezey(mol, C_occ_rot, pop_method = 'mulliken')
        L_vir_method = lo.pipek.PipekMezey(mol, C_vir_rot, pop_method = 'mulliken')

        L_occ_method.init_guess = C_occ_rot
        L_vir_method.init_guess = C_vir_rot        

        L_occ = L_occ_method.kernel()
        L_vir = L_vir_method.kernel()

        # Ensure consistent orbital ordering
        L_occ = MoleculeFeatureExtractor.permute_orbitals(C_occ_rot, S ,L_occ)
        L_vir = MoleculeFeatureExtractor.permute_orbitals(C_vir_rot, S, L_vir)

        positivify_coeffmats([L_occ])
        positivify_coeffmats([L_vir])

        # --- Stack localized occupied and virtual orbitals ---
        L = np.hstack([L_occ, L_vir])

        # Optional: permute relative to previous localization
        if L_prev is not None:
            L = MoleculeFeatureExtractor.permute_orbitals(L_prev, S, L)
        
        # --- Transformation matrix from canonical orbitals ---
        U = np.linalg.pinv(mo_coeff) @ L

        return L, U
    
    @staticmethod
    def population_analysis(mol, C_loc, mf):
        """
        Perform Mulliken population analysis on localized molecular orbitals .
        For each orbital, compute its Mulliken population per atom and assign it to the atoms where the orbital is mainly localised.
        
        Args:
            mol (pyscf.gto.Mole): Molecule object.
            C_loc (numpy.ndarray): Localized molecular orbitals coefficients.
            mf (pyscf.scf.hf.SCF): Mean-field (SCF) calculation result.

        Returns:
            list[list[int]]: For each orbital, a list of 1-3 atom indices (0-based) where the orbital has the largest Mulliken population.
        """

        # AO overlap matrix
        S = mf.get_ovlp()

        n_atoms = mol.natm
        indices_list = []
        
        # Precompute products to avoid recomputing inside the loop
        C_dagger = C_loc.conj().T
        SC = S @ C_loc

        for i in range(C_loc.shape[1]):
             # i-th localized orbital
            c_dagger = C_dagger[i, :]
            sc = SC[:, i]

            # Population per atom
            pop_per_atom = np.zeros(n_atoms)
            for A in range(n_atoms):
                ao_slice = mol.aoslice_by_atom()[A]
                p0, p1 = ao_slice[2], ao_slice[3]

                pop_per_atom[A] = c_dagger[p0:p1] @ sc[p0:p1]

            # Keep 1–3 atoms where population is significant (≥ 0.15)
            num_atoms = sum(pop_per_atom >= 0.15)
            num_atoms = min(max(num_atoms, 1), 3) 

            indices = pop_per_atom.argsort()[-num_atoms:][::-1]
            indices_list.append(indices)

        return indices_list

    @staticmethod
    def find_inverse_distances_and_atoms_on_which_MOs_are_centered(mol, indices_list):

        """
        For each localised molecular orbital (LMO), find the atoms it is centered on,
        their charges, and inverse interatomic distances between them.

        Handles cases where an orbital is assigned to 1-3 atoms.

        Args:
            mol (pyscf.gto.Mole): Molecule object.
            indices_list (list[list[int]]): For each orbital,  a list of 1-3 atom indices
                (sorted py population weight)

        Returns:
            tuple: 
                atoms_0: (list[str]): Symbol of the atom most strongly associated with each orbital
                atoms_1 (list[Union[str, int]]): Second atom symbol, or 0 if not applicable
                atoms_2 (list[Union[str, int]]): Third atom symbol, or 0 if not applicable
                charges_0 (list[int]): Nuclear charge of atom atom_0.
                charges_1 (list[int]): Nuclear charge of atom_1 (0 if not applicable).
                charges_2 (list[int]): Nuclear charge of atom_2 (0 if not applicable).
                inv_R_01 (list[float]): 1/ distance(atom_0, atom_1), or 0 if not applicable.
                inv_R_02 (list[float]): 1/ distance(atom_0, atom_2), or 0 if not applicable.
                inv_R_12 (list[float]): 1/ distance(atom_1, atom_2), or 0 if not applicable.
        """
        
        atoms_0, atoms_1, atoms_2 = [], [], []
        charges_0, charges_1, charges_2 = [], [], []
        inv_R_01, inv_R_12, inv_R_02 = [], [], []

        for indices in indices_list:
            # Pad with zeros if fewer than 3 atoms
            padded = list(indices) + [None] * (3-len(indices))
            idx0, idx1, idx2 = padded[:3]
            
            # Atom symbols
            atoms_0.append(mol.atom_symbol(idx0) if idx0 is not None else 0)
            atoms_1.append(mol.atom_symbol(idx1) if idx1 is not None else 0)
            atoms_2.append(mol.atom_symbol(idx2) if idx2 is not None else 0)

            # Charges
            charges_0.append(mol.atom_charges()[idx0] if idx0 is not None else 0)
            charges_1.append(mol.atom_charges()[idx1] if idx1 is not None else 0)
            charges_2.append(mol.atom_charges()[idx2] if idx2 is not None else 0)

            # Coordiantes
            coord_0 = mol.atom_coord(idx0) if idx0 is not None else None
            coord_1 = mol.atom_coord(idx1) if idx1 is not None else None
            coord_2 = mol.atom_coord(idx2) if idx2 is not None else None

            #Distances
            inv_R_01.append(1 / np.linalg.norm(coord_1 - coord_0) if idx0 is not None and idx1 is not None else 0)
            inv_R_02.append(1 / np.linalg.norm(coord_2 - coord_0) if idx0 is not None and idx2 is not None else 0)
            inv_R_12.append(1 / np.linalg.norm(coord_2 - coord_1) if idx1 is not None and idx2 is not None else 0)
        
        return atoms_0, atoms_1, atoms_2, charges_0, charges_1, charges_2, inv_R_01, inv_R_02, inv_R_12

    @staticmethod
    def find_mo_orientation_vectors(indices_list, mol):

        """
        Compute orientation vectors for localised molecular orbitals (LMOs).

        - For orbitals centered on a single atom: return a zero vector (no orientation).
        - For orbitals centered on two atoms: return the bond vector (atom1 - atom0).
        - For orbitals centered on three atoms: return the normal vector to the plane defined by the three atoms (via cross product).
        
        Args:
            indices_list (list[list[int]]): For each orbital, a list of atom indices
                (sorted by population contribution)
            mol (pyscf.gto.Mole): Molecule object.

        Returns:
            list[np.ndarray]: Orientation vectors for each orbital.
        """

        mo_orientation_vectors = []

        for indices in indices_list:
            if len(indices) == 1:
                # Single atom - no orientation
                mo_orientation_vectors.append(np.array([0, 0, 0]))

            if len(indices) == 2:
                # Two atoms - bond direction
                idx0, idx1 = indices[0], indices[1]
                coord_0 = mol.atom_coord(idx0)  
                coord_1 = mol.atom_coord(idx1)
                mo_orientation_vectors.append(coord_1 - coord_0)
            
            elif len(indices) == 3:
                # Three atoms - plane normal
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
        Compute the angle (in radians) required to rotate each orbital orientation vector onto the z-axis

        - If the vector is zero (e.g., orbital centered on a single atom), the angles is defined as 0
        
        Args:
            mo_orientation_vectors (list[np.ndarray]): Orientation vectors of localised molecular orbitals

        Returns:
            list[float]: Rotation angles (in radians) for each orbital.
        """

        z_axis = np.array([0.0, 0.0, 1.0])
        angles = []

        for vector in mo_orientation_vectors:
            
            if not np.array_equal(vector, [0, 0, 0]):
                angle = np.arccos(np.dot(np.array([0,0,1]), vector)/np.linalg.norm(vector))
                angles.append(angle)
            else:
                angles.append(math.radians(0))
        
        return angles
    
    @staticmethod
    def block_diagonalize_lz_matrix(lz_matrix, mol):
        """
        Construct a block-diagonal version of the Lz matrix in the AO basis,
        where only intra-atomic blocks are retained.

        Args:
            lz_matrix (np.ndarray): The full Lz matrix in the AO basis (nao x nao).
            mol (pyscf.gto.Mole): Molecule object.

        Returns:
            np.ndarray: Block-diagonalised Lz matrix (same shape as input),
            where off-diagonal couplings between different atoms are set to 0.
        """
        ao_slices = mol.aoslice_by_atom()
        lz_block = np.zeros_like(lz_matrix)

        for A in range(mol.natm):
                ao_slice = ao_slices[A]
                p0, p1 = ao_slice[2], ao_slice[3]
                lz_block[p0:p1, p0:p1] = lz_matrix[p0:p1, p0:p1]

        return lz_block
    
    @staticmethod
    def calculate_mag_lz(mol, rot_C_loc):

        """
        Calculate the expectation values of |Lz| for the localised molecular orbitals. 

        Uses the block-diagonalised atomic orbital Lz operator to compute orbital angular momentum along the z-axis.
        
        Args:
            mol (pyscf.gto.Mole): Molecule object.
            rot_C_loc (numpy.ndarray): Localized molecular orbitals coefficients.
            
        Returns:
            np.ndaray: Expectation values of |Lz| for each orbital.
        """

        # AO angular momentum integrals (3 components: Lx, Ly, Lz)
        lz_integrals = gto.moleintor.getints('int1e_cg_irxp_sph', mol._atm, mol._bas, mol._env, comp = 3)
        lz_matrix = lz_integrals[2] # Select Lz
        lz_matrix = MoleculeFeatureExtractor.block_diagonalize_lz_matrix(lz_matrix, mol)

        # Diagonalise and take absolute values of eigenvalues
        evals, evecs = np.linalg.eigh(lz_matrix)
        maglz = evecs @ np.diag(np.abs(evals)) @ evecs.T

        # Compute <MO| |Lz| |MO>
        maglz_expect = np.diag(rot_C_loc.conj().T @ maglz @ rot_C_loc).real

        return maglz_expect

    @staticmethod
    def rotate_orbitals_and_calculate_mag_lz(mol, C_loc, mf):

        """
        Rotate localized molecular orbitals so they align with the z-axis, and calculate the expectation values of |Lz|.
        
        - Orbitals on a single atom: return 0.
        - Orbitals already aligned with z (angle = 0 or pi): use unrotated |Lz|
        - Otherwise: rotate the state, transform coefficients back into PySCF format, and recalculate |Lz|.
        
        Args:
            mol (pyscf.gto.Mole): Molecule object
            C_loc (numpy.ndarray) Localised MO coefficients.
            mf (pyscf.scf.hf.SCF): Mean-field calculation (provides occupations).

        Returns:
            np.ndarray:
                maglz_expect: expectation values of |Lz| for each rotated orbital.
        """

        # Step 1: Get atom assignment and orientation vectors
        indices_list = MoleculeFeatureExtractor.population_analysis(mol, C_loc, mf)
        vectors = MoleculeFeatureExtractor.find_mo_orientation_vectors(indices_list, mol)
        angles = MoleculeFeatureExtractor.find_mo_rotation_angles(vectors)

        # Step 2: Build reference state with localised orbitals
        base_mf = scf.RHF(mol)
        base_mf.mo_coeff = C_loc
        base_state = scf_to_state(base_mf)
        config = base_state.configuration

        maglz_expect = []

        for i, angle in enumerate(angles):

            # case A: Orbital localised on a single atom - no angular momentum
            if len(indices_list[i]) == 1:
                maglz_expect.append(0)
                continue
            
            # case B: alreaady aligned with z-axis (angle ~ 0 or π)
            if np.isclose(angle % math.pi, 0.0, atol = 1e-8):
               
                maglz_expect.append(MoleculeFeatureExtractor.calculate_mag_lz(mol, C_loc)[i])
                continue
            
            # Case C: General rotation of quantum state
            axis_of_rotation = np.cross(vectors[i], [0.0, 0.0, 1.0])

            # Rotate quantum state
            state_rot = rotate_state(base_state, angle, Vector3D(axis_of_rotation))

            # Convert rotated coefficients into PySCF AO basis
            cbs=config.get_subconfiguration("ConvolvedBasisSet")
            coeffs_rot_pyscf = cbs.convert_coefficient_matrices(
                state_rot.coefficients, 
                format_from=BasisType.BT_LIBINT, 
                format_to=BasisType.BT_PYSCF
            )

            mol_rot = configuration_to_mol(state_rot.configuration)
            coeffs = coeffs_rot_pyscf[0]

            maglz_expect.append(MoleculeFeatureExtractor.calculate_mag_lz(mol_rot, coeffs)[i])
        
        return  np.array(maglz_expect)


    @staticmethod
    def calculate_energy(mf, U):

        """
        Calculate localised Molecular orbital (LMO) energies.

        The energies are obtained as the expectation values of the Fock operator in the localised MO basis:
            ε_i = ⟨L_i | F | L_i⟩

       Args:
        mf (pyscf.scf.hf.SCF): Mean-field (SCF) calculation result.
        U (np.ndarray): Unitary matrix transforming canonical MOs
            into localized MOs.

        Returns:
            np.ndarray: Energies of the localized MOs.
        """

        mo_energies = mf.mo_energy
        loc_mo_energies = np.diag(U.conj().T @ np.diag(mo_energies) @ U).real
        return loc_mo_energies
    
    @staticmethod
    def generate_cube_files(C_loc, mol, outdir = "cube_files"):

        """
        Generate Gaussian cube files for localized molecular orbitals (LMOs),
        which can be visualized with external software

        Args:
            C_loc (np.ndarray): Localized MO coefficients.
            mol (pyscf.gto.Mole): Molecule object.
            outdir (str, optional): Directory where cube files will be saved. Default: "cube_files".
        """

        # Create output directory if it doesn't exist
        os.makedirs(outdir, exist_ok=True)
        for mo_index in range(C_loc.shape[1]):
            coeff_vector = C_loc[:, mo_index]

            cube_filename = os.path.join(outdir, f'mo{mo_index}.cube')

            cubegen.orbital(mol, cube_filename, coeff_vector, nx=80, margin=3.0)
        
    def extract_molecule_features(self, L_prev = None, ntries = 20, write_cubes = True):

        """
        Extract molecule features by performing repeated orbital localizations and keeping the best result (based on the Pipek–Mezey functional).

        Args:
        L_prev (np.ndarray, optional): Localized MO coefficients from a previous
            molecule, used for orbital alignment.
        ntries (int, optional): Number of localization attempts with different
            random seeds. Default is 20.
        write_cubes (bool, optional): If True, generate cube files for LMOs.
            Default is True.

        Returns:
            tuple:
                maglz_expect (np.ndarray): Expectation values of |Lz| for each LMO.
                atoms_0 (list[str]): Most-contributing atom per orbital.
                atoms_1 (list[Union[str, int]]): Second-most contributing atom per orbital (0 if none).
                atoms_2 (list[Union[str, int]]): Third-most contributing atom per orbital (0 if none).
                charges_0 (list[int]): Nuclear charges of atoms_0.
                charges_1 (list[int]): Nuclear charges of atoms_1.
                charges_2 (list[int]): Nuclear charges of atoms_2.
                inv_R_01 (list[float]): Inverse distances between atoms_0 and atoms_1.
                inv_R_02 (list[float]): Inverse distances between atoms_0 and atoms_2.
                inv_R_12 (list[float]): Inverse distances between atoms_1 and atoms_2.
                mo_energies (np.ndarray): Energies of the localized MOs.
        """
        # --- SCF calculation ---
        mf = scf.RHF(self.mol)
        mf.kernel()
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ

        # --- Try multiple localisations, keep best ---
        best_val = -np.inf
        best_L, best_U = None, None

        for trial in range(ntries):

            # run localization (may converge to different minima)
            L, U = MoleculeFeatureExtractor.localize_orbitals_separately(self.mol, mo_coeff, mo_occ, L_prev)
            
            # compute PM functional
            val_localized = lo.pipek.PipekMezey(
                self.mol, L, pop_method = 'lowdin'
            ).cost_function()

            #keep the best-scoring result (maximize Pipek functional)
            if val_localized > best_val:
                best_val = val_localized
                best_L, best_U = L, U
        
        # Final best localisation
        C_loc, U = best_L, best_U

        # --- feature extraction ---
        indices_list = MoleculeFeatureExtractor.population_analysis(self.mol, C_loc, mf)
        
        atoms_0, atoms_1, atoms_2, charges_0, charges_1, charges_2, inv_R_01, inv_R_02, inv_R_12 = MoleculeFeatureExtractor.find_inverse_distances_and_atoms_on_which_MOs_are_centered(self.mol, indices_list)
        
        maglz_expect = MoleculeFeatureExtractor.rotate_orbitals_and_calculate_mag_lz(self.mol, C_loc, mf)
        
        if write_cubes:
            MoleculeFeatureExtractor.generate_cube_files(C_loc, self.mol)
        mo_energies = MoleculeFeatureExtractor.calculate_energy(mf, U)

        return maglz_expect, atoms_0, atoms_1, atoms_2, charges_0, charges_1, charges_2, inv_R_01, inv_R_02, inv_R_12, mo_energies
        