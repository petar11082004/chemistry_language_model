#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import pyscf
from pyscf import gto, scf
import scipy as sp


# In[ ]:


class MoleculeFeatureExtractor:

    def __init__(self, mol):
        self.mol = mol

    def localize_orbitals_separately(mol, mo_coeff, mo_occ):

        """
        Localize occupied and virtual molecular orbitals separatley using Boys method

        Args:
            mol: pyscf.gto.Mole
                Molecule object.
            mf: pyscf.scf.hf.SCF
                Mean-field (SCF) calculation result.

        Returns:
            numpy.ndarray:
                stacked localized occupied and virtual orbitals
        """

        #Create boolean masks
        occ_idx = mo_occ > 0
        vir_idx = mo_occ == 0

        # Use masks to select occupied and virtual orbitals
        occupied_orbitals_coeffs = mo_coeff[:, occ_idx]
        virtual_orbitals_coeffs = mo_coeff[:, vir_idx]

        localized_occupied_orbitals_method = lo.Boys(mol, occupied_orbitals_coeffs)
        localized_virtual_orbitals_method = lo.Boys(mol, virtual_orbitals_coeffs)        

        localized_occupied_orbitals_coeffs = localized_occupied_orbitals_method.kernel()
        localized_virtual_orbitals_coeffs = localized_virtual_orbitals_method.kernel()

        U_occ = localized_occupied_orbitals_coeffs @ np.linalg.inv(occupied_orbitals_coeffs)
        U_vir = localized_virtual_orbitals_coeffs @ np.linalg.inv(virtual_orbitals_coeffs)
        U = sp.linalg.block_diag(U_occ, U_vir)

        loc_mo_coeffs = np.hstack([localized_occupied_orbitals_coeffs, localized_virtual_orbitals_coeffs])

        return loc_mo_coeffs, U

    def population_analysis(mol, C_loc, mf):

        """
        Perform Mulliken population analyzis on molecular orbitals 
        to determine which atoms are they centered on

        Args:
            mol: pyscf.gto.Mole
                Molecule object.
            C_loc: numpy.ndarray
                localized molecular orbitals coefficients
            mf: pyscf.scf.hf.SCF
                Mean-field (SCF) calculation result.

        Returns:
            Tuple([numpy.ndarray, numpy.ndarray, numpy.ndarray]):
                atoms_0: list of most-contributing atom symbols per orbital
                atoms_1: list of second-most -contributing atom symbols per orb:
                distances: list of distances between those atom pairs
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

    def determine_orbital_type(mol, C_loc):

        """
        calculate the expectation values of Lz^2 for the molecular orbitals 
        to determine their character

        Args:
            mol: pyscf.gto.Mole
                Molecule object.
            C_loc: numpy.ndarray
                localized molecular orbitals coefficients
            
        Returns:
            list of str:
                Labels of the type of MO (σ, π, δ, or mixed).
        """

        lz_3comp = gto.moleintor.getints('int1e_cg_irxp_sph', mol._atm, mol._bas, mol._env, comp = 3)
        lz_matrix = lz_3comp[2]
        lz_squared = lz_matrix @ lz_matrix
        lz_expect = np.diag(C_loc.conj().T @ lz_squared @ C_loc).real

        labels = []

        for lz_val in lz_expect:
            if abs(lz_val) < 0.1:
                label = 'σ'
            elif abs(lz_val - 1.0) < 0.1:
                label = 'π'
            elif abs(lz_val - 4.0) < 0.1:
                label = 'δ'
            else:
                label = 'mixed'
            labels.append(label)

        return labels

    def calculate_energy(mol, C_loc, mf, U):

        """
        calculate the energies of the localized molecular orbitals

        Args:
            mol: pyscf.gto.Mole
                Molecule object.
            C_loc: numpy.ndarray
                localized molecular orbitals coefficients
            mf: pyscf.scf.hf.SCF
                Mean-field (SCF) calculation result.
            
        Returns:
            list of floats:
                Energies of MOs.
        """

        mo_energies = mf.mo_energies

        ## finish function
        
    def obtain_mo_coeffs(mol)
        mf = scf.RHF(mol)
        mf.kernel()
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        C_loc = localize_orbitals_separately(mol, mo_coeff, mo_occ)

    

        

