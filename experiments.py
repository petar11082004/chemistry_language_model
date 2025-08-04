"""Tests for :mod:`qcmagic.interfaces.converters.pyscf`.
"""
import sys
sys.path.append('/home/pp583/revqcmagic')
from pyscf import gto, scf  # type: ignore``
from pyscf.scf import addons  # type: ignore
import numpy as np
from qcmagic.auxiliary.linearalgebra3d import Vector3D
print('done')
from qcmagic.core.drivers.statetools.rotate import rotate_state
print('done')
from qcmagic.interfaces.converters.pyscf import scf_to_state, configuration_to_mol


def test_pyscf_rhf_to_state_H2O_rotate():
    r""" Runs pyscf H2O molecule with RHF and imports it into qcmagic.
         It tests that the energy calculated in qcmagic is the same as pyscf.
    """
    h2o_mol = gto.M(
        atom=(
            """
            O	0.0000	0.0000	0.1173
            H	0.0000	0.7572	-0.4692
            H	0.0000	-0.7572	-0.4692
            """
        ),
        basis="STO-3G",
        unit="Angstrom",
    )
    rhf_calc = scf.RHF(h2o_mol)
    e_pyscf = rhf_calc.kernel()

    #if you want to modify the coefficients you should do so in rhf_calc BEFORE the conversion to RevQCMagic.
    #RevQCMagic's Atomic Orbital Order may differ from pyscf's.

    #e.g. rhf_calc.mo_coeff = ...

    state = scf_to_state(rhf_calc)


    config = state.configuration
    print("Old Energy", e_pyscf, state.energy)
    print("Old Geom", config.get_subconfiguration("Geometry").structure.all_atoms)
    print("Old Coeffs", state.coefficients)


    state_rot = rotate_state(state,np.pi/4, Vector3D([0,0,1]))
    config_rot = state_rot.configuration

    print("New Energy", state_rot.energy)
    print("New Geom", config_rot.get_subconfiguration("Geometry").structure.all_atoms)
    print("New Coeffs", state_rot.coefficients)

    mol_new = configuration_to_mol(config_rot)
    rhf_new = scf.RHF(mol_new)
    e_pyscf_new = rhf_new.kernel()

    print("New Pyscf Energy", e_pyscf_new)
    print("New Pyscf", rhf_new.mo_coeff)

    rqc_coeff = state_rot.coefficients
    pyscf_coeff = rhf_new.mo_coeff

    assert np.allclose(e_pyscf_new, e_pyscf, rtol=0, atol=1e-12)

test_pyscf_rhf_to_state_H2O_rotate()




