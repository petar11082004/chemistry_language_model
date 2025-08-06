
import numpy as np
from pyscf import gto, scf, lo

# 1. Define molecule
mol = gto.Mole()
mol.atom ='''
C1	0.0000	0.0000	0.0000
O2	0.0000	0.0000	1.1621
O3	0.0000	0.0000	-1.1621
'''

mol.basis = 'sto-3g'
mol.build()

# 2. Run SCF
mf = scf.RHF(mol).run()
mo_coeff = mf.mo_coeff
S = mf.get_ovlp()


from scipy.linalg import fractional_matrix_power

def population_lowdin(mol, mo_coeff, S):
    # Löwdin orthogonalization: S^(-1/2)
    S_inv_sqrt = fractional_matrix_power(S, -0.5)
    mo_orth = S_inv_sqrt @ mo_coeff  # Transform MO coeffs to orthogonal AO basis

    ao_loc = mol.aoslice_by_atom()
    nmo = mo_coeff.shape[1]
    natm = mol.natm
    population = np.zeros((nmo, natm))

    for i in range(nmo):
        C = mo_orth[:, i]
        for A in range(natm):
            p0, p1 = ao_loc[A][2], ao_loc[A][3]
            pop = np.sum(C.conj().T[p0:p1] @ C[p0:p1])  # Now basis is orthonormal, so dot becomes square norm
            population[i, A] = pop

    population_percent = 100 * population / population.sum(axis=1, keepdims=True)
    return population_percent




# 3. Function for population analysis
def population_mulliken(mol, mo_coeff, S):
    ao_loc = mol.aoslice_by_atom()
    nmo = mo_coeff.shape[1]
    natm = mol.natm
    population = np.zeros((nmo, natm))

    for i in range(nmo):
        C = mo_coeff[:, i]
        PC = C @ S
        for A in range(natm):
            p0, p1 = ao_loc[A][2], ao_loc[A][3]
            pop = C[p0:p1] @ PC[p0:p1]
            population[i, A] = pop

    population_percent = 100 * population / population.sum(axis=1, keepdims=True)
    return population_percent

def print_population_table(title, pop, mol):
    natm = mol.natm
    atom_labels = [mol.atom_symbol(i) for i in range(natm)]
    header = "MO  " + "  ".join(f"{a:^8}" for a in atom_labels)
    print(f"\n=== {title} ===")
    print(header)
    print("-" * len(header))
    for i, row in enumerate(pop):
        line = f"{i:>2}  " + "  ".join(f"{p:8.2f}" for p in row)
        print(line)

# 4. Canonical MO population analysis
pop_canonical = population_lowdin(mol, mo_coeff, S)

# 5. Localize MOs using Edmiston–Ruedenberg
localizer = lo.edmiston.EdmistonRuedenberg(mol, mo_coeff)
mo_loc = localizer.kernel()

# 6. Localized MO population analysis
pop_localized = population_lowdin(mol, mo_loc, S)

print('########### Lowdin ########')
# 8. Print results
print_population_table("Canonical MO Population Analysis (%)", pop_canonical, mol)
print_population_table("Localized (Edmiston–Ruedenberg) MO Population Analysis (%)", pop_localized, mol)


pop_canonical = population_mulliken(mol, mo_coeff, S)

# 5. Localize MOs using Edmiston–Ruedenberg
localizer = lo.edmiston.EdmistonRuedenberg(mol, mo_coeff)
mo_loc = localizer.kernel()

# 6. Localized MO population analysis
pop_localized = population_mulliken(mol, mo_loc, S)

print('########### Mulliken ########')
# 8. Print results
print_population_table("Canonical MO Population Analysis (%)", pop_canonical, mol)
print_population_table("Localized (Edmiston–Ruedenberg) MO Population Analysis (%)", pop_localized, mol)

