from rdkit import Chem
from rdkit.Chem import rdMolTransforms as T
from rdkit.Chem import rdDetermineBonds
import numpy as np

xyz = """4
ammonia
N	0.0000	0.0000	0.0000
H	0.0000	-0.9377	-0.3816
H	0.8121	0.4689	-0.3816
H	-0.8121	0.4689	-0.3816
"""

mol = Chem.MolFromXYZBlock(xyz)
# 1) perceive bonds from distances
rdDetermineBonds.DetermineConnectivity(mol)
# 2) make sure caches and ring info are initialized
mol.UpdatePropertyCache(strict=False)
Chem.GetSymmSSSR(mol)   # initializes RingInfo; no-op if no rings

conf = mol.GetConformer()

# Now transforms work without errors:
'''
i, j = 0, 1  
d = T.GetBondLength(conf, i, j)
T.SetBondLength(conf, i, j, d * 1.1)  # stretch +10%
'''
i, j, k = 1, 0, 2      
ang = T.GetAngleDeg(conf, i, j, k)
T.SetAngleDeg(conf, i, j, k, ang * 0.9)

# Extract coordinates back out
pts = conf.GetPositions()
geom = "\n".join(f"{a.GetSymbol()}\t{xyz[0]:.4f}\t{xyz[1]:.4f}\t{xyz[2]:.4f}"
                 for a, xyz in zip(mol.GetAtoms(), pts))
print(geom)


