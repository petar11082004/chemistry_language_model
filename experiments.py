from rdkit import Chem
from rdkit.Chem import rdMolTransforms as T
from rdkit.Chem import rdDetermineBonds
import numpy as np

xyz = """9
ethanol
C	1.1879	-0.3829	0.0000
C	0.0000	0.5526	0.0000
O	-1.1867	-0.2472	0.0000
H	-1.9237	0.3850	0.0000
H	2.0985	0.2306	0.0000
H	1.1184	-1.0093	0.8869
H	1.1184	-1.0093	-0.8869
H	-0.0227	1.1812	0.8852
H	-0.0227	1.1812	-0.8852
"""

mol = Chem.MolFromXYZBlock(xyz)
# 1) perceive bonds from distances
rdDetermineBonds.DetermineConnectivity(mol)
# 2) make sure caches and ring info are initialized
mol.UpdatePropertyCache(strict=False)
Chem.GetSymmSSSR(mol)   # initializes RingInfo; no-op if no rings

conf = mol.GetConformer()

# Now transforms work without errors:

i, j = 1, 0 
d = T.GetBondLength(conf, i, j)
T.SetBondLength(conf, i, j, d * 1.1)  # stretch +10%
"""
i, j, k = 2, 0, 1     
ang = T.GetAngleDeg(conf, i, j, k)
T.SetAngleDeg(conf, i, j, k, ang * 1.1)
"""
# Extract coordinates back out
pts = conf.GetPositions()
geom = "\n".join(f"{a.GetSymbol()}\t{xyz[0]:.4f}\t{xyz[1]:.4f}\t{xyz[2]:.4f}"
                 for a, xyz in zip(mol.GetAtoms(), pts))
print(geom)
