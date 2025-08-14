from rdkit import Chem
from rdkit.Chem import rdMolTransforms as T
from rdkit.Chem import rdDetermineBonds
import numpy as np

xyz = """7
methylamine
C	0.0583	0.7129	0.0000	 	
N	0.0583	-0.7726	0.0000	 	
H	-0.9425	1.1529	0.0000	 	
H	0.5877	1.0695	0.8821	 	
H	0.5877	1.0695	-0.8821	 	
H	-0.4953	-1.0804	-0.8165	 	
H	-0.4953	-1.0804	0.8165
"""

mol = Chem.MolFromXYZBlock(xyz)
# 1) perceive bonds from distances
rdDetermineBonds.DetermineConnectivity(mol)
# 2) make sure caches and ring info are initialized
mol.UpdatePropertyCache(strict=False)
Chem.GetSymmSSSR(mol)   # initializes RingInfo; no-op if no rings

conf = mol.GetConformer()

# Now transforms work without errors:
"""
i, j = 1,5 
d = T.GetBondLength(conf, i, j)
T.SetBondLength(conf, i, j, d * 0.9)  # stretch +10%
"""
i, j, k = 6, 1, 5     
ang = T.GetAngleDeg(conf, i, j, k)
T.SetAngleDeg(conf, i, j, k, ang * 1.1)

# Extract coordinates back out
pts = conf.GetPositions()
geom = "\n".join(f"{a.GetSymbol()}\t{xyz[0]:.4f}\t{xyz[1]:.4f}\t{xyz[2]:.4f}"
                 for a, xyz in zip(mol.GetAtoms(), pts))
print(geom)
