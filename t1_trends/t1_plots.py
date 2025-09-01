import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_parquet('t1_pairs_f2_20250829_155449.parquet')

bond_lengths = [1.1, 1.075, 1.05, 1.025, 1.00, 0.975, 0.95, 0.925, 0.9]
t1 = np.abs(df['t1'])
t1 = np.array(t1).reshape(9, int(len(t1)/9))

plt.figure(figsize = (8,6))
for i in range(t1.shape[1]):
    plt.plot(bond_lengths, t1[])

plt.title(r"$F_2$")
plt.xlabel("bond length multiplier")
plt.ylabel("t1")
plt.show()

