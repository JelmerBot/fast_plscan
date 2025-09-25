import numpy as np
import pandas as pd

# pandas read csv from URL:
url = "https://raw.githubusercontent.com/jadsoncastro/UnifiedView/refs/heads/master/data/mfeat-factors.data"
df = pd.read_csv(url, sep=",", header=None)
df.to_parquet("docs/data/mfeat_factors/sources/mfeat_factors.parquet", index=False)
np.save("docs/data/mfeat_factors/generated/X.npy", df.iloc[:, :-1].values)
np.save("docs/data/mfeat_factors/generated/y.npy", df.iloc[:, -1].values)
