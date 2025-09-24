import numpy as np
import pandas as pd

# pandas read csv from URL:
url = "https://raw.githubusercontent.com/jadsoncastro/UnifiedView/refs/heads/master/data/mfeat-karhunen.data"
df = pd.read_csv(url, sep=",", header=None)
df.to_parquet("docs/data/mfeat_karhunen/sources/mfeat_karhunen.parquet", index=False)
np.save("docs/data/mfeat_karhunen/generated/X.npy", df.iloc[:, :-1].values)
np.save("docs/data/mfeat_karhunen/generated/y.npy", df.iloc[:, -1].values)
