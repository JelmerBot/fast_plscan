import numpy as np
import pandas as pd

# pandas read csv from URL:
url = "https://raw.githubusercontent.com/jadsoncastro/UnifiedView/refs/heads/master/data/analcatdata_authorship-458.data"
df = pd.read_csv(url, sep=",", header=None)
df.to_parquet("docs/data/authorship/sources/authorship.parquet", index=False)
np.save("docs/data/authorship/generated/X.npy", df.iloc[:, :-1].values)
np.save("docs/data/authorship/generated/y.npy", df.iloc[:, -1].values)
