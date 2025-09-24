import numpy as np
import pandas as pd

# pandas read csv from URL:
url = "https://raw.githubusercontent.com/jadsoncastro/UnifiedView/refs/heads/master/data/articles_1442_80.data"
df = pd.read_csv(url, sep=",", header=None)
df.to_parquet(
    "docs/data/articles_1442_80/sources/articles_1442_80.parquet", index=False
)
np.save("docs/data/articles_1442_80/generated/X.npy", df.iloc[:, :-1].values)
np.save("docs/data/articles_1442_80/generated/y.npy", df.iloc[:, -1].values)
