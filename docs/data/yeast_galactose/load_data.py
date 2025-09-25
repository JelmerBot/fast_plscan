import requests
import numpy as np
import pandas as pd
from io import BytesIO

df = pd.read_csv(
    BytesIO(
        requests.get(
            "https://static-content.springer.com/esm/art%3A10.1186%2Fgb-2003-4-5-r34/MediaObjects/13059_2002_544_MOESM8_ESM.txt"
        ).content
    ),
    sep="\t",
)

df.to_parquet("docs/data/yeast_galactose/sources/knn12_gal205.parquet")
np.save("docs/data/yeast_galactose/generated/y.npy", df["class"].to_numpy())
np.save("docs/data/yeast_galactose/generated/X.npy", df.iloc[:, 3:].to_numpy())
