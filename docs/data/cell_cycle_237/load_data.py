#

import requests
import numpy as np
import pandas as pd
from io import BytesIO

df = pd.read_csv(
    BytesIO(
        requests.get(
            "https://faculty.washington.edu/kayee/cluster/raw_cellcycle_384_17.txt"
        ).content
    ),
    sep="\t",
    index_col=0,
)

df.to_parquet("docs/data/cell_cycle_237/sources/cell_cycle.parquet")
np.save("docs/data/cell_cycle_237/generated/y.npy", df.iloc[:, 0].to_numpy())
np.save("docs/data/cell_cycle_237/generated/X.npy", df.iloc[:, 1:].to_numpy())
