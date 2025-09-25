import numpy as np
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("docs/data/elegans/sources/aligned_data.csv", index_col=0)
    metadata = pd.read_csv("docs/data/elegans/sources/cell_metadata.csv", index_col=0)

    X = df.to_numpy()
    cell_type = metadata["cell.type"][df.index]
    type_label = cell_type.astype("category").cat.codes.to_numpy()

    np.save("docs/data/elegans/generated/X.npy", X)
    np.save("docs/data/elegans/generated/y.npy", type_label)
