import os
import zipfile
import requests
import numpy as np
import pandas as pd

# Download and extract the data
response = requests.get("https://archive.ics.uci.edu/static/public/39/ecoli.zip")
with open("docs/data/ecoli/sources/data.zip", "wb") as f:
    f.write(response.content)
with zipfile.ZipFile("docs/data/ecoli/sources/data.zip") as z:
    z.extract("ecoli.data", path="docs/data/ecoli/sources/")
os.remove("docs/data/ecoli/sources/data.zip")

# Convert to X, y format and save as numpy arrays
data = pd.read_csv(
    "docs/data/ecoli/sources/ecoli.data", header=None, index_col=0, sep=r"\s+"
)
X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].astype("category").cat.codes.to_numpy()
np.save("docs/data/ecoli/generated/y.npy", y)
np.save("docs/data/ecoli/generated/X.npy", X)
