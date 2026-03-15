"""Regenerate stored test data under tests/data/.

Run once from the project root whenever the test data needs to be updated:

    python tests/generate_fixture.py
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# ---- X (float32 feature matrix) ----
X, y = make_blobs(n_samples=200, random_state=10)
X, y = shuffle(X, y, random_state=7)
X = StandardScaler().fit_transform(X).astype(np.float32)
np.save("tests/data/X.npy", X)
print(f"Saved X.npy  {X.shape}  {X.dtype}")

# ---- X_bool (boolean feature matrix as float32) ----
p = 0.25
rng = np.random.Generator(np.random.PCG64(10))
X_bool = rng.choice(a=[True, False], size=(200, 100), p=[p, 1 - p]).astype(np.float32)
np.save("tests/data/X_bool.npy", X_bool)
print(f"Saved X_bool.npy  {X_bool.shape}  {X_bool.dtype}")

# ---- knn (10-NN with some missing slots) ----
distances, indices = NearestNeighbors(n_neighbors=10).fit(X).kneighbors(X)
distances = distances.astype(np.float32)
indices = indices.astype(np.int32)
distances[0:5, -1] = np.inf
indices[0:5, -1] = -1
np.savez_compressed("tests/data/fixture_knn.npz", distances=distances, indices=indices)
print(f"Saved fixture_knn.npz  distances{distances.shape}  indices{indices.shape}")

# ---- knn_no_loops (8-NN without self-loops, with some missing slots) ----
distances_nl, indices_nl = NearestNeighbors(n_neighbors=8).fit(X).kneighbors()
distances_nl = distances_nl.astype(np.float32)
indices_nl = indices_nl.astype(np.int32)
distances_nl[0:5, -1] = np.inf
indices_nl[0:5, -1] = -1
np.savez_compressed(
    "tests/data/knn_no_loops.npz", distances=distances_nl, indices=indices_nl
)
print(
    f"Saved knn_no_loops.npz  distances{distances_nl.shape}  indices{indices_nl.shape}"
)
