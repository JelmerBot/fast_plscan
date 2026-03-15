"""Regenerate tests/data/fixture_knn.npz.

Run once from the project root whenever the test data needs to be updated:

    python tests/generate_fixture.py
"""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

X, y = make_blobs(n_samples=200, random_state=10)
X, y = shuffle(X, y, random_state=7)
X = StandardScaler().fit_transform(X).astype(np.float32)

distances, indices = NearestNeighbors(n_neighbors=10).fit(X).kneighbors(X)

# Mark the last neighbor slot of the first 5 points as missing,
# confirming that knn_to_csr handles -1/-inf entries.
distances = distances.astype(np.float32)
indices = indices.astype(np.int32)
distances[0:5, -1] = np.inf
indices[0:5, -1] = -1

np.savez_compressed("tests/data/fixture_knn.npz", distances=distances, indices=indices)
print(f"Saved fixture_knn.npz  distances{distances.shape}  indices{indices.shape}")
