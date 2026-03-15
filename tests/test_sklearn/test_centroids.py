"""Tests for centroid post-fit methods."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from fast_plscan import PLSCAN

from ..checks import *


def test_centroids(X):
    c = PLSCAN().fit(X)
    centroids = c.compute_centroids()
    expected = np.array(
        [[-0.08430787, 1.345701], [-1.1071285, -0.3461533], [1.1412238, -1.0004447]],
        dtype=np.float32,
    )
    valid_centroids(centroids, X, c.labels_)
    assert np.allclose(centroids, expected)


def test_centroids_precomputed_raises(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    with pytest.raises(ValueError):
        c.compute_centroids()


def test_not_fitted_compute_centroids_method():
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        c.compute_centroids()
