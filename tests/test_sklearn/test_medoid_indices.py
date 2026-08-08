"""Tests for medoid-index post-fit methods."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from fast_plscan import PLSCAN

from ..checks import valid_medoid_indices


def test_medoid_indices(X):
    c = PLSCAN().fit(X)
    medoid_indices = c.compute_medoid_indices()
    valid_medoid_indices(medoid_indices, X, c.labels_)


def test_medoid_indices_precomputed(X, g_knn):
    c = PLSCAN(metric="precomputed").fit(g_knn)
    medoid_indices = c.compute_medoid_indices()
    valid_medoid_indices(medoid_indices, X, c.labels_)


def test_medoid_indices_mst_raises(X, mst):
    c = PLSCAN(metric="precomputed").fit((mst, X.shape[0]))
    with pytest.raises(ValueError):
        c.compute_medoid_indices()


def test_medoid_indices_wrong_labels_dtype(X):
    c = PLSCAN().fit(X)
    with pytest.raises(ValueError):
        c.compute_medoid_indices(c.labels_.astype(np.float64))


def test_not_fitted_compute_medoid_indices_method():
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        c.compute_medoid_indices()
