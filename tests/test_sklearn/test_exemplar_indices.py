"""Tests for exemplar-index post-fit methods."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from fast_plscan import PLSCAN

from ..checks import *


def test_compute_exemplar_indices_basic(X):
    c = PLSCAN().fit(X)
    exemplars_per_cluster = c.compute_exemplar_indices()
    valid_exemplar_indices(exemplars_per_cluster, X, c.labels_)


def test_compute_exemplar_indices_with_custom_labels(X, knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    labels, _ = c.min_cluster_size_cut(c._persistence_trace.min_size[0])
    exemplars_per_cluster = c.compute_exemplar_indices(labels)
    valid_exemplar_indices(exemplars_per_cluster, X, labels)


def test_compute_exemplar_indices_all_noise(X):
    c = PLSCAN(min_cluster_size=X.shape[0]).fit(X)
    exemplars_per_cluster = c.compute_exemplar_indices()
    assert exemplars_per_cluster == []


def test_bad_compute_exemplar(X):
    c = PLSCAN().fit(X)
    labels = c.labels_.copy()
    labels[labels == 1] = 2
    with pytest.raises(ValueError):
        c.compute_exemplar_indices(np.zeros(5, dtype=np.int64))
    with pytest.raises(ValueError):
        c.compute_exemplar_indices(labels)


def test_not_fitted_compute_exemplar_indices_method():
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        c.compute_exemplar_indices()
