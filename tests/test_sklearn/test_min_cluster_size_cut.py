"""Tests for min-cluster-size cut post-fit method."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from fast_plscan import PLSCAN

from ..checks import *


def test_min_cluster_size_cut(X, knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    labels, probs = c.min_cluster_size_cut(7.0)
    valid_labels(labels, X)
    valid_probabilities(probs, X)


def test_min_cluster_size_cut_edge_values(X, knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    labels, probs = c.min_cluster_size_cut(2.0)
    valid_labels(labels, X)
    valid_probabilities(probs, X)
    labels, probs = c.min_cluster_size_cut(1e9)
    valid_labels(labels, X)
    valid_probabilities(probs, X)
    assert np.all(labels == -1)


def test_not_fitted_min_cluster_size_cut_method():
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        c.min_cluster_size_cut(6.0)
