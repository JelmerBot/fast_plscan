"""Tests for distance cut post-fit method."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from fast_plscan import PLSCAN

from ..checks import valid_labels, valid_probabilities


def test_distance_cut(X, knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    labels, probs = c.distance_cut(0.5)
    valid_labels(labels, X)
    valid_probabilities(probs, X)


def test_distance_cut_edge_values(X, knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    labels, probs = c.distance_cut(0.0)
    valid_labels(labels, X)
    valid_probabilities(probs, X)
    assert np.all(labels == -1)
    labels, probs = c.distance_cut(np.inf)
    valid_labels(labels, X)
    valid_probabilities(probs, X)


def test_not_fitted_distance_cut_method():
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        c.distance_cut(0.5)
