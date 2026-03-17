"""Tests for cluster layer post-fit methods."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from fast_plscan import PLSCAN

from ..checks import *


def test_cluster_layers(X, knn):
    c = PLSCAN(min_samples=7, metric="precomputed").fit(knn)
    layers = c.cluster_layers()
    assert isinstance(layers, list)
    assert len(layers) == 1
    for x, labels, probabilities in layers:
        assert isinstance(x, np.float32)
        valid_labels(labels, X)
        valid_probabilities(probabilities, X)


def test_cluster_layers_params(X, knn):
    c = PLSCAN(min_samples=7, metric="precomputed").fit(knn)
    layers = c.cluster_layers(
        max_peaks=2, min_size=4.0, max_size=10.0, height=0.1, threshold=0.05
    )
    assert isinstance(layers, list)
    assert len(layers) == 1
    for x, labels, probabilities in layers:
        assert isinstance(x, np.float32)
        valid_labels(labels, X)
        valid_probabilities(probabilities, X)


def test_cluster_layers_no_peaks(X, knn):
    c = PLSCAN(min_samples=7, metric="precomputed").fit(knn)
    layers = c.cluster_layers(height=np.inf)
    assert layers == []


def test_not_fitted_cluster_layers_method():
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        c.cluster_layers()
