"""Tests for sample_weights parameter behavior and validation."""

import numpy as np
import pytest

from fast_plscan import PLSCAN

from ..checks import valid_fitted_clustering_state


def test_sample_weights(X, knn):
    sample_weights = np.full(X.shape[0], 0.5, dtype=np.float32)
    sample_weights[:10] = 1.0
    sample_weights[-10:] = 2.0
    c = PLSCAN(metric="precomputed").fit(knn, sample_weights=sample_weights)

    valid_fitted_clustering_state(
        c,
        X,
        expect_mutual_graph=True,
        expect_neighbors=False,
    )
    assert c.labels_.max() == 3
    assert np.any(c.labels_ == -1)


def test_bad_sample_weights_wrong_length(X, knn):
    with pytest.raises(ValueError):
        sample_weights = np.full(X.shape[0] - 1, 0.5, dtype=np.float32)
        PLSCAN(metric="precomputed").fit(knn, sample_weights=sample_weights)


def test_bad_sample_weights_negative_value(X, knn):
    with pytest.raises(ValueError):
        sample_weights = np.full(X.shape[0], -0.5, dtype=np.float32)
        PLSCAN(metric="precomputed").fit(knn, sample_weights=sample_weights)


def test_bad_sample_weights_nan_value(X, knn):
    with pytest.raises(ValueError):
        sample_weights = np.full(X.shape[0], 0.5, dtype=np.float32)
        sample_weights[0] = np.nan
        PLSCAN(metric="precomputed").fit(knn, sample_weights=sample_weights)


def test_bad_sample_weights_inf_value(X, knn):
    with pytest.raises(ValueError):
        sample_weights = np.full(X.shape[0], 0.5, dtype=np.float32)
        sample_weights[0] = np.inf
        PLSCAN(metric="precomputed").fit(knn, sample_weights=sample_weights)


def test_bad_sample_weights_exceeds_min_cluster_size(X, knn):
    with pytest.raises(ValueError):
        sample_weights = np.full(X.shape[0], 0.5, dtype=np.float32)
        sample_weights[0] = 10.0
        PLSCAN(metric="precomputed", min_cluster_size=5.0).fit(
            knn, sample_weights=sample_weights
        )
