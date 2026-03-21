"""Tests for min_cluster_size parameter behavior and validation."""

import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from fast_plscan import PLSCAN

from ..checks import valid_fitted_clustering_state


def test_min_cluster_size(X, dists):
    c = PLSCAN(metric="precomputed", min_cluster_size=15).fit(dists)

    valid_fitted_clustering_state(
        c,
        X,
        expect_mutual_graph=True,
        expect_neighbors=False,
    )
    assert c.labels_.max() == 2
    assert np.all(c.labels_ > -1)


def test_bad_min_cluster_size_negative(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_cluster_size=-1).fit(knn)


def test_bad_min_cluster_size_smaller_than_min_samples(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples=5, min_cluster_size=4).fit(knn)


def test_bad_min_cluster_size_infinite(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_cluster_size=np.inf).fit(knn)


def test_bad_min_cluster_size_string(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_cluster_size="bla").fit(knn)


def test_bad_min_cluster_size_list(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_cluster_size=[0.1, 0.2]).fit(knn)
