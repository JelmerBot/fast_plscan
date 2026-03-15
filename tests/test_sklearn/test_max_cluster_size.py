"""Tests for max_cluster_size parameter behavior and validation."""

import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from fast_plscan import PLSCAN

from ..checks import *


def test_max_cluster_size(X, knn):
    c = PLSCAN(metric="precomputed", min_samples=4, max_cluster_size=5).fit(knn)

    valid_fitted_clustering_state(
        c,
        X,
        expect_mutual_graph=True,
        expect_neighbors=False,
    )
    assert c.labels_.max() in [5, 6]
    assert np.any(c.labels_ == -1)


def test_bad_max_cluster_size_negative(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", max_cluster_size=-1).fit(knn)


def test_bad_max_cluster_size_equal_min_samples(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples=5, max_cluster_size=5).fit(knn)


def test_bad_max_cluster_size_string(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", max_cluster_size="bla").fit(knn)


def test_bad_max_cluster_size_list(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", max_cluster_size=[0.1, 0.2]).fit(knn)


def test_bad_max_cluster_size_none(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", max_cluster_size=None).fit(knn)

