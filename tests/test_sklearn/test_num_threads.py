"""Tests for num_threads parameter behavior and validation."""

import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from fast_plscan import PLSCAN

from ..checks import valid_fitted_clustering_state


def test_num_threads(X, knn):
    c = PLSCAN(metric="precomputed", num_threads=2).fit(knn)

    valid_fitted_clustering_state(
        c,
        X,
        expect_mutual_graph=True,
        expect_neighbors=False,
    )
    assert c.labels_.max() == 3
    assert np.any(c.labels_ == -1)


def test_bad_num_threads_negative(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", num_threads=-1).fit(knn)


def test_bad_num_threads_zero(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", num_threads=0).fit(knn)


def test_bad_num_threads_float(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", num_threads=2.6).fit(knn)


def test_bad_num_threads_string(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", num_threads="bla").fit(knn)


def test_bad_num_threads_list(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", num_threads=[0.1, 0.2]).fit(knn)
