"""Tests for min_samples parameter behavior and validation."""

import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from fast_plscan import PLSCAN

from ..checks import *


def test_min_samples(X, dists):
    c = PLSCAN(metric="precomputed", min_samples=70).fit(dists)

    valid_fitted_clustering_state(
        c,
        X,
        expect_mutual_graph=True,
        expect_neighbors=False,
    )
    assert np.all(c.labels_ == -1)


def test_bad_min_samples_too_large_for_features(X):
    with pytest.raises(ValueError):
        PLSCAN(min_samples=X.shape[0]).fit(X)


def test_bad_min_samples_too_large_for_precomputed(X, knn):
    with pytest.raises(ValueError):
        PLSCAN(metric="precomputed", min_samples=X.shape[0] - 1).fit(knn)


def test_bad_min_samples_negative(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples=-1).fit(knn)


def test_bad_min_samples_zero(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples=0).fit(knn)


def test_bad_min_samples_one(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples=1).fit(knn)


def test_bad_min_samples_float(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples=2.5).fit(knn)


def test_bad_min_samples_string(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples="bla").fit(knn)


def test_bad_min_samples_list(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples=[0.1, 0.2]).fit(knn)


def test_bad_min_samples_none(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples=None).fit(knn)

