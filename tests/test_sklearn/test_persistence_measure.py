"""Tests for persistence_measure parameter behavior and validation."""

import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from fast_plscan import PLSCAN

from ..checks import *


@pytest.mark.parametrize(
    "persistence_measure", ["distance", "density", "size-density", "size-distance"]
)
def test_persistence_measure(X, knn, persistence_measure):
    c = PLSCAN(metric="precomputed", persistence_measure=persistence_measure).fit(knn)

    valid_fitted_clustering_state(
        c,
        X,
        expect_mutual_graph=True,
        expect_neighbors=False,
    )
    assert c.labels_.max() <= 3
    assert np.any(c.labels_ == -1)


def test_bad_persistence_measure_int(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", persistence_measure=1).fit(knn)


def test_bad_persistence_measure_float(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", persistence_measure=2.0).fit(knn)


def test_bad_persistence_measure_string(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", persistence_measure="bla").fit(knn)


def test_bad_persistence_measure_list(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", persistence_measure=[0.1, 0.2]).fit(knn)


def test_bad_persistence_measure_none(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", persistence_measure=None).fit(knn)
