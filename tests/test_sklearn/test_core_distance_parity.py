"""Tests for core-distance parity across parameterizations."""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import pairwise_distances

from fast_plscan import PLSCAN

from ..conftest import numerical_balltree_metrics, boolean_metrics


@pytest.mark.parametrize(
    "space_tree,metric",
    [("kd_tree", m) for m in PLSCAN.VALID_KDTREE_METRICS]
    + [("ball_tree", m) for m in numerical_balltree_metrics],
)
def test_equal_core_distances(X, space_tree, metric):
    if metric == "braycurtis":
        pytest.skip("Don't compare balltree braycurtis against scipy braycurtis")

    metric_kws = dict()
    if metric in ["p", "minkowski"]:
        metric_kws["p"] = 2.5

    _metric = metric
    if _metric == "p":
        _metric = "minkowski"
    if _metric == "infinity":
        _metric = "chebyshev"

    c1 = PLSCAN(metric="precomputed").fit(
        pairwise_distances(X, metric=_metric, **metric_kws)
    )
    c2 = PLSCAN(metric=metric, space_tree=space_tree, metric_kws=metric_kws).fit(X)
    tolerance = 2e-3 if metric == "seuclidean" else 1e-08
    assert np.allclose(c1.core_distances_, c2.core_distances_, atol=tolerance)


@pytest.mark.parametrize("metric", boolean_metrics)
def test_equal_core_distances_boolean(X_bool, metric):
    metric_kws = dict()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataConversionWarning)
        dists = pairwise_distances(X_bool, metric=metric, **metric_kws)
    c1 = PLSCAN(metric="precomputed").fit(dists)
    c2 = PLSCAN(metric=metric, metric_kws=metric_kws).fit(X_bool)
    assert np.allclose(c1.core_distances_, c2.core_distances_)

