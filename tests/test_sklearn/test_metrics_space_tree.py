"""Tests for metric and space-tree parameter behavior and validation."""

import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from fast_plscan import PLSCAN

from ..checks import *
from ..conftest import boolean_metrics


@pytest.mark.parametrize("metric", PLSCAN.VALID_KDTREE_METRICS)
@pytest.mark.parametrize("space_tree", ["kd_tree", "ball_tree"])
def test_kdtree_l1_l2(X, metric, space_tree):
    metric_kws = dict()
    if metric in ["p", "minkowski"]:
        metric_kws["p"] = 2.5

    c = PLSCAN(space_tree=space_tree, metric=metric, metric_kws=metric_kws).fit(X)

    valid_fitted_clustering_state(
        c,
        X,
        expect_mutual_graph=False,
        expect_neighbors=True,
    )
    assert c.labels_.max() == 2


@pytest.mark.parametrize(
    "metric,num_clusters",
    [
        ("braycurtis", 2),
        ("mahalanobis", 2),
        ("canberra", 3),
        ("seuclidean", 2),
        ("haversine", 2),
    ],
)
def test_balltree_numerical_metrics(X, metric, num_clusters):
    c = PLSCAN(metric=metric).fit(X)

    valid_fitted_clustering_state(
        c,
        X,
        expect_mutual_graph=False,
        expect_neighbors=True,
    )
    assert c.labels_.max() == num_clusters


@pytest.mark.parametrize("metric", boolean_metrics)
def test_balltree_boolean_metrics(X_bool, metric):
    c = PLSCAN(metric=metric).fit(X_bool)

    valid_fitted_clustering_state(
        c,
        X_bool,
        expect_mutual_graph=False,
        expect_neighbors=True,
    )
    assert c.labels_.max() < 3


@pytest.mark.parametrize(
    "metric",
    sorted(set(PLSCAN.VALID_BALLTREE_METRICS) - set(PLSCAN.VALID_KDTREE_METRICS)),
)
def test_bad_space_tree_with_balltree_only_metric(X, metric):
    with pytest.raises(InvalidParameterError):
        PLSCAN(space_tree="kd_tree", metric=metric).fit(X)


def test_bad_space_tree_name(X):
    with pytest.raises(InvalidParameterError):
        PLSCAN(space_tree="bla").fit(X)


@pytest.mark.parametrize("metric", ["bla", "cosine", "sqeuclidean"])
def test_bad_metric_name(X, metric):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric=metric).fit(X)


@pytest.mark.parametrize("metric", [0, 2.0])
def test_bad_metric_type(X, metric):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric=metric).fit(X)


@pytest.mark.parametrize("metric", ["minkowski", "p"])
def test_bad_metric_missing_p_parameter(X, metric):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric=metric).fit(X)


def test_bad_metric_kws_without_metric(X):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric_kws=dict(p=2.0)).fit(X)


def test_bad_metric_minkowski_unknown_kw(X):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="minkowski", metric_kws=dict(c=2.0)).fit(X)


def test_bad_metric_minkowski_invalid_p(X):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="minkowski", metric_kws=dict(p=0.2)).fit(X)


def test_bad_metric_p_unknown_kw(X):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="p", metric_kws=dict(c=2.0)).fit(X)


def test_bad_metric_p_invalid_p(X):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="p", metric_kws=dict(p=0.2)).fit(X)


def test_seuclidean_user_V(X):
    V = np.var(X, axis=0)
    c1 = PLSCAN(metric="seuclidean").fit(X)
    c2 = PLSCAN(metric="seuclidean", metric_kws={"V": V}).fit(X)
    assert np.allclose(c1.core_distances_, c2.core_distances_, atol=2e-3)


def test_mahalanobis_user_VI(X):
    VI = np.linalg.inv(np.cov(X, rowvar=False))
    c1 = PLSCAN(metric="mahalanobis").fit(X)
    c2 = PLSCAN(metric="mahalanobis", metric_kws={"VI": VI}).fit(X)
    assert np.allclose(c1.core_distances_, c2.core_distances_)
