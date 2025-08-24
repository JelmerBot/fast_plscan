import pytest
import warnings
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.exceptions import DataConversionWarning

from plscan._lib import get_dist

from .conftest import numerical_balltree_metrics, duplicate_metrics, boolean_metrics


@pytest.mark.parametrize("metric", [*numerical_balltree_metrics - duplicate_metrics])
def test_numerical_distance(X, metric):
    if metric == "braycurtis":
        pytest.skip("Don't compare balltree braycurtis against scipy braycurtis.")

    metric_kws = dict()
    if metric == "minkowski":
        metric_kws["p"] = 2.5
    if metric == "seuclidean":
        metric_kws["V"] = np.var(X, axis=0)
    if metric == "mahalanobis":
        metric_kws["VI"] = np.linalg.inv(np.cov(X, rowvar=False))

    d1 = pairwise_distances(X, metric=metric, **metric_kws)
    d2 = pairwise_distances(X, metric=get_dist(metric, **metric_kws))

    assert np.allclose(d2, d1)


@pytest.mark.parametrize("metric", boolean_metrics)
def test_boolean_distance(X_bool, metric):
    metric_kws = dict()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataConversionWarning)
        d1 = pairwise_distances(X_bool, metric=metric, **metric_kws)
    d2 = pairwise_distances(X_bool, metric=get_dist(metric, **metric_kws))
    if metric == "russellrao":
        np.fill_diagonal(d2, 0.0)

    assert np.allclose(d2, d1)
