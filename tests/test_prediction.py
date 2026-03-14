"""Tests for the prediction module."""

import pytest
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.exceptions import NotFittedError

from fast_plscan import PLSCAN
from fast_plscan.prediction import (
    all_points_membership_vectors,
    approximate_predict,
    membership_vectors,
)
from fast_plscan._helpers import to_scipy_csr

from .checks import *
from .conftest import numerical_balltree_metrics, boolean_metrics, duplicate_metrics


def assert_zero_membership_across_components(c, mv):
    labels = c.labels_
    n_clusters = int(labels.max()) + 1
    if n_clusters <= 0:
        return

    g = to_scipy_csr(c._mutual_graph)
    g = g.maximum(g.T)
    _, component = connected_components(g, directed=False, return_labels=True)
    all_clusters = set(range(n_clusters))

    for comp in np.unique(component):
        point_mask = component == comp
        component_clusters = set(labels[point_mask][labels[point_mask] >= 0])
        other_clusters = sorted(all_clusters - component_clusters)
        if other_clusters:
            assert np.all(
                mv[point_mask][:, other_clusters] == 0
            ), "Memberships to clusters in other connected components should be zero"


# --- Inputs


def test_membership_vectors_basic(X):
    c = PLSCAN().fit(X)
    mv = all_points_membership_vectors(c)
    valid_membership_vectors(mv, X, c.labels_)
    non_noise = c.labels_ >= 0
    assert np.all(
        mv[non_noise].argmax(axis=1) == c.labels_[non_noise]
    ), "argmax of membership row must match cluster label for non-noise points"


def test_membership_vectors_precomputed_sparse(g_dists, X):
    """This is a fully connected input"""
    c = PLSCAN(metric="precomputed").fit(g_dists)
    mv = all_points_membership_vectors(c)
    valid_membership_vectors(mv, X, c.labels_)


def test_membership_vectors_precomputed_knn_graph(g_knn, X):
    """This is a minimum spanning forest input"""
    c = PLSCAN(metric="precomputed").fit(g_knn)
    mv = all_points_membership_vectors(c)
    valid_membership_vectors(mv, X, c.labels_)
    assert_zero_membership_across_components(c, mv)


def test_membership_vectors_precomputed_knn(knn, X):
    """This is a minimum spanning forest input"""
    c = PLSCAN(metric="precomputed").fit(knn)
    mv = all_points_membership_vectors(c)
    valid_membership_vectors(mv, X, c.labels_)
    assert_zero_membership_across_components(c, mv)


# --- Approximate prediction


def test_approximate_predict_basic(X):
    c = PLSCAN().fit(X)
    labels, probabilities = approximate_predict(c, X)
    valid_labels(labels, X)
    valid_probabilities(probabilities, X)


def test_approximate_predict_subset(X):
    c = PLSCAN().fit(X)
    X_new = X[:10]
    labels, probabilities = approximate_predict(c, X_new)
    valid_labels(labels, X_new)
    valid_probabilities(probabilities, X_new)


def test_membership_vectors_unseen_points_subset(X):
    c = PLSCAN().fit(X)
    X_new = X[:10]
    mv = membership_vectors(c, X_new)
    labels, _ = approximate_predict(c, X_new)
    valid_membership_vectors(mv, X_new, c.labels_)
    assigned = labels >= 0
    if np.any(assigned):
        assert np.all(mv[assigned].argmax(axis=1) == labels[assigned])


def test_membership_vectors_unseen_points_all_noise(X):
    c = PLSCAN(min_cluster_size=X.shape[0]).fit(X)
    X_new = X[:8]
    mv = membership_vectors(c, X_new)
    valid_membership_vectors(mv, X_new, c.labels_)


def test_membership_vectors_unseen_points_custom_labels(X):
    c = PLSCAN().fit(X)
    X_new = X[:10]
    labels, _ = c.min_cluster_size_cut(c._persistence_trace.min_size[0])
    mv = membership_vectors(c, X_new, labels)
    valid_membership_vectors(mv, X_new, labels)


# --- Parameters


def test_membership_vectors_with_custom_labels(X):
    c = PLSCAN().fit(X)
    labels, _ = c.min_cluster_size_cut(c._persistence_trace.min_size[0])
    mv = all_points_membership_vectors(c, labels)
    valid_membership_vectors(mv, X, labels)


# --- Edge cases


def test_membership_vectors_all_noise(X):
    # An enormous min_cluster_size forces all-noise output
    c = PLSCAN(min_cluster_size=X.shape[0]).fit(X)
    mv = all_points_membership_vectors(c)
    assert mv.shape == (X.shape[0], 0)


# --- Metric parametrization


@pytest.mark.parametrize(
    "metric",
    sorted(numerical_balltree_metrics - duplicate_metrics - {"haversine", "minkowski"}),
)
def test_membership_vectors_numerical_metrics(X, metric):
    c = PLSCAN(metric=metric, space_tree="ball_tree").fit(X)
    mv = all_points_membership_vectors(c)
    valid_membership_vectors(mv, X, c.labels_)


@pytest.mark.parametrize("metric", sorted(boolean_metrics))
def test_membership_vectors_boolean_metrics(X_bool, metric):
    c = PLSCAN(metric=metric, space_tree="ball_tree").fit(X_bool)
    mv = all_points_membership_vectors(c)
    valid_membership_vectors(mv, X_bool, c.labels_)


# --- Error paths


def test_bad_membership_vectors_precomputed_mst(X, mst):
    c = PLSCAN(metric="precomputed").fit((mst, X.shape[0]))
    with pytest.raises(ValueError):
        all_points_membership_vectors(c)


def test_bad_approximate_predict_precomputed_sparse(g_dists):
    c = PLSCAN(metric="precomputed").fit(g_dists)
    with pytest.raises(ValueError):
        approximate_predict(c, np.zeros((3, 2), dtype=np.float32))


def test_bad_membership_vectors_precomputed_sparse(g_dists):
    c = PLSCAN(metric="precomputed").fit(g_dists)
    with pytest.raises(ValueError):
        membership_vectors(c, np.zeros((3, 2), dtype=np.float32))


def test_bad_approximate_predict_precomputed_knn(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    with pytest.raises(ValueError):
        approximate_predict(c, np.zeros((3, 2), dtype=np.float32))


def test_bad_membership_vectors_precomputed_knn(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    with pytest.raises(ValueError):
        membership_vectors(c, np.zeros((3, 2), dtype=np.float32))


def test_bad_approximate_predict_precomputed_mst(X, mst):
    c = PLSCAN(metric="precomputed").fit((mst, X.shape[0]))
    with pytest.raises(ValueError):
        approximate_predict(c, np.zeros((3, 2), dtype=np.float32))


def test_bad_membership_vectors_precomputed_mst(X, mst):
    c = PLSCAN(metric="precomputed").fit((mst, X.shape[0]))
    with pytest.raises(ValueError):
        membership_vectors(c, np.zeros((3, 2), dtype=np.float32))


def test_bad_membership_vectors_unfitted():
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        all_points_membership_vectors(c)


def test_bad_approximate_predict_unfitted(X):
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        approximate_predict(c, X)


def test_bad_membership_vectors_unfitted(X):
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        membership_vectors(c, X)


def test_bad_membership_vectors_wrong_labels_shape(X):
    c = PLSCAN().fit(X)
    with pytest.raises(ValueError):
        all_points_membership_vectors(c, np.zeros(5, dtype=np.int32))


def test_bad_approximate_predict_wrong_num_features(X):
    c = PLSCAN().fit(X)
    with pytest.raises(ValueError):
        approximate_predict(c, np.zeros((5, X.shape[1] + 1), dtype=np.float32))


def test_bad_membership_vectors_wrong_num_features(X):
    c = PLSCAN().fit(X)
    with pytest.raises(ValueError):
        membership_vectors(c, np.zeros((5, X.shape[1] + 1), dtype=np.float32))


def test_bad_membership_vectors_wrong_labels_shape(X):
    c = PLSCAN().fit(X)
    with pytest.raises(ValueError):
        membership_vectors(c, X[:5], np.zeros(5, dtype=np.int64))
