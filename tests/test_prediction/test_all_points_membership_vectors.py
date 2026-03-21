"""Tests for all_points_membership_vectors."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from fast_plscan import PLSCAN
from fast_plscan.prediction import all_points_membership_vectors

from ..checks import valid_membership_vectors, assert_zero_membership_across_components
from ..conftest import numerical_balltree_metrics, boolean_metrics, duplicate_metrics

# --- Positive Input Modes


def test_membership_vectors_basic(X):
    c = PLSCAN().fit(X)
    mv = all_points_membership_vectors(c)
    valid_membership_vectors(mv, X, c.labels_)
    non_noise = c.labels_ >= 0
    assert np.all(
        mv[non_noise].argmax(axis=1) == c.labels_[non_noise]
    ), "argmax of membership row must match cluster label for non-noise points"


def test_membership_vectors_precomputed_sparse(g_dists, X):
    c = PLSCAN(metric="precomputed").fit(g_dists)
    mv = all_points_membership_vectors(c)
    valid_membership_vectors(mv, X, c.labels_)


def test_membership_vectors_precomputed_knn_graph(g_knn, X):
    c = PLSCAN(metric="precomputed").fit(g_knn)
    mv = all_points_membership_vectors(c)
    valid_membership_vectors(mv, X, c.labels_)
    assert_zero_membership_across_components(c, mv)


def test_membership_vectors_precomputed_knn(knn, X):
    c = PLSCAN(metric="precomputed").fit(knn)
    mv = all_points_membership_vectors(c)
    valid_membership_vectors(mv, X, c.labels_)
    assert_zero_membership_across_components(c, mv)


# --- Negative Input Modes


def test_bad_membership_vectors_precomputed_mst(X, mst):
    c = PLSCAN(metric="precomputed").fit((mst, X.shape[0]))
    with pytest.raises(ValueError):
        all_points_membership_vectors(c)


def test_bad_membership_vectors_unfitted():
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        all_points_membership_vectors(c)


# --- Parameters


def test_membership_vectors_with_custom_labels(X):
    c = PLSCAN().fit(X)
    labels, _ = c.min_cluster_size_cut(c._persistence_trace.min_size[0])
    mv = all_points_membership_vectors(c, labels)
    valid_membership_vectors(mv, X, labels)


def test_membership_vectors_all_noise(X):
    c = PLSCAN(min_cluster_size=X.shape[0]).fit(X)
    mv = all_points_membership_vectors(c)
    assert mv.shape == (X.shape[0], 0)


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


def test_bad_membership_vectors_wrong_labels_shape(X):
    c = PLSCAN().fit(X)
    with pytest.raises(ValueError):
        all_points_membership_vectors(c, np.zeros(5, dtype=np.int32))
