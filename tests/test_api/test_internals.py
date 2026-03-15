"""Tests for the sklearn interface."""

import pytest
import numpy as np
from sklearn.neighbors._kd_tree import KDTree32
from sklearn.neighbors._ball_tree import BallTree32

from fast_plscan import PLSCAN
from fast_plscan._api import (
    SpaceTree,
    SpanningTree,
    kdtree_query,
    balltree_query,
    check_node_data,
    compute_linkage_tree,
    compute_condensed_tree,
)
from fast_plscan.api import clusters_from_spanning_forest

from ..conftest import numerical_balltree_metrics, duplicate_metrics, boolean_metrics
from ..checks import *


def _condensed_from_mst(parent, child, distance, num_points, min_cluster_size):
    """Build a condensed tree from raw MST arrays (must be sorted by distance)."""
    mst = SpanningTree(
        np.array(parent, dtype=np.uint32),
        np.array(child, dtype=np.uint32),
        np.array(distance, dtype=np.float32),
    )
    linkage = compute_linkage_tree(mst, num_points)
    return compute_condensed_tree(linkage, mst, num_points, min_cluster_size)


def test_equal_distance_edge_order():
    """Condensed tree should contain only the maximal connected components at each distance"""
    num_points = 8
    chains = [
        (0, 1, 0.5),
        (1, 2, 0.5),
        (2, 3, 0.5),
        (4, 5, 0.5),
        (5, 6, 0.5),
        (6, 7, 0.5),
        (3, 4, 0.5),
    ]
    groups = [
        (0, 1, 0.5),
        (2, 3, 0.5),
        (1, 2, 0.5),
        (4, 5, 0.5),
        (6, 7, 0.5),
        (5, 6, 0.5),
        (3, 4, 0.5),
    ]
    ct_a = _condensed_from_mst(*zip(*chains), num_points, min_cluster_size=2)
    ct_b = _condensed_from_mst(*zip(*groups), num_points, min_cluster_size=2)

    # Both orderings must produce the same number of cluster-segment rows.
    assert ct_a.cluster_rows.size == 2
    assert ct_a.cluster_rows.size == ct_b.cluster_rows.size


def test_clusters_from_empty_mst():
    empty = SpanningTree(
        np.array([], dtype=np.uint32),
        np.array([], dtype=np.uint32),
        np.array([], dtype=np.float32),
    )
    with pytest.raises(ValueError):
        clusters_from_spanning_forest(empty, 10)


def test_node_data_conversion(kdtree):
    _, _, node_data, _ = kdtree.get_arrays()
    converted_copy = check_node_data(node_data.view(np.float64))
    for (idx_start, idx_end, is_leaf, radius), n2 in zip(node_data, converted_copy):
        assert idx_start == n2.idx_start
        assert idx_end == n2.idx_end
        assert is_leaf == n2.is_leaf
        assert radius == n2.radius
    node_data["idx_start"][0] = -1
    assert converted_copy[0].idx_start != -1


@pytest.mark.parametrize(
    "space_tree,metric",
    [("kd_tree", m) for m in set(PLSCAN.VALID_KDTREE_METRICS) - duplicate_metrics]
    + [("ball_tree", m) for m in numerical_balltree_metrics - duplicate_metrics],
)
def test_space_tree_query(X, space_tree, metric):
    # Fill in defaults for parameterized metrics
    metric_kws = dict()
    if metric == "minkowski":
        metric_kws["p"] = 2.5
    elif metric == "seuclidean":
        metric_kws["V"] = np.var(X, axis=0)
    elif metric == "mahalanobis":
        metric_kws["VI"] = np.linalg.inv(np.cov(X, rowvar=False))

    if space_tree == "kd_tree":
        tree = KDTree32(X, metric=metric, **metric_kws)
        query_fun = kdtree_query
    else:
        tree = BallTree32(X, metric=metric, **metric_kws)
        query_fun = balltree_query

    data, idx_array, node_data, node_bounds = tree.get_arrays()
    dists, indices = tree.query(data, 10)

    knn_csr = query_fun(
        SpaceTree(data, idx_array, node_data.view(np.float64), node_bounds),
        10,
        metric,
        metric_kws,
    )

    _indices = knn_csr.indices.reshape(data.shape[0], 10)
    _dists = knn_csr.data.reshape(data.shape[0], 10)

    # Apply rdist to dist conversions!
    if metric in ["euclidean", "seuclidean", "mahalanobis"]:
        _dists = np.sqrt(_dists)
    elif metric == "minkowski":
        _dists = np.pow(_dists, 1 / metric_kws["p"])
    elif metric == "haversine":
        _dists = 2 * np.arcsin(np.sqrt(_dists))

    assert np.allclose(dists, _dists)
    assert np.allclose(indices, _indices)


@pytest.mark.parametrize("metric", boolean_metrics)
def test_ball_tree_boolean_query(X_bool, metric):
    # Fill in defaults for parameterized metrics
    metric_kws = dict()
    tree = BallTree32(X_bool, metric=metric, **metric_kws)
    data, idx_array, node_data, node_bounds = tree.get_arrays()

    dists, indices = tree.query(data, 10)
    knn_csr = balltree_query(
        SpaceTree(data, idx_array, node_data.view(np.float64), node_bounds),
        10,
        metric,
        metric_kws,
    )

    _indices = knn_csr.indices.reshape(data.shape[0], 10)
    _dists = knn_csr.data.reshape(data.shape[0], 10)

    assert np.allclose(dists, _dists)
    # Don't test indices, there can be draws!
