"""Tests for the sklearn interface."""

import pytest
import numpy as np
from sklearn.neighbors._kd_tree import KDTree32
from sklearn.neighbors._ball_tree import BallTree32

from fast_plscan import PLSCAN
from fast_plscan._api import (
    Labelling,
    SpanningTree,
    SpaceTree,
    kdtree_query,
    balltree_query,
    check_node_data,
)
from fast_plscan.api import clusters_from_spanning_forest

from ..conftest import numerical_balltree_metrics, duplicate_metrics, boolean_metrics


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
        n2_start, n2_end, n2_leaf, n2_radius = n2  # covers NodeData.__iter__
        assert idx_start == n2_start
        assert idx_end == n2_end
        assert is_leaf == n2_leaf
        assert radius == n2_radius
    node_data["idx_start"][0] = -1
    assert converted_copy[0].idx_start != -1


def test_space_tree_iter(kdtree):
    data, idx_array, node_data, node_bounds = kdtree.get_arrays()
    tree = SpaceTree(data, idx_array, node_data.view(np.float64), node_bounds)
    d, idx, nd, nb = tree  # covers SpaceTree.__iter__
    assert np.array_equal(d, data)
    assert np.array_equal(idx, idx_array)
    assert np.array_equal(nd, node_data.view(np.float64), equal_nan=True)
    assert np.array_equal(nb, node_bounds)


def test_c_type_pickles(kdtree):
    import pickle

    # Labelling: Python-side constructor (covers Labelling.__init__ asarray path)
    # and pickle round-trip (covers Labelling.__reduce__)
    lab = Labelling(
        np.array([0, 1, -1], dtype=np.int32),
        np.array([0.9, 0.5, 0.0], dtype=np.float32),
    )
    lab2 = pickle.loads(pickle.dumps(lab))
    assert np.array_equal(lab.label, lab2.label)
    assert np.array_equal(lab.probability, lab2.probability)

    # NodeData: pickle round-trip (covers NodeData.__reduce__)
    _, _, raw_node_data, _ = kdtree.get_arrays()
    node_list = check_node_data(raw_node_data.view(np.float64))
    nd = node_list[0]
    nd2 = pickle.loads(pickle.dumps(nd))
    assert nd.idx_start == nd2.idx_start
    assert nd.idx_end == nd2.idx_end
    assert nd.is_leaf == nd2.is_leaf
    assert nd.radius == nd2.radius

    # SpaceTree: pickle round-trip (covers SpaceTree.__reduce__)
    data, idx_array, raw_node_data, node_bounds = kdtree.get_arrays()
    tree = SpaceTree(data, idx_array, raw_node_data.view(np.float64), node_bounds)
    tree2 = pickle.loads(pickle.dumps(tree))
    d2, idx2, nd2, nb2 = tree2
    assert np.array_equal(d2, data)
    assert np.array_equal(idx2, idx_array)
    assert np.array_equal(nd2, raw_node_data.view(np.float64), equal_nan=True)


@pytest.mark.parametrize(
    "metric,expected_message",
    [
        ("hamming", "Missing KDTree query"),  # valid metric, not in KDTree lookup
        ("bogus_metric", "Unsupported metric"),  # invalid metric name entirely
    ],
)
def test_kdtree_query_invalid_metrics(kdtree, metric, expected_message):
    data, idx_array, node_data, node_bounds = kdtree.get_arrays()
    tree = SpaceTree(data, idx_array, node_data.view(np.float64), node_bounds)
    with pytest.raises(ValueError, match=expected_message):
        kdtree_query(tree, 5, metric, {})


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
