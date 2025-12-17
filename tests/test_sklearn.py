"""Tests for the sklearn interface."""

import pytest
import warnings
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.exceptions import NotFittedError, DataConversionWarning

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import pandas as pd
except ImportError:
    pd = None

from fast_plscan import PLSCAN

from .checks import *
from .conftest import numerical_balltree_metrics, boolean_metrics


# Valid non-feature vector inputs


def test_mst(X, mst):
    _in = mst.copy()
    c = PLSCAN(metric="precomputed").fit((mst, X.shape[0]))
    assert np.allclose(mst, _in)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    assert c._mutual_graph is None
    assert c._neighbors is None
    assert c.core_distances_ is None
    valid_labels(c.labels_, X)
    assert c.labels_.max() == 2
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    valid_condensed(c._condensed_tree, X)
    valid_linkage(c._linkage_tree, X)


def test_knn_graph(X, knn):
    """A knn matrix with the self-loop first column should produce the same
    results as one without the self-loop first column."""
    _in = (knn[0].copy(), knn[1].copy())
    c = PLSCAN(metric="precomputed").fit(knn)
    assert np.allclose(knn[0], _in[0])
    assert np.allclose(knn[1], _in[1])

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    assert c._neighbors is None
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert c.labels_.max() == 3
    assert np.any(c.labels_ == -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    valid_condensed(c._condensed_tree, X)
    valid_linkage(c._linkage_tree, X)


def test_knn_graph_no_loops(X, knn_no_loops):
    """A knn matrix without the self-loop first column should produce the same
    results as one with the self-loop first column."""
    _in = (knn_no_loops[0].copy(), knn_no_loops[1].copy())
    c = PLSCAN(metric="precomputed").fit(knn_no_loops)
    assert np.allclose(knn_no_loops[0], _in[0])
    assert np.allclose(knn_no_loops[1], _in[1])

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    assert c._neighbors is None
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert c.labels_.max() == 3
    assert np.any(c.labels_ == -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    valid_condensed(c._condensed_tree, X)
    valid_linkage(c._linkage_tree, X)


def test_distance_matrix(X, dists):
    _in = dists.copy()
    c = PLSCAN(metric="precomputed").fit(dists)
    assert np.allclose(dists, _in)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X)
    assert c._neighbors is None
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert c.labels_.max() == 2
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    valid_condensed(c._condensed_tree, X)
    valid_linkage(c._linkage_tree, X)


def test_condensed_matrix(X, con_dists):
    _in = con_dists.copy()
    c = PLSCAN(metric="precomputed").fit(con_dists)
    assert np.allclose(con_dists, _in)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X)
    assert c._neighbors is None
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert c.labels_.max() == 2
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    valid_condensed(c._condensed_tree, X)
    valid_linkage(c._linkage_tree, X)


def test_sparse_matrix(X, g_knn):
    _in = g_knn.copy()
    c = PLSCAN(metric="precomputed").fit(g_knn)
    assert np.allclose(g_knn.data, _in.data)
    assert np.allclose(g_knn.indices, _in.indices)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    assert c._neighbors is None
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert c.labels_.max() == 3
    assert np.any(c.labels_ == -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    valid_condensed(c._condensed_tree, X)
    valid_linkage(c._linkage_tree, X)


# Feature vector inputs with metrics valid in both space trees


@pytest.mark.parametrize("metric", PLSCAN.VALID_KDTREE_METRICS)
@pytest.mark.parametrize("space_tree", ["kd_tree", "ball_tree"])
def test_kdtree_l1_l2(X, metric, space_tree):
    # Fill in defaults for parameterized metrics
    metric_kws = dict()
    if metric in ["p", "minkowski"]:
        metric_kws["p"] = 2.5

    c = PLSCAN(space_tree=space_tree, metric=metric, metric_kws=metric_kws).fit(X)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    assert c._mutual_graph is None
    valid_core_distances(c.core_distances_, X)
    valid_neighbor_indices(c._neighbors, X, c.min_samples)
    valid_labels(c.labels_, X)
    assert c.labels_.max() == 2
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    valid_condensed(c._condensed_tree, X)
    valid_linkage(c._linkage_tree, X)


#  Feature vector inputs with metrics valid in ball trees


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

    valid_spanning_forest(c._minimum_spanning_tree, X)
    assert c._mutual_graph is None
    valid_core_distances(c.core_distances_, X)
    valid_neighbor_indices(c._neighbors, X, c.min_samples)
    valid_labels(c.labels_, X)
    assert c.labels_.max() == num_clusters
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    valid_condensed(c._condensed_tree, X)
    valid_linkage(c._linkage_tree, X)


@pytest.mark.parametrize("metric", boolean_metrics)
def test_balltree_boolean_metrics(X_bool, metric):
    c = PLSCAN(metric=metric).fit(X_bool)

    valid_spanning_forest(c._minimum_spanning_tree, X_bool)
    assert c._mutual_graph is None
    valid_core_distances(c.core_distances_, X_bool)
    valid_neighbor_indices(c._neighbors, X_bool, c.min_samples)
    valid_labels(c.labels_, X_bool)
    assert c.labels_.max() < 3  # ties can change num clusters!
    valid_probabilities(c.probabilities_, X_bool)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    valid_condensed(c._condensed_tree, X_bool)
    valid_linkage(c._linkage_tree, X_bool)


# Equal core distances for precomputed and space trees on all metrics


@pytest.mark.parametrize(
    "space_tree,metric",
    [("kd_tree", m) for m in PLSCAN.VALID_KDTREE_METRICS]
    + [("ball_tree", m) for m in numerical_balltree_metrics],
)
def test_equal_core_distances(X, space_tree, metric):
    if metric == "braycurtis":
        pytest.skip("Don't compare balltree braycurtis against scipy braycurtis")

    # Fill in defaults for parameterized metrics
    metric_kws = dict()
    if metric in ["p", "minkowski"]:
        metric_kws["p"] = 2.5

    # Pairwise distances does not support all names we support
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


# Parameters


def test_bad_space_tree(X):
    with pytest.raises(InvalidParameterError):
        PLSCAN(space_tree="bla").fit(X)
    for metric in set(PLSCAN.VALID_BALLTREE_METRICS) - set(PLSCAN.VALID_KDTREE_METRICS):
        with pytest.raises(InvalidParameterError):
            PLSCAN(space_tree="kd_tree", metric=metric).fit(X)


def test_bad_metrics(X):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="bla").fit(X)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="cosine").fit(X)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="sqeuclidean").fit(X)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric=0).fit(X)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric=2.0).fit(X)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="minkowski").fit(X)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="p").fit(X)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric_kws=dict(p=2.0)).fit(X)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="minkowski", metric_kws=dict(c=2.0)).fit(X)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="minkowski", metric_kws=dict(p=0.2)).fit(X)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="p", metric_kws=dict(c=2.0)).fit(X)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="p", metric_kws=dict(p=0.2)).fit(X)


def test_max_cluster_size(X, knn):
    c = PLSCAN(metric="precomputed", min_samples=4, max_cluster_size=5).fit(knn)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert c.labels_.max() == 5
    assert np.any(c.labels_ == -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    valid_condensed(c._condensed_tree, X)
    valid_linkage(c._linkage_tree, X)


def test_bad_max_cluster_size(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", max_cluster_size=-1).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples=5, max_cluster_size=5).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", max_cluster_size="bla").fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", max_cluster_size=[0.1, 0.2]).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", max_cluster_size=None).fit(knn)


def test_min_cluster_size(X, dists):
    c = PLSCAN(metric="precomputed", min_cluster_size=15).fit(dists)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert c.labels_.max() == 2
    assert np.all(c.labels_ > -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    valid_condensed(c._condensed_tree, X)
    valid_linkage(c._linkage_tree, X)


def test_bad_min_cluster_size(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_cluster_size=-1).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples=5, min_cluster_size=4).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_cluster_size=np.inf).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_cluster_size="bla").fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_cluster_size=[0.1, 0.2]).fit(knn)


def test_min_samples(X, dists):
    c = PLSCAN(metric="precomputed", min_samples=70).fit(dists)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert np.all(c.labels_ == -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    valid_condensed(c._condensed_tree, X)
    valid_linkage(c._linkage_tree, X)


def test_bad_min_samples(X, knn):
    with pytest.raises(ValueError):
        PLSCAN(min_samples=X.shape[0]).fit(X)
    with pytest.raises(ValueError):
        PLSCAN(metric="precomputed", min_samples=X.shape[0] - 1).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples=-1).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples=0).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples=1).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples=2.5).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples="bla").fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples=[0.1, 0.2]).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", min_samples=None).fit(knn)


@pytest.mark.parametrize(
    "persistence_measure", ["distance", "density", "size-density", "size-distance"]
)
def test_persistence_measure(X, knn, persistence_measure):
    c = PLSCAN(metric="precomputed", persistence_measure=persistence_measure).fit(knn)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert c.labels_.max() <= 3
    assert np.any(c.labels_ == -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    valid_condensed(c._condensed_tree, X)
    valid_linkage(c._linkage_tree, X)


def test_bad_persistence_measure(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", persistence_measure=1).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", persistence_measure=2.0).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", persistence_measure="bla").fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", persistence_measure=[0.1, 0.2]).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", persistence_measure=None).fit(knn)


def test_num_threads(X, knn):
    c = PLSCAN(metric="precomputed", num_threads=2).fit(knn)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert c.labels_.max() == 3
    assert np.any(c.labels_ == -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    valid_condensed(c._condensed_tree, X)
    valid_linkage(c._linkage_tree, X)


def test_bad_num_threads(knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", num_threads=-1).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", num_threads=0).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", num_threads=2.6).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", num_threads="bla").fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(metric="precomputed", num_threads=[0.1, 0.2]).fit(knn)


def test_sample_weights(X, knn):
    sample_weights = np.full(X.shape[0], 0.5, dtype=np.float32)
    sample_weights[:10] = 1.0
    sample_weights[-10:] = 2.0
    c = PLSCAN(metric="precomputed").fit(knn, sample_weights=sample_weights)

    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_mutual_graph(c._mutual_graph, X, missing=True)
    valid_core_distances(c.core_distances_, X)
    valid_labels(c.labels_, X)
    assert c.labels_.max() == 3
    assert np.any(c.labels_ == -1)
    valid_probabilities(c.probabilities_, X)
    valid_selected_clusters(c.selected_clusters_, c.labels_)
    valid_persistence_trace(c._persistence_trace)
    valid_leaf(c._leaf_tree)
    valid_condensed(c._condensed_tree, X)
    valid_linkage(c._linkage_tree, X)


def test_bad_sample_weights(X, knn):
    with pytest.raises(ValueError):
        sample_weights = np.full(X.shape[0] - 1, 0.5, dtype=np.float32)
        PLSCAN(metric="precomputed").fit(knn, sample_weights=sample_weights)
    with pytest.raises(ValueError):
        sample_weights = np.full(X.shape[0], -0.5, dtype=np.float32)
        PLSCAN(metric="precomputed").fit(knn, sample_weights=sample_weights)
    with pytest.raises(ValueError):
        sample_weights = np.full(X.shape[0], 0.5, dtype=np.float32)
        sample_weights[0] = np.nan
        PLSCAN(metric="precomputed").fit(knn, sample_weights=sample_weights)
    with pytest.raises(ValueError):
        sample_weights = np.full(X.shape[0], 0.5, dtype=np.float32)
        sample_weights[0] = np.inf
        PLSCAN(metric="precomputed").fit(knn, sample_weights=sample_weights)
    with pytest.raises(ValueError):
        sample_weights = np.full(X.shape[0], 0.5, dtype=np.float32)
        sample_weights[0] = 10.0
        PLSCAN(metric="precomputed", min_cluster_size=5.0).fit(
            knn, sample_weights=sample_weights
        )


# Attributes


@pytest.mark.skipif(nx is None, reason="networkx not installed")
def test_export_networkx(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    g = c.condensed_tree_.to_networkx()
    assert isinstance(g, nx.DiGraph)
    g = c.leaf_tree_.to_networkx()
    assert isinstance(g, nx.DiGraph)


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_export_pandas(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    df = c.condensed_tree_.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (c._condensed_tree.parent.size, 5)
    df = c.leaf_tree_.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (c._leaf_tree.parent.size, 5)
    df = c.persistence_trace_.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (c._persistence_trace.min_size.size, 2)


def test_export_numpy(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    arr = c.condensed_tree_.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (c._condensed_tree.parent.size,)
    arr = c.leaf_tree_.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (c._leaf_tree.parent.size,)
    arr = c.persistence_trace_.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (c._persistence_trace.min_size.size,)
    arr = c.single_linkage_tree_
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (c._linkage_tree.parent.size, 4)
    arr = c.minimum_spanning_tree_
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (c._minimum_spanning_tree.parent.size, 3)


def test_bad_attrs():
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        c.min_cluster_size_cut(6.0)
    with pytest.raises(NotFittedError):
        c.distance_cut(0.5)
    with pytest.raises(NotFittedError):
        c.cluster_layers()
    with pytest.raises(NotFittedError):
        c.leaf_tree_
    with pytest.raises(NotFittedError):
        c.condensed_tree_
    with pytest.raises(NotFittedError):
        c.persistence_trace_
    with pytest.raises(NotFittedError):
        c.single_linkage_tree_
    with pytest.raises(NotFittedError):
        c.minimum_spanning_tree_


# Methods


def test_cluster_layers(X, knn):
    c = PLSCAN(min_samples=7, metric="precomputed").fit(knn)
    layers = c.cluster_layers()
    assert isinstance(layers, list)
    assert len(layers) == 1
    for x, labels, probabilities in layers:
        assert isinstance(x, np.float32)
        valid_labels(labels, X)
        valid_probabilities(probabilities, X)


def test_cluster_layers_params(X, knn):
    c = PLSCAN(min_samples=7, metric="precomputed").fit(knn)
    layers = c.cluster_layers(
        max_peaks=2, min_size=4.0, max_size=10.0, height=0.1, threshold=0.05
    )
    assert isinstance(layers, list)
    assert len(layers) == 1
    for x, labels, probabilities in layers:
        assert isinstance(x, np.float32)
        valid_labels(labels, X)
        valid_probabilities(probabilities, X)


def test_distance_cut(X, knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    labels, probs = c.distance_cut(0.5)
    valid_labels(labels, X)
    valid_probabilities(probs, X)


def test_min_cluster_size_cut(X, knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    labels, probs = c.min_cluster_size_cut(7.0)
    valid_labels(labels, X)
    valid_probabilities(probs, X)


# Sklearn Estimator


def test_hdbscan_is_sklearn_estimator():
    check_estimator(PLSCAN())
