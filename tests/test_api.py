"""Tests for the sklearn interface."""

import numpy as np
import pytest

from fast_plscan import (
    PLSCAN,
    extract_mutual_spanning_forest,
    clusters_from_spanning_forest,
    compute_mutual_spanning_tree,
    compute_centroids_from_features,
    compute_exemplar_indices_from_trees,
    compute_medoid_indices_from_features,
    compute_medoid_indices_from_graph,
)
from fast_plscan._api import SpanningTree
from .checks import *


@pytest.mark.parametrize("space_tree", ["kd_tree", "ball_tree"])
def test_one_component_space_tree(X, space_tree):
    _, mst, neighbors, cd = compute_mutual_spanning_tree(X, space_tree=space_tree)
    (
        (labels, probabilities),
        selected_clusters,
        persistence_trace,
        leaf_tree,
        condensed_tree,
        linkage_tree,
    ) = clusters_from_spanning_forest(mst, X.shape[0])

    valid_spanning_forest(mst, X)
    valid_neighbor_indices(neighbors, X, 5)
    valid_core_distances(cd, X)
    valid_labels(labels, X)
    assert labels.max() == 2
    valid_probabilities(probabilities, X)
    valid_selected_clusters(selected_clusters, labels)
    valid_persistence_trace(persistence_trace)
    valid_leaf(leaf_tree)
    valid_condensed(condensed_tree, X)
    valid_linkage(linkage_tree, X)


@pytest.mark.parametrize(
    "persistence_measure",
    ["size", "distance", "density", "size-distance", "size-density"],
)
def test_one_component(X, persistence_measure):
    _, mst, neighbors, cd = compute_mutual_spanning_tree(X)
    (
        (labels, probabilities),
        selected_clusters,
        persistence_trace,
        leaf_tree,
        condensed_tree,
        linkage_tree,
    ) = clusters_from_spanning_forest(
        mst, X.shape[0], persistence_measure=persistence_measure
    )

    valid_spanning_forest(mst, X)
    valid_neighbor_indices(neighbors, X, 5)
    valid_core_distances(cd, X)
    valid_labels(labels, X)
    assert labels.max() == 2
    valid_probabilities(probabilities, X)
    valid_selected_clusters(selected_clusters, labels)
    valid_persistence_trace(persistence_trace)
    valid_leaf(leaf_tree)
    valid_condensed(condensed_tree, X)
    valid_linkage(linkage_tree, X)


def test_one_component_precomputed(X, g_dists):
    msf, mut_graph, cd = extract_mutual_spanning_forest(g_dists)
    (
        (labels, probabilities),
        selected_clusters,
        persistence_trace,
        leaf_tree,
        condensed_tree,
        linkage_tree,
    ) = clusters_from_spanning_forest(msf, X.shape[0])

    valid_spanning_forest(msf, X)
    valid_mutual_graph(mut_graph, X)
    valid_core_distances(cd, X)
    valid_labels(labels, X)
    assert labels.max() == 2
    valid_probabilities(probabilities, X)
    valid_selected_clusters(selected_clusters, labels)
    valid_persistence_trace(persistence_trace)
    valid_leaf(leaf_tree)
    valid_condensed(condensed_tree, X)
    valid_linkage(linkage_tree, X)


def test_compute_msf_partial_and_missing(X, g_knn):
    msf, mut_graph, cd = extract_mutual_spanning_forest(g_knn, is_sorted=True)
    (
        (labels, probabilities),
        selected_clusters,
        persistence_trace,
        leaf_tree,
        condensed_tree,
        linkage_tree,
    ) = clusters_from_spanning_forest(msf, X.shape[0])

    valid_spanning_forest(msf, X)
    valid_mutual_graph(mut_graph, X, missing=True)
    valid_core_distances(cd, X)
    valid_labels(labels, X)
    assert labels.max() == 3
    assert np.any(labels == -1)
    valid_probabilities(probabilities, X)
    valid_selected_clusters(selected_clusters, labels)
    valid_persistence_trace(persistence_trace)
    valid_leaf(leaf_tree)
    valid_condensed(condensed_tree, X)
    valid_linkage(linkage_tree, X)


@pytest.mark.parametrize(
    "space_tree,metric,metric_kws",
    [
        ("kd_tree", "euclidean", None),
        ("kd_tree", "manhattan", None),
        ("kd_tree", "minkowski", {"p": 2.5}),
        ("ball_tree", "canberra", None),
        ("ball_tree", "seuclidean", None),
    ],
)
def test_compute_mst_metrics(X, space_tree, metric, metric_kws):
    _, mst, neighbors, cd = compute_mutual_spanning_tree(
        X, space_tree=space_tree, metric=metric, metric_kws=metric_kws
    )
    valid_spanning_forest(mst, X)
    valid_neighbor_indices(neighbors, X, 5)
    valid_core_distances(cd, X)


def test_msf_non_default_min_samples(X, g_dists):
    msf, mut_graph, cd = extract_mutual_spanning_forest(g_dists, min_samples=3)
    valid_spanning_forest(msf, X)
    valid_mutual_graph(mut_graph, X)
    valid_core_distances(cd, X)


def test_clusters_from_forest_sample_weights(X):
    _, mst, _, _ = compute_mutual_spanning_tree(X)
    sample_weights = np.full(X.shape[0], 0.5, dtype=np.float32)
    (
        (labels, probabilities),
        selected_clusters,
        persistence_trace,
        leaf_tree,
        condensed_tree,
        linkage_tree,
    ) = clusters_from_spanning_forest(mst, X.shape[0], sample_weights=sample_weights)
    valid_labels(labels, X)
    valid_probabilities(probabilities, X)
    valid_selected_clusters(selected_clusters, labels)
    valid_persistence_trace(persistence_trace)
    valid_leaf(leaf_tree)
    valid_condensed(condensed_tree, X)
    valid_linkage(linkage_tree, X)


def test_clusters_from_forest_min_cluster_size(X):
    _, mst, _, _ = compute_mutual_spanning_tree(X)
    (
        (labels, probabilities),
        selected_clusters,
        persistence_trace,
        leaf_tree,
        condensed_tree,
        linkage_tree,
    ) = clusters_from_spanning_forest(mst, X.shape[0], min_cluster_size=15.0)
    valid_labels(labels, X)
    valid_probabilities(probabilities, X)
    valid_selected_clusters(selected_clusters, labels)
    valid_persistence_trace(persistence_trace)
    valid_leaf(leaf_tree)
    valid_condensed(condensed_tree, X)
    valid_linkage(linkage_tree, X)


def test_clusters_from_forest_max_cluster_size(X):
    _, mst, _, _ = compute_mutual_spanning_tree(X)
    (
        (labels, probabilities),
        selected_clusters,
        persistence_trace,
        leaf_tree,
        condensed_tree,
        linkage_tree,
    ) = clusters_from_spanning_forest(mst, X.shape[0], max_cluster_size=30.0)
    valid_labels(labels, X)
    valid_probabilities(probabilities, X)
    valid_selected_clusters(selected_clusters, labels)
    valid_persistence_trace(persistence_trace)
    valid_leaf(leaf_tree)
    valid_condensed(condensed_tree, X)
    valid_linkage(linkage_tree, X)


def test_clusters_from_empty_mst():
    empty = SpanningTree(
        np.array([], dtype=np.uint32),
        np.array([], dtype=np.uint32),
        np.array([], dtype=np.float32),
    )
    with pytest.raises(ValueError):
        clusters_from_spanning_forest(empty, 10)


def test_compute_medoid_indices_from_features(X):
    # API-regression check: this function should stay in lockstep with the
    # sklearn entry point. Behavioral correctness (edge cases, validation,
    # semantics) is tested in test_sklearn.py.
    c = PLSCAN().fit(X)
    medoid_indices = compute_medoid_indices_from_features(
        c._X,
        c.core_distances_,
        c.probabilities_,
        c.labels_,
        metric=c.metric,
        metric_kws=c.metric_kws,
    )
    valid_medoid_indices(medoid_indices, X, c.labels_)
    np.testing.assert_array_equal(medoid_indices, c.compute_medoid_indices())


def test_compute_centroids_from_features(X):
    # API-regression check: assert public functional API parity with sklearn.
    # Detailed centroid behavior assertions live in test_sklearn.py.
    c = PLSCAN().fit(X)
    centroids = compute_centroids_from_features(c._X, c.probabilities_, c.labels_)
    valid_centroids(centroids, X, c.labels_)
    np.testing.assert_allclose(centroids, c.compute_centroids())


def test_compute_exemplar_indices_from_trees(X):
    # API-regression check: ensure tree-based helper output matches sklearn's
    # compute_exemplar_indices. Core behavior tests remain in test_sklearn.py.
    c = PLSCAN().fit(X)
    exemplars_per_cluster = compute_exemplar_indices_from_trees(
        c._leaf_tree,
        c._condensed_tree,
        c.labels_,
        c._num_points,
    )
    valid_exemplar_indices(exemplars_per_cluster, X, c.labels_)
    for e1, e2 in zip(exemplars_per_cluster, c.compute_exemplar_indices(), strict=True):
        np.testing.assert_array_equal(e1, e2)


def test_compute_medoid_indices_from_graph(X, g_knn):
    # API-regression check: sparse helper should match sklearn medoid results.
    # Sparse behavior/validation expectations are covered in test_sklearn.py.
    c = PLSCAN(metric="precomputed").fit(g_knn)
    medoid_indices = compute_medoid_indices_from_graph(
        c._mutual_graph,
        c.probabilities_,
        c.labels_,
    )
    valid_medoid_indices(medoid_indices, X, c.labels_)
    np.testing.assert_array_equal(medoid_indices, c.compute_medoid_indices())
