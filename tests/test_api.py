"""Tests for the sklearn interface."""

import numpy as np
import pytest

from fast_plscan import (
    extract_mutual_spanning_forest,
    clusters_from_spanning_forest,
    compute_mutual_spanning_tree,
)
from .checks import *


@pytest.mark.parametrize("space_tree", ["auto", "kd_tree", "ball_tree"])
def test_one_component(X, space_tree):
    mst, neighbors, cd = compute_mutual_spanning_tree(X, space_tree=space_tree)
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
    mst, neighbors, cd = compute_mutual_spanning_tree(X)
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
