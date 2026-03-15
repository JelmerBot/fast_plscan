"""API tests for input modes."""

import numpy as np
import pytest

from fast_plscan import (
    clusters_from_spanning_forest,
    compute_mutual_spanning_tree,
    extract_mutual_spanning_forest,
)
from fast_plscan._api import SpanningTree
from ..checks import *


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
    assert labels.max() == 2
    valid_cluster_outputs(
        labels,
        probabilities,
        selected_clusters,
        persistence_trace,
        leaf_tree,
        condensed_tree,
        linkage_tree,
        X,
    )


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
    assert labels.max() == 2
    valid_cluster_outputs(
        labels,
        probabilities,
        selected_clusters,
        persistence_trace,
        leaf_tree,
        condensed_tree,
        linkage_tree,
        X,
    )


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
    assert labels.max() == 2
    valid_cluster_outputs(
        labels,
        probabilities,
        selected_clusters,
        persistence_trace,
        leaf_tree,
        condensed_tree,
        linkage_tree,
        X,
    )


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
    valid_mutual_graph(mut_graph, X)
    valid_core_distances(cd, X)
    assert labels.max() == 3
    assert np.any(labels == -1)
    valid_cluster_outputs(
        labels,
        probabilities,
        selected_clusters,
        persistence_trace,
        leaf_tree,
        condensed_tree,
        linkage_tree,
        X,
    )

