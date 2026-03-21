"""API tests for input modes."""

import numpy as np
import pytest

from fast_plscan import (
    clusters_from_spanning_forest,
    compute_mutual_spanning_tree,
    extract_mutual_spanning_forest,
)
from ..checks import (
    valid_cluster_outputs,
    valid_core_distances,
    valid_mutual_graph,
    valid_neighbor_indices,
    valid_spanning_forest,
)


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
    lt_parent, lt_child, lt_child_count, lt_child_size = linkage_tree
    assert np.array_equal(lt_parent, linkage_tree.parent)
    assert np.array_equal(lt_child, linkage_tree.child)
    assert np.array_equal(lt_child_count, linkage_tree.child_count)
    assert np.array_equal(lt_child_size, linkage_tree.child_size)


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


def test_min_samples_exceeds_unsorted_graph_neighbors(g_knn):
    # g_knn has ~9 entries per row; min_samples=11 means C++ min_samples-1=10,
    # so every row triggers infinity core distances via fill_distances_unsorted.
    _, _, core_distances = extract_mutual_spanning_forest(
        g_knn, min_samples=11, is_sorted=False
    )
    assert np.all(np.isinf(core_distances))


def test_min_samples_exceeds_sorted_graph_neighbors(g_knn):
    # Same but exercises fill_distances_sorted (is_sorted=True path).
    _, _, core_distances = extract_mutual_spanning_forest(
        g_knn, min_samples=11, is_sorted=True
    )
    assert np.all(np.isinf(core_distances))
