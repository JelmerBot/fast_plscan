"""API tests for parameter behavior."""

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


def test_compute_mst_invalid_space_tree(X):
    with pytest.raises(ValueError):
        compute_mutual_spanning_tree(X, space_tree="invalid")


def test_clusters_from_forest_invalid_persistence_measure(X):
    _, mst, _, _ = compute_mutual_spanning_tree(X)
    with pytest.raises(ValueError):
        clusters_from_spanning_forest(mst, X.shape[0], persistence_measure="invalid")
