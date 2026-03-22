"""API tests for post-fit helper parity."""

import numpy as np

from fast_plscan import (
    PLSCAN,
    compute_centroids_from_features,
    compute_exemplar_indices_from_trees,
    compute_medoid_indices_from_features,
    compute_medoid_indices_from_graph,
)
from ..checks import valid_medoid_indices, valid_centroids, valid_exemplar_indices


def test_compute_medoid_indices_from_features(X):
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


def test_compute_medoid_indices_from_graph(X, g_knn):
    c = PLSCAN(metric="precomputed").fit(g_knn)
    medoid_indices = compute_medoid_indices_from_graph(
        c._mutual_graph,
        c.probabilities_,
        c.labels_,
    )
    valid_medoid_indices(medoid_indices, X, c.labels_)
    np.testing.assert_array_equal(medoid_indices, c.compute_medoid_indices())


def test_compute_centroids_from_features(X):
    c = PLSCAN().fit(X)
    centroids = compute_centroids_from_features(c._X, c.probabilities_, c.labels_)
    valid_centroids(centroids, X, c.labels_)
    np.testing.assert_allclose(centroids, c.compute_centroids())


def test_compute_exemplar_indices_from_trees(X):
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


def test_compute_exemplar_indices_from_trees_all_noise(X):
    c = PLSCAN().fit(X)
    noise_labels = np.full(c._num_points, -1, dtype=np.int64)
    result = compute_exemplar_indices_from_trees(
        c._leaf_tree, c._condensed_tree, noise_labels, c._num_points
    )
    assert result == []
