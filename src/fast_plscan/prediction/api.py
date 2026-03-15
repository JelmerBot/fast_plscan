"""Soft cluster membership prediction for fitted PLSCAN estimators."""

import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..sklearn import PLSCAN
from .._helpers import resolve_metric, resolve_metric_kws
from ._helpers import (
    compute_exemplar_distance_weights,
    compute_distance_weights,
    compute_topological_weights,
    query_approximate_neighbors,
)


def all_points_membership_vectors(
    clusterer: PLSCAN, labels: np.ndarray[tuple[int], np.dtype[np.int64]] | None = None
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """Compute soft cluster membership vectors for all points.

    Parameters
    ----------
    clusterer
        A fitted :py:class:`~fast_plscan.PLSCAN` instance. Must have been fitted
        with feature-vector or (sparse) precomputed distance inputs.
    labels
        An optional integer array of shape ``(n_samples,)`` with cluster labels
        for each point. When ``None`` (default), the fitted ``labels_`` are
        used.

    Returns
    -------
    membership_vectors
        Float32 array of shape ``(n_samples, n_clusters)``. Entry ``[i, c]`` is
        the soft membership of point ``i`` in cluster ``c``. Row sums equal
        ``probability_in_some_cluster``. When no clusters exist, returns shape
        ``(n_samples, 0)``.

    Raises
    ------
    NotFittedError
        If ``clusterer`` has not been fitted yet.
    ValueError
        If ``clusterer`` was fitted with a precomputed minimum spanning forest.
    ValueError
        If the shape of ``labels`` does not match the number of samples in
        ``clusterer``.
    """

    # Validate inputs
    check_is_fitted(clusterer, "_minimum_spanning_tree")
    if clusterer._X is None and clusterer._mutual_graph is None:
        raise ValueError(
            "all_points_membership_vectors is only available for feature-vector "
            "or (sparse) precomputed distance inputs."
        )

    # Default to fitted labels if not provided
    selected_clusters = None
    if labels is None:
        labels = clusterer.labels_
        selected_clusters = clusterer.selected_clusters_

    # Validate labels shape
    if len(labels) != clusterer._num_points:
        raise ValueError("labels must match the number of samples")

    # Short-circuit if no clusters
    n_clusters = max(0, labels.max() + 1)
    if n_clusters == 0:
        return np.zeros((clusterer._num_points, 0), dtype=np.float32)

    # Compute distances to exemplars and convert to weights
    exemplars_per_cluster = clusterer.compute_exemplar_indices(labels)
    distance_weights = compute_exemplar_distance_weights(
        clusterer, exemplars_per_cluster
    )

    # Compute distances at which points connect to clusters
    outlier_weights, probability_in_some_cluster = compute_topological_weights(
        clusterer, labels, n_clusters, selected_clusters
    )

    # Normalize the blend and scale by per-point cluster probability.
    weights = distance_weights * outlier_weights
    totals = weights.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = np.nan_to_num(weights / totals, 0.0)
    return normalized * probability_in_some_cluster[:, np.newaxis]


def membership_vectors(
    clusterer: PLSCAN,
    X: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    labels: np.ndarray[tuple[int], np.dtype[np.int64]] | None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """Approximate soft cluster membership vectors for unseen points.

    This adapts HDBSCAN-style prediction to PLSCAN in distance space. Each new
    point is attached through its nearest mutual-reachability training neighbor,
    then blended distance and topology weights are scaled by the probability of
    belonging to any selected cluster.

    Parameters
    ----------
    clusterer
        A fitted :py:class:`~fast_plscan.PLSCAN` estimator trained on
        feature-vector input.
    X
        New feature vectors with shape ``(n_samples_new, n_features)``.
    labels
        Optional cluster labels of shape ``(n_samples,)`` used to define the
        selected clusters and exemplars on the fitted data. When ``None``
        (default), fitted ``labels_`` are used.

    Returns
    -------
    membership_vectors
        Float32 array of shape ``(n_samples_new, n_clusters)``. Entry
        ``[i, c]`` is the approximate soft membership of unseen point ``i`` in
        cluster ``c``. Row sums are less than or equal to 1.

    Raises
    ------
    NotFittedError
        If ``clusterer`` has not been fitted.
    ValueError
        If ``clusterer`` was fitted with precomputed input, or if ``X`` has an
        invalid number of features.
    ValueError
        If the shape of ``labels`` does not match the fitted number of
        samples.
    """
    check_is_fitted(clusterer, "_minimum_spanning_tree")
    if clusterer._X is None or clusterer._space_tree is None:
        raise ValueError(
            "membership_vectors is only available for feature-vector input."
        )

    X = check_array(X, dtype=np.float32, ensure_2d=True)
    if X.shape[1] != clusterer._X.shape[1]:
        raise ValueError(
            "X must have the same number of features as the fitted data: "
            f"expected {clusterer._X.shape[1]}, got {X.shape[1]}."
        )

    selected_clusters = None
    if labels is None:
        labels = clusterer.labels_
        selected_clusters = clusterer.selected_clusters_

    elif len(labels) != clusterer._num_points:
        raise ValueError("labels must match the number of samples")

    n_clusters = int(labels.max()) + 1
    if n_clusters <= 0:
        return np.zeros((X.shape[0], 0), dtype=np.float32)

    core_dists, best_neighbor = query_approximate_neighbors(clusterer, X)

    # 1. Distance-to-exemplar term.
    exemplars_per_cluster = clusterer.compute_exemplar_indices(labels)
    distance_weights = compute_distance_weights(
        X,
        core_dists,
        exemplars_per_cluster,
        resolve_metric(clusterer.metric),
        resolve_metric_kws(clusterer._X, clusterer.metric, clusterer.metric_kws),
        exemplar_core_dists=clusterer.core_distances_,
        exemplar_X=clusterer._X,
    )

    # 2. Topology term from nearest fitted neighbors and their core distances.
    outlier_weights, probability_in_some_cluster = compute_topological_weights(
        clusterer,
        labels,
        n_clusters,
        selected_clusters,
        best_neighbors=best_neighbor,
        new_core_dists=core_dists,
    )

    # 3. Blend and normalize.
    weights = distance_weights * outlier_weights
    totals = weights.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = np.nan_to_num(weights / totals, nan=0.0)
    return normalized * probability_in_some_cluster[:, np.newaxis]


def approximate_predict(
    clusterer: PLSCAN,
    X: np.ndarray[tuple[int, int], np.dtype[np.float32]],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.int64]],
    np.ndarray[tuple[int], np.dtype[np.float32]],
]:
    """Approximate labels and membership probabilities for unseen points.

    This follows HDBSCAN*-style approximate prediction: each new point is
    connected to the fitted structure through its nearest mutual-reachability
    neighbor in the training set.

    Parameters
    ----------
    clusterer
        A fitted :py:class:`~fast_plscan.PLSCAN` estimator trained on
        feature-vector input.
    X
        New feature vectors with shape ``(n_samples_new, n_features)``.

    Returns
    -------
    labels
        Predicted cluster labels for each new point. Points that cannot be
        linked to a selected cluster are labeled ``-1``.
    probabilities
        Approximate membership probabilities in ``[0, 1]`` for each new point.

    Raises
    ------
    NotFittedError
        If ``clusterer`` has not been fitted.
    ValueError
        If ``clusterer`` was fitted with precomputed input, or if ``X`` has an
        invalid number of features.
    """
    check_is_fitted(clusterer, "_minimum_spanning_tree")
    if clusterer._X is None or clusterer._space_tree is None:
        raise ValueError(
            "approximate_predict is only available for feature-vector input."
        )

    X = check_array(X, dtype=np.float32, ensure_2d=True)
    if X.shape[1] != clusterer._X.shape[1]:
        raise ValueError(
            "X must have the same number of features as the fitted data: "
            f"expected {clusterer._X.shape[1]}, got {X.shape[1]}."
        )

    n_points = X.shape[0]
    labels = np.full(n_points, -1, dtype=np.int64)
    probabilities = np.zeros(n_points, dtype=np.float32)
    n_clusters = int(clusterer.labels_.max()) + 1
    if n_clusters <= 0:
        return labels, probabilities

    core_dists, best_neighbor = query_approximate_neighbors(clusterer, X)
    labels = clusterer.labels_[best_neighbor]

    # PLSCAN-style probability proxy in distance space.
    assigned = labels >= 0
    if np.any(assigned):
        assigned_clusters = labels[assigned]
        cluster_nodes = clusterer.selected_clusters_[assigned_clusters]
        min_d_cluster = clusterer._leaf_tree.min_distance[cluster_nodes]
        max_d_cluster = clusterer._leaf_tree.max_distance[cluster_nodes]
        leaf_persistence = max_d_cluster - min_d_cluster
        point_persistence = max_d_cluster - core_dists[assigned]
        with np.errstate(invalid="ignore", divide="ignore"):
            probabilities[assigned] = np.clip(
                np.nan_to_num(point_persistence / leaf_persistence, nan=0.0), 0.0, 1.0
            )

    return labels, probabilities
