"""Soft cluster membership prediction for fitted PLSCAN estimators."""

import warnings
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.exceptions import DataConversionWarning
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .sklearn import PLSCAN
from ._helpers import resolve_metric, resolve_metric_kws, to_scipy_csr


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
    distance_weights = _compute_exemplar_distance_weights(
        clusterer, exemplars_per_cluster
    )

    # Compute distances at which points connect to clusters
    outlier_weights, probability_in_some_cluster = _compute_topological_weights(
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

    core_dists, best_neighbor = _query_approximate_neighbors(clusterer, X)

    # 1. Distance-to-exemplar term.
    exemplars_per_cluster = clusterer.compute_exemplar_indices(labels)
    distance_weights = _compute_distance_weights(
        X,
        core_dists,
        exemplars_per_cluster,
        resolve_metric(clusterer.metric),
        resolve_metric_kws(clusterer._X, clusterer.metric, clusterer.metric_kws),
        exemplar_core_dists=clusterer.core_distances_,
        exemplar_X=clusterer._X,
    )

    # 2. Topology term from nearest fitted neighbors and their core distances.
    outlier_weights, probability_in_some_cluster = _compute_topological_weights(
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

    core_dists, best_neighbor = _query_approximate_neighbors(clusterer, X)
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


def _build_point_arrays(condensed_tree, n_points: int) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.int64]],
    np.ndarray[tuple[int], np.dtype[np.float32]],
]:
    """Maps condensed tree child to leaf-tree index and distance"""
    mask = condensed_tree.child < n_points
    child = condensed_tree.child[mask]
    child_to_leaf_tree = np.empty(n_points, dtype=np.int64)
    child_to_dist = np.empty(n_points, dtype=np.float32)
    child_to_leaf_tree[child] = condensed_tree.parent[mask] - n_points
    child_to_dist[child] = condensed_tree.distance[mask]
    return child_to_leaf_tree, child_to_dist


def _compute_exemplar_distance_weights(
    clusterer: PLSCAN,
    exemplars_per_cluster: list[np.ndarray],
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """Dispatch to sparse or dense exemplar distance weights."""
    if clusterer._X is None:
        return _extract_sparse_distance_weights(
            clusterer._mutual_graph, exemplars_per_cluster
        )
    return _compute_distance_weights(
        clusterer._X,
        clusterer.core_distances_,
        exemplars_per_cluster,
        resolve_metric(clusterer.metric),
        resolve_metric_kws(clusterer._X, clusterer.metric, clusterer.metric_kws),
    )


def _compute_distance_weights(
    X,
    core_dists,
    exemplars_per_cluster,
    metric,
    metric_kws,
    exemplar_core_dists=None,
    exemplar_X=None,
):
    """Compute distance-based weights towards each cluster."""
    if exemplar_core_dists is None:
        exemplar_core_dists = core_dists
    if exemplar_X is None:
        exemplar_X = X

    # Compute distances to closest exemplars
    D = np.empty((X.shape[0], len(exemplars_per_cluster)), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataConversionWarning)
        for c, exc in enumerate(exemplars_per_cluster):
            D[:, c] = np.maximum(
                np.maximum(
                    pairwise_distances(X, exemplar_X[exc], metric=metric, **metric_kws),
                    exemplar_core_dists[exc],
                ),
                core_dists[:, None],
            ).min(axis=1)

    # Convert to densities
    D = np.exp(-D)
    return D / D.sum(axis=1, keepdims=True)


def _extract_sparse_distance_weights(mutual_graph, exemplars_per_cluster):
    """Extract per-cluster distance weights from a sparse mutual-reachability graph."""
    # Convert to scipy CSR for efficient row and column slicing
    g = to_scipy_csr(mutual_graph)
    n_points = g.shape[0]
    n_clusters = len(exemplars_per_cluster)
    D = np.full((n_points, n_clusters), np.inf, dtype=np.float32)

    # Flatten exemplar lists and create a mapping from exemplar to cluster.
    all_exemplars = np.asarray([e for es in exemplars_per_cluster for e in es])
    exemplar_cluster = np.repeat(
        range(n_clusters), [len(es) for es in exemplars_per_cluster]
    )

    # Fill in the distance matrix.
    def _scatter(D, g_sub):
        coo = g_sub.tocoo()
        np.minimum.at(D, (coo.col, exemplar_cluster[coo.row]), coo.data)

    _scatter(D, g[all_exemplars])
    _scatter(D, g.T.tocsr()[all_exemplars])
    D[all_exemplars, exemplar_cluster] = 0.0

    densities = np.exp(-D)
    totals = densities.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(densities / totals, nan=0.0)


def _compute_topological_weights(
    clusterer: PLSCAN,
    labels: np.ndarray[tuple[int], np.dtype[np.int64]],
    n_clusters: int,
    selected_clusters: np.ndarray[tuple[int], np.dtype[np.intp]] | None,
    *,
    best_neighbors: np.ndarray[tuple[int], np.dtype[np.intp]] | None = None,
    new_core_dists: np.ndarray[tuple[int], np.dtype[np.float32]] | None = None,
) -> tuple[
    np.ndarray[tuple[int, int], np.dtype[np.float32]],
    np.ndarray[tuple[int], np.dtype[np.float32]],
]:
    """Compute tree-topology-based outlier weights and cluster probabilities.

    Returns ``(outlier_weights, probability_in_some_cluster)``, both derived
    from the leaf-tree merge distances.
    """
    child_to_leaf_tree, child_to_dist = _build_point_arrays(
        clusterer._condensed_tree, clusterer._num_points
    )

    # Compute selected clusters if not provided
    if selected_clusters is None:
        selected_clusters = _derive_selected_clusters(
            child_to_leaf_tree, labels, n_clusters
        )

    # If approximate neighbors and core distances are provided, use them to update
    # the child-to-leaf-tree mapping and child-to-distance for the new points.
    if best_neighbors is not None and new_core_dists is not None:
        child_to_leaf_tree = child_to_leaf_tree[best_neighbors]
        child_to_dist = np.minimum(child_to_dist[best_neighbors], new_core_dists)

    d_merge = _compute_merge_distances(
        clusterer._leaf_tree, child_to_leaf_tree, child_to_dist, selected_clusters
    )
    outlier_weights = _all_points_outlier_memberships(
        clusterer._leaf_tree, d_merge, child_to_leaf_tree
    )
    probability_in_some_cluster = _all_points_prob_in_some_cluster(
        clusterer._leaf_tree, d_merge, child_to_dist, selected_clusters
    )
    return outlier_weights, probability_in_some_cluster


def _derive_selected_clusters(
    point_cluster: np.ndarray[tuple[int], np.dtype[np.int64]],
    labels: np.ndarray[tuple[int], np.dtype[np.int64]],
    n_clusters: int,
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    """Recover the leaf-tree segment index for each cluster label."""
    selected = np.empty(n_clusters, dtype=np.intp)
    for c in range(n_clusters):
        selected[c] = point_cluster[labels == c].min()
    return selected


def _compute_merge_distances(
    leaf_tree,
    child_to_leaf_tree: np.ndarray[tuple[int], np.dtype[np.int64]],
    child_to_dist: np.ndarray[tuple[int], np.dtype[np.float32]],
    selected_clusters: np.ndarray[tuple[int], np.dtype[np.int64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """Merge distance matrix, shape (n_points, n_clusters).

    Entry [i, j] is the LCA-split distance between point i's leaf segment and
    cluster j's lineage in the leaf tree. When point i's segment is already
    inside cluster j's subtree, the merge distance is the point's own drop-out
    distance (``child_to_dist[i]``).
    """
    # Allocate output
    n_points = len(child_to_leaf_tree)
    n_clusters = len(selected_clusters)
    n_segments = leaf_tree.parent.shape[0]
    result = np.empty((n_points, n_clusters), dtype=np.float32)

    # Build child lists for downward subtree detection (once, shared across clusters).
    children = [[] for _ in range(n_segments)]
    for s in range(1, n_segments):
        p = leaf_tree.parent[s]
        if p != 0:
            children[p].append(s)

    # Loop over the selected clusters
    for j, selected_cluster in enumerate(selected_clusters):

        # 1. Mark ancestors of selected_cluster.
        strict_ancestor = np.zeros(n_segments, dtype=bool)
        node = leaf_tree.parent[selected_cluster]
        while node != 0:
            strict_ancestor[node] = True
            node = leaf_tree.parent[node]

        # 2. Mark selected_cluster's subtree.
        in_subtree = np.zeros(n_segments, dtype=bool)
        stack = [selected_cluster]
        while stack:
            node = stack.pop()
            in_subtree[node] = True
            stack.extend(children[node])

        # 3. Vectorized upward walk for nodes outside selected_cluster's lineage.
        terminate = in_subtree | strict_ancestor
        terminate[0] = True  # phantom-root always stops the walk

        # Ignore non-pending nodes with current = 0
        pending = ~terminate
        current = np.where(pending, np.arange(n_segments, dtype=np.intp), 0)
        d_last = np.full(n_segments, np.inf, dtype=np.float32)

        while pending.any():
            d_last[pending] = leaf_tree.max_distance[current[pending]]
            current[pending] = leaf_tree.parent[current[pending]]
            pending &= ~terminate[current]  # mark done once the new current terminates

        # 4. Per-point merge distance.
        in_lineage = (
            in_subtree[child_to_leaf_tree] | strict_ancestor[child_to_leaf_tree]
        )
        result[:, j] = np.where(in_lineage, child_to_dist, d_last[child_to_leaf_tree])

    return result


def _all_points_outlier_memberships(
    leaf_tree,
    d_merge: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    child_to_leaf_tree: np.ndarray[tuple[int], np.dtype[np.int64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """Per-cluster outlier scores from merge distances, shape (n_points, n_clusters).

    For each point i and cluster c:

        outlier[i, c] = exp(-d_merge[i, c] / d_min(s_i))
    """
    point_min_dist = leaf_tree.min_distance[child_to_leaf_tree]
    result = np.exp(-d_merge / point_min_dist[:, np.newaxis])
    totals = result.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nan_to_num(result / totals)  # 0/0 → 0 for phantom-root points


def _all_points_prob_in_some_cluster(
    leaf_tree,
    d_merge: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    child_to_dist: np.ndarray[tuple[int], np.dtype[np.float32]],
    selected_clusters: np.ndarray[tuple[int], np.dtype[np.intp]],
) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
    """Probability of each point belonging to any cluster, shape (n_points,).

    Matches HDBSCAN's ``all_points_prob_in_some_cluster`` formula, translated
    from lambda to distance space:

        prob = min(min_dist(c*), point_dist) / d_merge(c*)

    where ``c*`` is the cluster with the smallest merge distance for each point.
    ``min_dist(c*)`` is the distance at which cluster ``c*`` first forms, and
    ``point_dist`` is the distance at which the point drops out. Noise points
    receive a non-zero probability reflecting how close they came to connecting
    inside some cluster.
    """
    j_best = d_merge.argmin(axis=1)
    d_merge_best = d_merge[np.arange(len(d_merge)), j_best]
    min_dist_best = leaf_tree.min_distance[selected_clusters[j_best]]
    normalizer = np.minimum(min_dist_best, child_to_dist)
    return np.clip(normalizer / d_merge_best, 0.0, 1.0)


def _query_approximate_neighbors(
    clusterer: PLSCAN,
    X: np.ndarray[tuple[int, int], np.dtype[np.float32]],
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float32]],
    np.ndarray[tuple[int], np.dtype[np.intp]],
]:
    """Return query core distances and nearest mutual-reachability neighbors."""
    n_points = X.shape[0]
    k = min(clusterer.min_samples, clusterer._num_points)
    dists, indices = clusterer._space_tree.query(X, k=k, return_distance=True)
    dists = dists.astype(np.float32, copy=False)

    # Approximate new-point core distance from the kth training neighbor.
    core_dists = dists[:, -1]
    core_neighbor = clusterer.core_distances_[indices]

    # Label by nearest mutual-reachability neighbor's cluster.
    mreach = np.maximum(np.maximum(dists, core_neighbor), core_dists[:, None])
    best_idx = mreach.argmin(axis=1)
    rows = np.arange(n_points)
    best_neighbor = indices[rows, best_idx]
    return core_dists, best_neighbor
