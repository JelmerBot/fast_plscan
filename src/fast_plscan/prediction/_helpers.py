"""Soft cluster membership prediction for fitted PLSCAN estimators."""

import warnings
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.exceptions import DataConversionWarning
from ..sklearn import PLSCAN
from .._helpers import resolve_metric, resolve_metric_kws, to_scipy_csr


def build_point_arrays(condensed_tree, n_points: int) -> tuple[
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


def compute_distance_weights(
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


def extract_sparse_distance_weights(mutual_graph, exemplars_per_cluster):
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


def compute_exemplar_distance_weights(
    clusterer: PLSCAN,
    exemplars_per_cluster: list[np.ndarray],
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """Dispatch to sparse or dense exemplar distance weights."""
    if clusterer._X is None:
        return extract_sparse_distance_weights(
            clusterer._mutual_graph, exemplars_per_cluster
        )
    return compute_distance_weights(
        clusterer._X,
        clusterer.core_distances_,
        exemplars_per_cluster,
        resolve_metric(clusterer.metric),
        resolve_metric_kws(clusterer._X, clusterer.metric, clusterer.metric_kws),
    )


def derive_selected_clusters(
    point_cluster: np.ndarray[tuple[int], np.dtype[np.int64]],
    labels: np.ndarray[tuple[int], np.dtype[np.int64]],
    n_clusters: int,
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    """Recover the leaf-tree segment index for each cluster label."""
    selected = np.empty(n_clusters, dtype=np.intp)
    for c in range(n_clusters):
        selected[c] = point_cluster[labels == c].min()
    return selected


def compute_merge_distances(
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


def all_points_outlier_memberships(
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


def all_points_prob_in_some_cluster(
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


def compute_topological_weights(
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
    child_to_leaf_tree, child_to_dist = build_point_arrays(
        clusterer._condensed_tree, clusterer._num_points
    )

    # Compute selected clusters if not provided
    if selected_clusters is None:
        selected_clusters = derive_selected_clusters(
            child_to_leaf_tree, labels, n_clusters
        )

    # If approximate neighbors and core distances are provided, use them to update
    # the child-to-leaf-tree mapping and child-to-distance for the new points.
    if best_neighbors is not None and new_core_dists is not None:
        child_to_leaf_tree = child_to_leaf_tree[best_neighbors]
        child_to_dist = np.minimum(child_to_dist[best_neighbors], new_core_dists)

    d_merge = compute_merge_distances(
        clusterer._leaf_tree, child_to_leaf_tree, child_to_dist, selected_clusters
    )
    outlier_weights = all_points_outlier_memberships(
        clusterer._leaf_tree, d_merge, child_to_leaf_tree
    )
    probability_in_some_cluster = all_points_prob_in_some_cluster(
        clusterer._leaf_tree, d_merge, child_to_dist, selected_clusters
    )
    return outlier_weights, probability_in_some_cluster


def query_approximate_neighbors(
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
