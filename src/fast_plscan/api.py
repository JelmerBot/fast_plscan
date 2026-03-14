import numpy as np
from typing import Any, Callable
from scipy.sparse import csr_array
from sklearn.neighbors._ball_tree import BallTree32
from sklearn.neighbors._kd_tree import KDTree32
from sklearn.metrics import pairwise_distances

from ._helpers import (
    resolve_metric,
    resolve_metric_kws,
    sort_spanning_tree,
    most_persistent_clusters,
    to_scipy_csr,
)
from ._api import (
    get_dist,
    LeafTree,
    compute_leaf_tree,
    Labelling,
    compute_cluster_labels,
    LinkageTree,
    compute_linkage_tree,
    CondensedTree,
    compute_condensed_tree,
    SpaceTree,
    kdtree_query,
    balltree_query,
    PersistenceTrace,
    compute_size_persistence,
    compute_distance_persistence,
    compute_density_persistence,
    compute_size_distance_bi_persistence,
    compute_size_density_bi_persistence,
    SpanningTree,
    extract_spanning_forest,
    compute_spanning_tree_kdtree,
    compute_spanning_tree_balltree,
    SparseGraph,
    extract_core_distances,
    compute_mutual_reachability,
)


def compute_mutual_spanning_tree(
    data: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    *,
    min_samples: int = 5,
    space_tree: str = "kd_tree",
    metric: str = "euclidean",
    metric_kws: dict[str, Any] | None = None,
) -> tuple[
    SpanningTree,
    np.ndarray[tuple[int, int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.float32]],
]:
    """
    Computes a mutual reachability spanning tree from data features using a
    KDTree.

    Parameters
    ----------
    data
        High dimensional data features. Values must be finite and not missing.
    space_tree
        The type of spatial tree to use. Valid options are: "kd_tree",
        "ball_tree". See ``metric`` for an overview of supported metrics on each
        tree type.
    min_samples
        Core distances are the distance to the ``min_samples``-th nearest
        neighbor.
    metric
        The distance metric to use. See
        :py:attr:`~plscan.PLSCAN.VALID_KDTREE_METRICS` and
        :py:attr:`~plscan.PLSCAN.VALID_BALLTREE_METRICS` for lists of valid
        metrics. See sklearn documentation for metric definitions.
    metric_kws
        Additional keyword arguments for the distance metric.

    Returns
    -------
    spanning_tree
        A spanning tree of the input sparse distance matrix.
    indices
        A 2D array with knn indices.
    core_distances
        A 1D array with core distances.
    """
    metric_kws = resolve_metric_kws(data, metric, metric_kws)

    if space_tree == "kd_tree":
        tree = KDTree32(data, metric=metric, **metric_kws)
        query_fun = kdtree_query
        spanning_tree_fun = compute_spanning_tree_kdtree
    elif space_tree == "ball_tree":
        tree = BallTree32(data, leaf_size=5, metric=metric, **metric_kws)
        query_fun = balltree_query
        spanning_tree_fun = compute_spanning_tree_balltree
    else:
        raise ValueError(f"Invalid space tree type: {space_tree}.")

    data, idx_array, node_data, node_bounds = tree.get_arrays()
    cpp_tree = SpaceTree(data, idx_array, node_data.view(np.float64), node_bounds)

    # This knn contains explicit self-loops (first edge on each row), increment
    # min_samples to correct for that!
    knn = query_fun(cpp_tree, min_samples + 1, metric, metric_kws)
    core_distances = extract_core_distances(knn, min_samples, is_sorted=True)
    spanning_tree = spanning_tree_fun(cpp_tree, knn, core_distances, metric, metric_kws)

    return (
        sort_spanning_tree(spanning_tree),
        knn.indices.reshape(data.shape[0], min_samples + 1),
        core_distances,
    )


def extract_mutual_spanning_forest(
    graph: csr_array, *, min_samples: int = 5, is_sorted: bool = False
) -> tuple[
    SpanningTree,
    SparseGraph,
    np.ndarray[tuple[int], np.dtype[np.float32]],
]:
    """
    Computes a mutual spanning forest from a sparse CSR distance matrix.

    Parameters
    ----------
    graph
        A sparse (square) distance matrix in CSR format. Each point must have at
        least `min_samples` neighbors, with no explicit self-loops. The function
        is most efficient when the matrix is explicitly symmetric.
    min_samples
        Core distances are the distance to the `min_samples`-th nearest
        neighbor.
    is_sorted
        If True, the input graph rows are assumed to be sorted by distance.

    Returns
    -------
    spanning_forest
        A spanning forest of the input sparse distance matrix. If the input data
        forms a single connected component, the spanning forest is a minimum
        spanning tree. Otherwise, it is a collection of minimum spanning trees,
        one for each connected component.
    graph
        A copy of the input graph with edges weighted and sorted by mutual
        reachability.
    core_distances
        A 1D array with core distances.
    """
    graph = SparseGraph(graph.data, graph.indices, graph.indptr)
    core_distances = extract_core_distances(
        graph, min_samples=min_samples - 1, is_sorted=is_sorted
    )
    graph = compute_mutual_reachability(graph, core_distances)
    spanning_tree = extract_spanning_forest(graph)
    return sort_spanning_tree(spanning_tree), graph, core_distances


def clusters_from_spanning_forest(
    sorted_mst: SpanningTree,
    num_points: int,
    *,
    min_cluster_size: float = 2.0,
    max_cluster_size: float = np.inf,
    persistence_measure: str = "size",
    sample_weights: np.ndarray[tuple[int], np.dtype[np.float32]] | None = None,
) -> tuple[
    Labelling,
    np.ndarray[tuple[int], np.dtype[np.uint32]],
    PersistenceTrace,
    LeafTree,
    CondensedTree,
    LinkageTree,
]:
    """
    Compute PLSCAN clusters from a sorted minimum spanning forest.

    Parameters
    ----------
    sorted_mst
        A sorted (partial) minimum spanning forest.
    num_points
        The number of points in the sorted minimum spanning forest.
    min_cluster_size
        The minimum size of a cluster.
    max_cluster_size
        The maximum size of a cluster.
    persistence_measure
        Selects a persistence measure. Valid options are "size", "distance",
        "density", "size-distance", and "size-density". The "size",
        "distance", and "density" options compute persistence as the range
        of size/distance/density values for which clusters are leaves. The
        "size-distance" and "size-density" options compute bi-persistence as
        the distance/density -- minimum cluster size areas for which clusters
        are leaves. Density is computed as exp(-dist).
    sample_weights
        Sample weights for the points in the sorted minimum spanning tree. If
        None, all samples are considered equally weighted.

    Returns
    -------
    labels
        Essentially a tuple of cluster labels and membership probabilities for
        each point.
    trace
        A trace of the total (bi-)persistence per minimum cluster size.
    leaf_tree
        A leaf tree with cluster-leaves at minimum cluster sizes.
    condensed_tree
        A condensed tree with the cluster merge distances.
    linkage_tree
        A single linkage dendrogram of the sorted minimum spanning tree. (order
        matches the input sorted_mst!)
    """
    if sorted_mst.parent.shape[0] == 0:
        raise ValueError("Input minimum spanning tree is empty.")

    linkage_tree = compute_linkage_tree(
        sorted_mst, num_points, sample_weights=sample_weights
    )
    condensed_tree = compute_condensed_tree(
        linkage_tree,
        sorted_mst,
        num_points,
        min_cluster_size=min_cluster_size,
        sample_weights=sample_weights,
    )
    if condensed_tree.cluster_rows.size == 0:
        leaf_tree = LeafTree(
            np.array([0], dtype=np.uint32),
            np.array([sorted_mst.distance[0]], dtype=np.float32),
            np.array([sorted_mst.distance[-1]], dtype=np.float32),
            np.array([min_cluster_size], dtype=np.float32),
            np.array([num_points], dtype=np.float32),
        )
    else:
        leaf_tree = compute_leaf_tree(
            condensed_tree, num_points, min_cluster_size=min_cluster_size
        )
    if persistence_measure == "size":
        trace = compute_size_persistence(leaf_tree)
    elif persistence_measure == "distance":
        trace = compute_distance_persistence(leaf_tree, condensed_tree, num_points)
    elif persistence_measure == "density":
        trace = compute_density_persistence(leaf_tree, condensed_tree, num_points)
    elif persistence_measure == "size-distance":
        trace = compute_size_distance_bi_persistence(
            leaf_tree, condensed_tree, num_points
        )
    elif persistence_measure == "size-density":
        trace = compute_size_density_bi_persistence(
            leaf_tree, condensed_tree, num_points
        )
    else:
        raise ValueError(f"Invalid persistence measure: {persistence_measure}.")
    selected_clusters = most_persistent_clusters(
        leaf_tree, trace, max_cluster_size=max_cluster_size
    )
    labels = compute_cluster_labels(
        leaf_tree, condensed_tree, selected_clusters, num_points
    )
    return labels, selected_clusters, trace, leaf_tree, condensed_tree, linkage_tree


def get_distance_callback(
    metric: str,
    *,
    p: float | None = None,
    V: np.ndarray[tuple[int], np.dtype[np.float32]] | None = None,
    VI: np.ndarray[tuple[int, int], np.dtype[np.float32]] | None = None,
) -> Callable[
    [
        np.ndarray[tuple[int], np.dtype[np.float32]],
        np.ndarray[tuple[int], np.dtype[np.float32]],
    ],
    float,
]:
    """
    Returns a fast distance function callback for the specified metric.
    Potential keyword arguments are processed once here, and do not have to be
    repeated on the callback.

    Parameters
    ----------
    metric
        The distance metric to use. See
        :py:attr:`~plscan.PLSCAN.VALID_BALLTREE_METRICS` for a list of valid
        metrics. See sklearn documentation for metric definitions.
    p
        The order of the Minkowski distance. Required if `metric` is
        "minkowski".
    V
        The variance vector for the standardized Euclidean distance. Required if
        `metric` is "seuclidean".
    VI
        The inverse covariance matrix for the Mahalanobis distance. Required if
        `metric` is "mahalanobis".

    Returns
    -------
    dist
        The distance function callback. Inputs must be 1D c-contiguous numpy
        arrays of float32.
    """
    return get_dist(metric, p=p, V=V, VI=VI)


def compute_centroids_from_features(
    data: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    probabilities: np.ndarray[tuple[int], np.dtype[np.float32]],
    labels: np.ndarray[tuple[int], np.dtype[np.int64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """Compute probability-weighted centroids from feature vectors.

    Assumes inputs are already validated.

    Parameters
    ----------
    data
        Feature matrix of shape ``(n_samples, n_features)``.
    probabilities
        Cluster membership probabilities for all points, shape ``(n_samples,)``.
    labels
        Cluster labels with shape ``(n_samples,)``.

    Returns
    -------
    centroids
        Float32 array of shape ``(n_clusters, n_features)``.
    """
    n_clusters = int(labels.max()) + 1
    mask = labels >= 0
    W = csr_array(
        (probabilities[mask], (labels[mask], np.flatnonzero(mask))),
        shape=(n_clusters, data.shape[0]),
    )
    denom = np.asarray(W.sum(axis=1)).ravel()
    return (W @ data) / denom[:, np.newaxis]


def compute_exemplar_indices_from_trees(
    leaf_tree: LeafTree,
    condensed_tree: CondensedTree,
    labels: np.ndarray[tuple[int], np.dtype[np.int64]],
    num_points: int,
) -> list[np.ndarray[tuple[int], np.dtype[np.intp]]]:
    """Compute exemplar point indices per cluster from fitted tree structures.

    Assumes inputs are already validated.

    Parameters
    ----------
    leaf_tree
        Fitted leaf tree.
    condensed_tree
        Fitted condensed tree.
    labels
        Dense cluster labels with shape ``(n_samples,)``.
    num_points
        Number of points in the original dataset.

    Returns
    -------
    exemplars_per_cluster
        List of arrays with exemplar point indices per cluster.
    """
    n_clusters = int(labels.max()) + 1
    if n_clusters <= 0:
        return []

    non_leaves = np.unique(
        condensed_tree.parent[condensed_tree.cluster_rows] - num_points
    )

    mask = condensed_tree.child < num_points
    points = condensed_tree.child[mask]
    segments = condensed_tree.parent[mask] - num_points
    dists = condensed_tree.distance[mask]
    min_dists = leaf_tree.min_distance

    valid_points = np.flatnonzero(
        ~np.isin(segments, non_leaves) & (labels[points] >= 0)
    )
    is_exemplar = dists[valid_points] == min_dists[segments[valid_points]]
    exemplar_pts = points[valid_points[is_exemplar]]
    exemplar_clusters = labels[exemplar_pts]
    return [exemplar_pts[exemplar_clusters == c] for c in range(n_clusters)]


def compute_medoid_indices_from_features(
    data: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    core_distances: np.ndarray[tuple[int], np.dtype[np.float32]],
    probabilities: np.ndarray[tuple[int], np.dtype[np.float32]],
    labels: np.ndarray[tuple[int], np.dtype[np.int64]],
    *,
    metric: str = "euclidean",
    metric_kws: dict[str, Any] | None = None,
) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
    """Compute cluster medoid indices from feature vectors.

    Assumes inputs are already validated. Uses probability-weighted mutual
    reachability distances within each cluster.

    Parameters
    ----------
    data
        Feature matrix of shape ``(n_samples, n_features)``.
    core_distances
        Core distances for all points, shape ``(n_samples,)``.
    probabilities
        Cluster membership probabilities for all points, shape ``(n_samples,)``.
    labels
        Dense cluster labels with shape ``(n_samples,)``.
    metric
        Pairwise distance metric.
    metric_kws
        Optional metric keyword arguments.

    Returns
    -------
    medoid_indices
        Integer array of shape ``(n_clusters,)`` with indices into ``data``.
    """
    n_clusters = int(labels.max()) + 1
    medoid_indices = np.empty(n_clusters, dtype=np.intp)
    metric = resolve_metric(metric)
    metric_kws = resolve_metric_kws(data, metric, metric_kws)

    for c in range(n_clusters):
        cluster_pts = np.flatnonzero(labels == c)
        dists = pairwise_distances(data[cluster_pts], metric=metric, **metric_kws)
        core_c = core_distances[cluster_pts]
        dists = np.maximum(dists, np.maximum.outer(core_c, core_c))
        np.fill_diagonal(dists, 0.0)
        dists = dists * probabilities[cluster_pts]
        medoid_indices[c] = cluster_pts[np.argmin(dists.sum(axis=1))]

    return medoid_indices


def compute_medoid_indices_from_graph(
    graph: SparseGraph | csr_array,
    probabilities: np.ndarray[tuple[int], np.dtype[np.float32]],
    labels: np.ndarray[tuple[int], np.dtype[np.int64]],
) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
    """Compute cluster medoid indices from a sparse mutual-reachability graph.

    Assumes inputs are already validated. Within each cluster, computes the
    probability-weighted mean mutual reachability distance over reachable
    neighbours.

    Parameters
    ----------
    graph
        Sparse mutual-reachability graph as ``SparseGraph`` or ``csr_array``.
    probabilities
        Cluster membership probabilities for all points, shape ``(n_samples,)``.
    labels
        Dense cluster labels with shape ``(n_samples,)``.

    Returns
    -------
    medoid_indices
        Integer array of shape ``(n_clusters,)`` with indices into the original
        point ordering.
    """
    n_clusters = int(labels.max()) + 1
    medoid_indices = np.empty(n_clusters, dtype=np.intp)
    g = graph if isinstance(graph, csr_array) else to_scipy_csr(graph)

    for c in range(n_clusters):
        cluster_pts = np.flatnonzero(labels == c)
        probs_c = probabilities[cluster_pts]
        g_sub = g[cluster_pts][:, cluster_pts]
        g_sym = g_sub.maximum(g_sub.T)
        scores = np.asarray(g_sym.dot(probs_c)).ravel()
        g_indicator = g_sym.copy()
        g_indicator.data[:] = 1.0
        denom = np.asarray(g_indicator.dot(probs_c)).ravel()
        with np.errstate(invalid="ignore", divide="ignore"):
            avg_scores = np.where(denom > 0, scores / denom, np.inf)
        medoid_indices[c] = cluster_pts[np.argmin(avg_scores)]

    return medoid_indices
