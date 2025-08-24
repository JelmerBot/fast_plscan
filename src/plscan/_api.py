"Internal API of the plscan package."

from typing import Any

import numpy as np
from scipy.sparse import csr_array
from sklearn.neighbors._ball_tree import BallTree32
from sklearn.neighbors._kd_tree import KDTree32

from ._condensed_tree import CondensedTree, compute_condensed_tree
from ._distances import get_dist
from ._labelling import Labelling, compute_cluster_labels
from ._leaf_tree import LeafTree, compute_leaf_tree, apply_size_cut, apply_distance_cut
from ._linkage_tree import LinkageTree, compute_linkage_tree
from ._persistence_trace import (
    PersistenceTrace,
    compute_size_persistence,
    compute_bi_persistence,
    compute_stability_icicles,
)
from ._space_tree import NodeData, SpaceTree, kdtree_query, balltree_query
from ._spanning_tree import (
    SpanningTree,
    extract_spanning_forest,
    compute_spanning_tree_kdtree,
    compute_spanning_tree_balltree,
)
from ._sparse_graph import (
    SparseGraph,
    extract_core_distances,
    compute_mutual_reachability,
)
from ._threads import get_max_threads, set_num_threads


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
    metric_kws = metric_kws or dict()
    if metric == "seuclidean" and "V" not in metric_kws:
        metric_kws["V"] = np.var(data, axis=0)
    elif metric == "mahalanobis" and "VI" not in metric_kws:
        metric_kws["VI"] = np.linalg.inv(np.cov(data, rowvar=False))

    if space_tree == "kd_tree":
        tree = KDTree32(data, metric=metric, **metric_kws)
        query_fun = kdtree_query
        spanning_tree_fun = compute_spanning_tree_kdtree
    else:
        tree = BallTree32(data, metric=metric, **metric_kws)
        query_fun = balltree_query
        spanning_tree_fun = compute_spanning_tree_balltree

    data, idx_array, node_data, node_bounds = tree.get_arrays()
    cpp_tree = SpaceTree(data, idx_array, node_data.view(np.float64), node_bounds)

    # This knn contains explicit self-loops (first edge on each row), increment
    # min_samples to correct for that!
    knn = query_fun(cpp_tree, min_samples + 2, metric, metric_kws)
    core_distances = extract_core_distances(knn, min_samples + 1, is_sorted=True)
    spanning_tree = spanning_tree_fun(cpp_tree, knn, core_distances, metric, metric_kws)

    return (
        sort_spanning_tree(spanning_tree),
        knn.indices.reshape(data.shape[0], min_samples + 2),
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
    X
        A sparse (square) distance matrix in CSR format. Each point must have at
        least `min_samples` neighbors. The function is most efficient when the
        matrix is explicitly symmetric.
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
        graph, min_samples=min_samples, is_sorted=is_sorted
    )
    graph = compute_mutual_reachability(graph, core_distances)
    spanning_tree = extract_spanning_forest(graph)
    return sort_spanning_tree(spanning_tree), graph, core_distances


def sort_spanning_tree(spanning_tree: SpanningTree) -> SpanningTree:
    """
    Sorts the edges of a spanning tree by their distance.

    Parameters
    ----------
    spanning_tree
        The spanning tree to sort.

    Returns
    -------
    sorted_mst
        A new spanning tree with sorted edges.
    """
    order = np.argsort(spanning_tree.distance)
    return SpanningTree(
        parent=spanning_tree.parent[order],
        child=spanning_tree.child[order],
        distance=spanning_tree.distance[order],
    )


def clusters_from_spanning_forest(
        sorted_mst: SpanningTree,
        num_points: int,
        *,
        min_cluster_size: float = 2.0,
        max_cluster_size: float = np.inf,
        use_bi_persistence: bool = False,
        sample_weights: np.ndarray[tuple[int], np.dtype[np.float32]] | None = None,
) -> tuple[
    Labelling, np.ndarray[tuple[int], np.dtype[np.uint32]], PersistenceTrace, LeafTree, CondensedTree, LinkageTree]:
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
    use_bi_persistence
        Whether to use total bi-persistence or total size-persistence for
        selecting the optimal minimum cluster size.
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
    if use_bi_persistence:
        trace = compute_bi_persistence(leaf_tree, condensed_tree, num_points)
    else:
        trace = compute_size_persistence(leaf_tree)
    selected_clusters = most_persistent_clusters(
        leaf_tree, trace, max_cluster_size=max_cluster_size
    )
    labels = compute_cluster_labels(
        leaf_tree, condensed_tree, selected_clusters, num_points
    )
    return labels, selected_clusters, trace, leaf_tree, condensed_tree, linkage_tree


def most_persistent_clusters(
        leaf_tree: LeafTree, trace: PersistenceTrace, max_cluster_size: float = np.inf
) -> np.ndarray[tuple[int], np.dtype[np.uint32]]:
    """
    Selects the most persistent clusters based on the total persistence trace.

    Parameters
    ----------
    leaf_tree
        The input leaf tree.
    trace
        The total persistence trace.

    Returns
    -------
    selected_clusters
        The condensed tree parent IDS for the most persistent leaf-clusters.
    """
    idx = np.searchsorted(trace.min_size, max_cluster_size, side="right")
    persistences = trace.persistence[:idx]
    if persistences.size == 0:
        return np.array([], dtype=np.uint32)
    best_birth = trace.min_size[np.argmax(persistences)]
    return apply_size_cut(leaf_tree, best_birth)


def knn_to_csr(
        distances: np.ndarray[tuple[int, int], np.dtype[np.float32]],
        indices: np.ndarray[tuple[int, int], np.dtype[np.int64]],
) -> csr_array:
    """
    Converts k-nearest neighbor distances and indices into a CSR matrix.

    Parameters
    ----------
    distances
        A 2D array of distances between points. Self-loops are ignored if
        present.
    indices
        A 2D array of indices corresponding to the nearest neighbors. The first
        column is ignored and should contain self-loop indices.

    Returns
    -------
    graph
        A sparse distance matrix in CSR format.
    """
    indices = indices.astype(np.int32)
    distances = distances.astype(np.float32)
    num_points, num_neighbors = distances.shape
    indptr = np.arange(num_points + 1, dtype=np.int32) * num_neighbors
    g = csr_array(
        (distances.reshape(-1), indices.reshape(-1), indptr),
        shape=(num_points, num_points),
    )
    g.eliminate_zeros()
    return g


def distance_matrix_to_csr(
        distances: np.ndarray[tuple[int, int], np.dtype[np.float32]], copy: bool = True
) -> csr_array:
    """
    Converts a dense 2D distance matrix into a CSR matrix.

    Parameters
    ----------
    distances
        A 2D array representing the distance matrix.
    copy:
        A flag indicating whether to create a copy.

    Returns
    -------
    graph:
        A sparse distance matrix in CSR format.
    """
    num_points, num_neighbors = distances.shape
    distances = distances.astype(np.float32, order="C", copy=copy)
    np.fill_diagonal(distances, 0.0)
    distances = distances.reshape(-1)
    indices = np.tile(np.arange(num_points, dtype=np.int32), num_points)
    indptr = np.arange(num_points + 1, dtype=np.int32) * num_neighbors
    g = csr_array((distances, indices, indptr), shape=(num_points, num_points))
    g.eliminate_zeros()
    return g


def remove_self_loops(graph: csr_array) -> csr_array:
    """
    Removes self-loops from a sparse CSR matrix in place.

    Parameters
    ----------
    graph
        A sparse matrix in CSR format.

    Returns
    -------
    graph
        The input sparse graph with self-loops removed.
    """
    # Remove self-loops
    diag = graph.diagonal().nonzero()
    graph[diag, diag] = 0.0
    graph.eliminate_zeros()

    graph = csr_array(
        (
            graph.data.astype(np.float32),
            graph.indices.astype(np.int32),
            graph.indptr.astype(np.int32),
        ),
        shape=graph.shape,
    )
    return graph
