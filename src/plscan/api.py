import numpy as np
from scipy.sparse import csr_array
from ._threads import get_num_threads, set_num_threads
from ._leaf_tree import LeafTree, compute_leaf_tree, apply_size_cut, apply_distance_cut
from ._condense_tree import CondensedTree, compute_condensed_tree
from ._linkage_tree import LinkageTree, compute_linkage_tree
from ._labelling import Labelling, compute_cluster_labels
from ._spanning_tree import SpanningTree, compute_spanning_forest
from ._sparse_graph import (
    SparseGraph,
    extract_core_distances,
    compute_mutual_reachability,
)
from ._persistence_trace import (
    PersistenceTrace,
    compute_size_persistence,
    compute_bi_persistence,
    compute_stability_icicles,
)


def compute_mutual_spanning_forest(
    graph: csr_array, *, min_samples: int = 5, is_sorted: bool = False
):
    """
    Computes a mutual spanning forest from a sparse CSR distance matrix.

    Parameters
    ----------
    X : csr_array
        A sparse (square) distance matrix in CSR format. Each point must have at
        least `min_samples` neighbors. The function is most efficient when the
        matrix is explicitly symmetric.
    min_samples : int, optional
        Core distances are the distance to the `min_samples`-th nearest
        neighbor. Default is 5.
    is_sorted : bool, optional
        If True, the input graph rows are assumed to be sorted by distance.

    Returns
    -------
    spanning_forest : plscan._spanning_Tree.SpanningTree
        A spanning forest of the input sparse distance matrix. If the input data
        forms a single connected component, the spanning forest is a minimum
        spanning tree. Otherwise, it is a collection of minimum spanning trees,
        one for each connected component.
    """
    graph = SparseGraph(graph.data, graph.indices, graph.indptr)
    core_distances = extract_core_distances(
        graph, min_samples=min_samples, is_sorted=is_sorted
    )
    graph = compute_mutual_reachability(graph, core_distances)
    spanning_tree = compute_spanning_forest(graph)
    order = np.argsort(spanning_tree.distance)
    spanning_tree = SpanningTree(
        parent=spanning_tree.parent[order],
        child=spanning_tree.child[order],
        distance=spanning_tree.distance[order],
    )
    return spanning_tree, core_distances


def clusters_from_spanning_forest(
    sorted_mst: SpanningTree,
    num_points: int,
    *,
    min_cluster_size: float = 2.0,
    max_cluster_size: float = np.inf,
    use_bi_persistence: bool = False,
    sample_weights: np.ndarray[tuple[int], np.dtype[np.float32]] | None = None,
):
    """
    Compute PLSCAN clusters from a sorted minimum spanning forest.

    Parameters
    ----------
    sorted_mst : SpanningTuple
        A sorted (partial) minimum spanning forest.
    num_points : int
        The number of points in the sorted minimum spanning forest.
    min_cluster_size : float, optional
        The minimum size of a cluster, by default 2.0.
    max_cluster_size : float, optional
        The maximum size of a cluster, by default np.inf.
    use_bi_persistence : bool, optional
        Whether to use total bi-persistence or total size-persistence for
        selecting the optimal minimum cluster size. Default is False.
    sample_weights : np.ndarray[tuple[int], np.dtype[np.float32]], optional
        Sample weights for the points in the sorted minimum spanning tree. If
        None, all samples are considered equally weighted. Default is None.

    Returns
    -------
    labels : plscan._labelling.Labelling
        Essentially a tuple of cluster labels and membership probabilities for
        each point.
    trace : plscan._persistence_trace.PersistenceTrace
        A trace of the total (bi-)persistence per minimum cluster size.
    leaf_tree : plscan._leaf_tree.LeafTree
        A leaf tree with cluster-leaves at minimum cluster sizes.
    condensed_tree : plscan._condense_tree.CondensedTree
        A condensed tree with the cluster merge distances.
    linkage_tree : plscan._linkage_tree.LinkageTree
        A single linkage dendrogram of the sorted minimum spanning tree. (order
        matches the input sorted_mst!)
    """
    linkage_tree = compute_linkage_tree(
        sorted_mst, num_points, sample_weights=sample_weights
    )
    condensed_tree = compute_condensed_tree(
        linkage_tree,
        sorted_mst,
        num_points,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        sample_weights=sample_weights,
    )
    leaf_tree = compute_leaf_tree(
        condensed_tree, num_points, min_cluster_size=min_cluster_size
    )
    if use_bi_persistence:
        trace = compute_bi_persistence(leaf_tree, condensed_tree, num_points)
    else:
        trace = compute_size_persistence(leaf_tree)
    selected_clusters = most_persistent_clusters(leaf_tree, trace)
    labels = compute_cluster_labels(
        leaf_tree, condensed_tree, selected_clusters, num_points
    )
    return labels, selected_clusters, trace, leaf_tree, condensed_tree, linkage_tree


def most_persistent_clusters(leaf_tree: LeafTree, trace: PersistenceTrace):
    """
    Selects the most persistent clusters based on the total persistence trace.

    Parameters
    ----------
    leaf_tree : LeafTuple
        The input leaf tree.
    trace : PersistenceTrace
        The total persistence trace.

    Returns
    -------
    selected_clusters : np.ndarray[tuple[int], np.dtype[np.int64]]
        The condensed tree parent IDS for the most persistent leaf-clusters.
    """
    best_birth = trace.min_size[np.argmax(trace.persistence)]
    return apply_size_cut(leaf_tree, best_birth)


def knn_to_csr(
    distances: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    indices: np.ndarray[tuple[int, int], np.dtype[np.int64]],
):
    """
    Converts k-nearest neighbor distances and indices into a CSR matrix.

    Parameters
    ----------
    distances : np.ndarray[tuple[int, int], np.dtype[np.float32]]
        A 2D array of distances between points. First column is ignored and
        should contain self-loop zeros.
    indices : np.ndarray[tuple[int, int], np.dtype[np.int64]]
        A 2D array of indices corresponding to the nearest neighbors. The first
        column is ignored and should contain self-loop indices.

    Returns
    -------
    graph : scipy.sparse.csr_array
        A sparse distance matrix in CSR format.
    """
    indices = indices[:, 1:].astype(np.int32, order="C", copy=False)
    distances = distances[:, 1:].astype(np.float32, order="C", copy=False)
    num_points, num_neighbors = distances.shape
    indptr = np.arange(num_points + 1, dtype=np.int32) * num_neighbors
    return csr_array(
        (distances.reshape(-1), indices.reshape(-1), indptr),
        shape=(num_points, num_points),
    )


def distance_matrix_to_csr(
    distances: np.ndarray[tuple[int, int], np.dtype[np.float32]],
):
    """
    Converts a dense 2D distance matrix into a CSR matrix.

    Parameters
    ----------
    distances : np.ndarray[tuple[int, int], np.dtype[np.float32]]
        A 2D array representing the distance matrix.

    Returns
    -------
    graph : scipy.sparse.csr_array
        A sparse distance matrix in CSR format.
    """
    num_points, num_neighbors = distances.shape
    distances = distances.reshape(-1)
    indices = np.tile(np.arange(num_points, dtype=np.int32), num_points)
    indptr = np.arange(num_points + 1, dtype=np.int32) * num_neighbors
    g = csr_array((distances, indices, indptr), shape=(num_points, num_points))
    g.eliminate_zeros()
    return g


def remove_self_loops(graph: csr_array):
    """
    Removes self-loops from a sparse CSR matrix in place.

    Parameters
    ----------
    graph : csr_array
        A sparse matrix in CSR format.

    Returns
    -------
    graph : csr_array
        The input sparse graph with self-loops removed.
    """
    # Remove self-loops
    diag = graph.diagonal().nonzero()
    graph[diag, diag] = 0.0
    graph.eliminate_zeros()
    return graph
