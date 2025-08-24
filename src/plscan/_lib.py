"Internal API of the plscan package."

import numpy as np
from scipy.sparse import csr_array

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
