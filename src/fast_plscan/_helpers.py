"Internal API of the plscan package."

import numpy as np
from scipy.sparse import csr_array

from ._api import (
    SpanningTree,
    LeafTree,
    apply_size_cut,
    PersistenceTrace,
)


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
    indices = indices.astype(np.int32, copy=False)
    distances = distances.astype(np.float32, copy=False)
    num_points, _ = distances.shape

    # Drop explicit missing neighbors and non-finite distances at input time.
    valid = (indices >= 0) & np.isfinite(distances)
    counts = valid.sum(axis=1, dtype=np.int32)
    indptr = np.empty(num_points + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    g = csr_array(
        (distances[valid], indices[valid], indptr),
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


def resolve_metric(metric: str) -> str:
    """
    Resolves metric aliases to names recognized by `pairwise-distances`.

    Parameters
    ----------
    metric
        The distance metric to resolve.

    Returns
    -------
    resolved_metric
        The resolved distance metric name.
    """
    return {"p": "minkowski", "infinity": "chebyshev"}.get(metric, metric)


def resolve_metric_kws(data, metric, metric_kws):
    """
    Resolves the metric keyword arguments based on the metric and data.

    Parameters
    ----------
    data
        The input data array.
    metric
        The distance metric for which to resolve keyword arguments.
    metric_kws
        The initial metric keyword arguments.

    Returns
    -------
    resolved_metric_kws
        A dictionary of resolved metric keyword arguments.
    """
    if metric_kws is None:
        metric_kws = dict()

    if metric == "seuclidean" and "V" not in metric_kws:
        metric_kws["V"] = np.var(data, axis=0)
    elif metric == "mahalanobis" and "VI" not in metric_kws:
        metric_kws["VI"] = np.linalg.inv(np.cov(data, rowvar=False))

    return metric_kws


def to_scipy_csr(graph) -> csr_array:
    """
    Converts _api.SparseGraph to a scipy sparse CSR matrix.
    """
    return csr_array(tuple(graph))
