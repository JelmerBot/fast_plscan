import numpy as np

from fast_plscan._api import (
    SparseGraph,
    SpanningTree,
    LinkageTree,
    CondensedTree,
    LeafTree,
    PersistenceTrace,
)


def valid_spanning_forest(msf, X):
    assert isinstance(msf, SpanningTree)
    assert np.all(np.diff(msf.distance) >= 0.0)
    assert np.all(msf.child >= 0)
    assert np.all(msf.parent >= 0)
    assert msf.parent.size <= (X.shape[0] - 1)


def valid_neighbor_indices(indices, X, min_samples):
    assert isinstance(indices, np.ndarray)
    assert indices.shape[0] == X.shape[0]
    assert indices.shape[1] == min_samples + 1
    assert indices.dtype == np.int32
    assert np.all(indices >= 0) and np.all(indices < X.shape[0])


def valid_mutual_graph(mut_graph, X, *, missing=False):
    assert isinstance(mut_graph, SparseGraph)
    assert mut_graph.indptr.shape[0] == X.shape[0] + 1
    if not missing:
        assert np.all(mut_graph.indices >= 0)
    for start, end in zip(mut_graph.indptr[:-1], mut_graph.indptr[1:]):
        assert np.all(np.diff(mut_graph.data[start:end]) >= 0.0)


def valid_core_distances(cd, X):
    assert isinstance(cd, np.ndarray)
    assert np.all(np.isfinite(cd))
    assert cd.shape[0] == X.shape[0]


def valid_labels(labels, X):
    assert isinstance(labels, np.ndarray)
    assert labels.shape[0] == X.shape[0]
    assert labels.dtype == np.int64
    assert np.all(labels >= -1)


def valid_probabilities(probabilities, X):
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape[0] == X.shape[0]
    assert probabilities.dtype == np.float32
    assert np.all(probabilities >= 0.0)
    assert np.all(np.isfinite(probabilities))


def valid_selected_clusters(selected_clusters, labels):
    assert isinstance(selected_clusters, np.ndarray)
    assert selected_clusters.dtype == np.uint32
    if np.all(labels == -1):
        assert selected_clusters.shape[0] == 0 or selected_clusters == np.array(
            [0], dtype=np.uint32
        )
    else:
        assert selected_clusters.shape[0] == labels.max() + 1
    assert np.all(selected_clusters >= 0)


def valid_persistence_trace(persistence_trace):
    assert isinstance(persistence_trace, PersistenceTrace)
    assert isinstance(persistence_trace.min_size, np.ndarray)
    assert persistence_trace.min_size.dtype == np.float32
    assert np.all(persistence_trace.min_size >= 2.0)
    assert isinstance(persistence_trace.persistence, np.ndarray)
    assert persistence_trace.persistence.dtype == np.float32
    assert np.all(persistence_trace.persistence >= 0.0)


def valid_leaf(leaf_tree):
    assert isinstance(leaf_tree, LeafTree)
    assert isinstance(leaf_tree.parent, np.ndarray)
    assert leaf_tree.parent.dtype == np.uint32
    assert leaf_tree.parent.max() < leaf_tree.parent.size
    assert leaf_tree.parent[0] == 0
    assert isinstance(leaf_tree.min_distance, np.ndarray)
    assert leaf_tree.min_distance.dtype == np.float32
    assert isinstance(leaf_tree.max_distance, np.ndarray)
    assert leaf_tree.max_distance.dtype == np.float32
    assert np.all(leaf_tree.min_distance <= leaf_tree.max_distance)
    assert isinstance(leaf_tree.min_size, np.ndarray)
    assert leaf_tree.min_size.dtype == np.float32
    assert isinstance(leaf_tree.max_size, np.ndarray)
    assert leaf_tree.max_size.dtype == np.float32


def valid_linkage(linkage_tree, X):
    assert isinstance(linkage_tree, LinkageTree)
    assert isinstance(linkage_tree.parent, np.ndarray)
    assert linkage_tree.parent.dtype == np.uint32
    assert np.all(
        linkage_tree.parent.astype(np.int32) - X.shape[0] <= linkage_tree.parent.size
    )
    assert isinstance(linkage_tree.child, np.ndarray)
    assert linkage_tree.child.dtype == np.uint32
    assert np.all(linkage_tree.parent >= linkage_tree.child)
    assert isinstance(linkage_tree.child_count, np.ndarray)
    assert linkage_tree.child_count.dtype == np.uint32
    assert isinstance(linkage_tree.child_size, np.ndarray)
    assert linkage_tree.child_size.dtype == np.float32
    assert np.all(linkage_tree.child_size >= 0)
    assert np.all(np.isfinite(linkage_tree.child_size))


def valid_condensed(condensed_tree, X):
    assert isinstance(condensed_tree, CondensedTree)
    assert isinstance(condensed_tree.parent, np.ndarray)
    assert condensed_tree.parent.dtype == np.uint32
    assert isinstance(condensed_tree.child, np.ndarray)
    assert condensed_tree.child.dtype == np.uint32
    assert np.all(condensed_tree.parent != condensed_tree.child)
    assert np.all(condensed_tree.parent >= X.shape[0])
    assert isinstance(condensed_tree.distance, np.ndarray)
    assert condensed_tree.distance.dtype == np.float32
    assert np.all(condensed_tree.distance >= 0)
    assert isinstance(condensed_tree.child_size, np.ndarray)
    assert condensed_tree.child_size.dtype == np.float32
    assert np.all(condensed_tree.child_size >= 0)
