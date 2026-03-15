import numpy as np
from scipy.sparse.csgraph import connected_components

from fast_plscan._api import (
    SparseGraph,
    SpanningTree,
    LinkageTree,
    CondensedTree,
    LeafTree,
    PersistenceTrace,
)
from fast_plscan._helpers import to_scipy_csr


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


def valid_mutual_graph(mut_graph, X):
    assert isinstance(mut_graph, SparseGraph)
    assert mut_graph.indptr.shape[0] == X.shape[0] + 1
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


def valid_cluster_outputs(
    labels,
    probabilities,
    selected_clusters,
    persistence_trace,
    leaf_tree,
    condensed_tree,
    linkage_tree,
    X,
):
    valid_labels(labels, X)
    valid_probabilities(probabilities, X)
    valid_selected_clusters(selected_clusters, labels)
    valid_persistence_trace(persistence_trace)
    valid_leaf(leaf_tree)
    valid_condensed(condensed_tree, X)
    valid_linkage(linkage_tree, X)


def valid_fitted_clustering_state(
    clusterer,
    X,
    expect_mutual_graph=None,
    expect_neighbors=None,
    expect_core_distances=True,
):
    valid_spanning_forest(clusterer._minimum_spanning_tree, X)

    if expect_mutual_graph is True:
        valid_mutual_graph(clusterer._mutual_graph, X)
    elif expect_mutual_graph is False:
        assert clusterer._mutual_graph is None

    if expect_neighbors is True:
        valid_neighbor_indices(clusterer._neighbors, X, clusterer.min_samples)
    elif expect_neighbors is False:
        assert clusterer._neighbors is None

    if expect_core_distances:
        valid_core_distances(clusterer.core_distances_, X)
    else:
        assert clusterer.core_distances_ is None

    valid_cluster_outputs(
        clusterer.labels_,
        clusterer.probabilities_,
        clusterer.selected_clusters_,
        clusterer._persistence_trace,
        clusterer._leaf_tree,
        clusterer._condensed_tree,
        clusterer._linkage_tree,
        X,
    )


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


def valid_centroids(centroids, X, labels):
    n_clusters = int(labels.max()) + 1
    assert isinstance(centroids, np.ndarray)
    assert centroids.dtype == np.float32
    assert centroids.shape == (n_clusters, X.shape[1])
    assert np.all(np.isfinite(centroids))


def valid_medoid_indices(medoid_indices, X, labels):
    n_clusters = int(labels.max()) + 1
    assert isinstance(medoid_indices, np.ndarray)
    assert medoid_indices.dtype.kind in ("i", "u")
    assert medoid_indices.shape == (n_clusters,)
    assert np.all(medoid_indices >= 0)
    assert np.all(medoid_indices < X.shape[0])
    for c, idx in enumerate(medoid_indices):
        assert labels[idx] == c


def valid_exemplar_indices(exemplars_per_cluster, X, labels):
    n_clusters = max(0, int(labels.max()) + 1)
    assert isinstance(exemplars_per_cluster, list)
    assert len(exemplars_per_cluster) == n_clusters
    for c, exemplars in enumerate(exemplars_per_cluster):
        assert isinstance(exemplars, np.ndarray)
        assert exemplars.ndim == 1
        assert exemplars.dtype.kind in ("i", "u")
        assert np.all(exemplars >= 0)
        assert np.all(exemplars < X.shape[0])
        assert np.all(labels[exemplars] == c)


def valid_membership_vectors(mv, X, labels):
    n_clusters = max(0, int(labels.max()) + 1)
    assert isinstance(mv, np.ndarray)
    assert mv.dtype == np.float32
    assert mv.shape == (X.shape[0], n_clusters)
    assert np.all(mv >= 0.0)
    assert np.all(np.isfinite(mv))
    assert np.all(mv.sum(axis=1) <= 1.0 + 1e-5)


def assert_zero_membership_across_components(c, mv):
    labels = c.labels_
    n_clusters = int(labels.max()) + 1
    if n_clusters <= 0:
        return

    g = to_scipy_csr(c._mutual_graph)
    g = g.maximum(g.T)
    _, component = connected_components(g, directed=False, return_labels=True)
    all_clusters = set(range(n_clusters))

    for comp in np.unique(component):
        point_mask = component == comp
        component_clusters = set(labels[point_mask][labels[point_mask] >= 0])
        other_clusters = sorted(all_clusters - component_clusters)
        if other_clusters:
            assert np.all(
                mv[point_mask][:, other_clusters] == 0
            ), "Memberships to clusters in other connected components should be zero"
