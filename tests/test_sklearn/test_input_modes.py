"""Tests for sklearn input modes."""

import numpy as np
import pytest

from fast_plscan import PLSCAN

from ..checks import valid_fitted_clustering_state, valid_spanning_forest, valid_labels


def test_mst(X, mst):
    _in = mst.copy()
    c = PLSCAN(metric="precomputed").fit((mst, X.shape[0]))
    assert np.allclose(mst, _in)

    valid_fitted_clustering_state(
        c,
        X,
        expect_mutual_graph=False,
        expect_neighbors=False,
        expect_core_distances=False,
    )
    assert c.labels_.max() == 2


def test_knn_graph(X, knn):
    """A knn matrix with the self-loop first column should produce the same
    results as one without the self-loop first column."""
    _in = (knn[0].copy(), knn[1].copy())
    c = PLSCAN(metric="precomputed").fit(knn)
    assert np.allclose(knn[0], _in[0])
    assert np.allclose(knn[1], _in[1])

    valid_fitted_clustering_state(
        c,
        X,
        expect_mutual_graph=True,
        expect_neighbors=False,
    )
    assert c.labels_.max() == 3
    assert np.any(c.labels_ == -1)


def test_knn_graph_no_loops(X, knn_no_loops):
    """A knn matrix without the self-loop first column should produce the same
    results as one with the self-loop first column."""
    _in = (knn_no_loops[0].copy(), knn_no_loops[1].copy())
    c = PLSCAN(metric="precomputed").fit(knn_no_loops)
    assert np.allclose(knn_no_loops[0], _in[0])
    assert np.allclose(knn_no_loops[1], _in[1])

    valid_fitted_clustering_state(
        c,
        X,
        expect_mutual_graph=True,
        expect_neighbors=False,
    )
    assert c.labels_.max() == 3
    assert np.any(c.labels_ == -1)


def test_distance_matrix(X, dists):
    _in = dists.copy()
    c = PLSCAN(metric="precomputed").fit(dists)
    assert np.allclose(dists, _in)

    valid_fitted_clustering_state(
        c,
        X,
        expect_mutual_graph=True,
        expect_neighbors=False,
    )
    assert c.labels_.max() == 2


def test_condensed_matrix(X, con_dists):
    _in = con_dists.copy()
    c = PLSCAN(metric="precomputed").fit(con_dists)
    assert np.allclose(con_dists, _in)

    valid_fitted_clustering_state(
        c,
        X,
        expect_mutual_graph=True,
        expect_neighbors=False,
    )
    assert c.labels_.max() == 2


def test_sparse_matrix(X, g_knn):
    _in = g_knn.copy()
    c = PLSCAN(metric="precomputed").fit(g_knn)
    assert np.allclose(g_knn.data, _in.data)
    assert np.allclose(g_knn.indices, _in.indices)

    valid_fitted_clustering_state(
        c,
        X,
        expect_mutual_graph=True,
        expect_neighbors=False,
    )
    assert c.labels_.max() == 3
    assert np.any(c.labels_ == -1)


def test_bad_mst_input(mst):
    with pytest.raises(ValueError):
        PLSCAN(metric="precomputed").fit((mst,))
    with pytest.raises(ValueError):
        PLSCAN(metric="precomputed").fit((mst, 200, "extra"))


def test_bad_mst_num_points(mst):
    with pytest.raises(ValueError):
        PLSCAN(metric="precomputed", min_samples=5).fit((mst, 5))


def test_bad_mst_edge_shape(X):
    with pytest.raises(ValueError):
        PLSCAN(metric="precomputed").fit((np.ones((5, 2)), X.shape[0]))
    with pytest.raises(ValueError):
        PLSCAN(metric="precomputed").fit((np.ones((X.shape[0] + 1, 3)), X.shape[0]))


def test_bad_knn_input(knn):
    with pytest.raises(ValueError):
        PLSCAN(metric="precomputed").fit((knn[0],))


def test_bad_knn_tuple_length(knn):
    # Three-element tuple with a numpy second element routes to _check_knn,
    # which rejects tuples with length != 2.
    with pytest.raises(ValueError):
        PLSCAN(metric="precomputed").fit((knn[0], knn[1], knn[0]))


def test_bad_knn_shape_mismatch(X):
    n = X.shape[0]
    with pytest.raises(ValueError):
        PLSCAN(metric="precomputed").fit(
            (np.ones((n, 8), dtype=np.float32), np.ones((n, 9), dtype=np.int32))
        )


def test_non_square_distance_matrix(X):
    with pytest.raises(ValueError):
        PLSCAN(metric="precomputed").fit(np.ones((X.shape[0], X.shape[0] + 1)))


def test_feature_vector_spanning_tree(X):
    """Feature vector input always produces a full spanning tree (n-1 edges), not a forest."""
    c = PLSCAN().fit(X)
    assert c._minimum_spanning_tree.parent.size == X.shape[0] - 1
    valid_spanning_forest(c._minimum_spanning_tree, X)
    valid_labels(c.labels_, X)
