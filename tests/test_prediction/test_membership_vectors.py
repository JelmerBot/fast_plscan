"""Tests for membership_vectors."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from fast_plscan import PLSCAN
from fast_plscan.prediction import approximate_predict, membership_vectors

from ..checks import valid_membership_vectors

# --- Positive Input Modes


def test_membership_vectors_unseen_points_subset(X):
    c = PLSCAN().fit(X)
    X_new = X[:10]
    mv = membership_vectors(c, X_new)
    labels, _ = approximate_predict(c, X_new)
    valid_membership_vectors(mv, X_new, c.labels_)
    assigned = labels >= 0
    if np.any(assigned):
        assert np.all(mv[assigned].argmax(axis=1) == labels[assigned])


def test_membership_vectors_unseen_points_all_noise(X):
    c = PLSCAN(min_cluster_size=X.shape[0]).fit(X)
    X_new = X[:8]
    mv = membership_vectors(c, X_new)
    valid_membership_vectors(mv, X_new, c.labels_)


# --- Negative Input Modes


def test_bad_membership_vectors_precomputed_sparse(g_dists):
    c = PLSCAN(metric="precomputed").fit(g_dists)
    with pytest.raises(ValueError):
        membership_vectors(c, np.zeros((3, 2), dtype=np.float32))


def test_bad_membership_vectors_precomputed_mst(X, mst):
    c = PLSCAN(metric="precomputed").fit((mst, X.shape[0]))
    with pytest.raises(ValueError):
        membership_vectors(c, np.zeros((3, 2), dtype=np.float32))


def test_bad_membership_vectors_precomputed_knn(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    with pytest.raises(ValueError):
        membership_vectors(c, np.zeros((3, 2), dtype=np.float32))


def test_bad_membership_vectors_unfitted(X):
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        membership_vectors(c, X)


# --- Parameters


def test_membership_vectors_unseen_points_custom_labels(X):
    c = PLSCAN().fit(X)
    X_new = X[:10]
    labels, _ = c.min_cluster_size_cut(c._persistence_trace.min_size[0])
    mv = membership_vectors(c, X_new, labels)
    valid_membership_vectors(mv, X_new, labels)


def test_bad_membership_vectors_wrong_num_features(X):
    c = PLSCAN().fit(X)
    with pytest.raises(ValueError):
        membership_vectors(c, np.zeros((5, X.shape[1] + 1), dtype=np.float32))


def test_bad_membership_vectors_wrong_labels_shape(X):
    c = PLSCAN().fit(X)
    with pytest.raises(ValueError):
        membership_vectors(c, X[:5], np.zeros(5, dtype=np.int64))
