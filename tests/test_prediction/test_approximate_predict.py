"""Tests for approximate_predict."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from fast_plscan import PLSCAN
from fast_plscan.prediction import approximate_predict

from ..checks import *


# --- Positive Input Modes


def test_approximate_predict_basic(X):
    c = PLSCAN().fit(X)
    labels, probabilities = approximate_predict(c, X)
    valid_labels(labels, X)
    valid_probabilities(probabilities, X)


def test_approximate_predict_subset(X):
    c = PLSCAN().fit(X)
    X_new = X[:10]
    labels, probabilities = approximate_predict(c, X_new)
    valid_labels(labels, X_new)
    valid_probabilities(probabilities, X_new)


# --- Negative Input Modes


def test_bad_approximate_predict_precomputed_sparse(g_dists):
    c = PLSCAN(metric="precomputed").fit(g_dists)
    with pytest.raises(ValueError):
        approximate_predict(c, np.zeros((3, 2), dtype=np.float32))


def test_bad_approximate_predict_precomputed_knn(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    with pytest.raises(ValueError):
        approximate_predict(c, np.zeros((3, 2), dtype=np.float32))


def test_bad_approximate_predict_precomputed_mst(X, mst):
    c = PLSCAN(metric="precomputed").fit((mst, X.shape[0]))
    with pytest.raises(ValueError):
        approximate_predict(c, np.zeros((3, 2), dtype=np.float32))


def test_bad_approximate_predict_unfitted(X):
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        approximate_predict(c, X)


# --- Parameters


def test_bad_approximate_predict_wrong_num_features(X):
    c = PLSCAN().fit(X)
    with pytest.raises(ValueError):
        approximate_predict(c, np.zeros((5, X.shape[1] + 1), dtype=np.float32))
