"""Sklearn estimator compatibility tests."""

import pickle

from sklearn.utils.estimator_checks import check_estimator

from fast_plscan import PLSCAN
from ..checks import *


def test_pickle_fitted(X, knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    loaded = pickle.loads(pickle.dumps(c))
    assert np.array_equal(c.labels_, loaded.labels_)
    assert np.allclose(c.probabilities_, loaded.probabilities_)
    valid_fitted_clustering_state(loaded, X)
    labels, probs = loaded.distance_cut(0.5)
    valid_labels(labels, X)
    valid_probabilities(probs, X)


def test_hdbscan_is_sklearn_estimator():
    check_estimator(PLSCAN())
