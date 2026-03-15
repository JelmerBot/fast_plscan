"""Tests for minimum spanning tree post-fit behavior."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from fast_plscan import PLSCAN


def test_export_minimum_spanning_tree_numpy(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    arr = c.minimum_spanning_tree_
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (c._minimum_spanning_tree.parent.size, 3)


def test_not_fitted_minimum_spanning_tree_attribute():
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        c.minimum_spanning_tree_
