"""Tests for single linkage tree post-fit behavior."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from fast_plscan import PLSCAN


def test_export_single_linkage_tree_numpy(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    arr = c.single_linkage_tree_
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (c._linkage_tree.parent.size, 4)


def test_not_fitted_single_linkage_tree_attribute():
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        c.single_linkage_tree_

