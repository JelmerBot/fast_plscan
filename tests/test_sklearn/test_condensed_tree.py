"""Tests for condensed tree post-fit behavior."""

import numpy as np
import pytest

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import pandas as pd
except ImportError:
    pd = None

from sklearn.exceptions import NotFittedError

from fast_plscan import PLSCAN


@pytest.mark.skipif(nx is None, reason="networkx not installed")
def test_condensed_tree_networkx_attributes(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    g = c.condensed_tree_.to_networkx()
    for u, v, data in g.edges(data=True):
        if u != c._num_points:
            assert "distance" in data
            assert "density" in data
            assert data["distance"] >= 0
            assert 0 < data["density"] <= 1


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_export_condensed_tree_pandas(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    df = c.condensed_tree_.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (c._condensed_tree.parent.size, 5)


def test_export_condensed_tree_numpy(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    arr = c.condensed_tree_.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (c._condensed_tree.parent.size,)


def test_not_fitted_condensed_tree_attribute():
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        c.condensed_tree_

