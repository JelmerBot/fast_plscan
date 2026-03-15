"""Tests for leaf tree post-fit behavior."""

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
def test_leaf_tree_networkx_attributes(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    g = c.leaf_tree_.to_networkx()
    for u, v, data in g.edges(data=True):
        assert "size" in data
        assert "distance" in data

    source_nodes = [n for n in g.nodes if n >= c._num_points]
    assert len(source_nodes) > 0
    for node in source_nodes:
        attrs = g.nodes[node]
        for key in ("min_size", "max_size", "min_distance", "max_distance"):
            assert key in attrs


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_export_leaf_tree_pandas(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    df = c.leaf_tree_.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (c._leaf_tree.parent.size, 5)


def test_export_leaf_tree_numpy(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    arr = c.leaf_tree_.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (c._leaf_tree.parent.size,)


def test_not_fitted_leaf_tree_attribute():
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        c.leaf_tree_
