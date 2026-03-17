"""Tests for persistence trace post-fit behavior."""

import numpy as np
import pytest

try:
    import pandas as pd
except ImportError:
    pd = None

from sklearn.exceptions import NotFittedError

from fast_plscan import PLSCAN


@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_export_persistence_trace_pandas(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    df = c.persistence_trace_.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (c._persistence_trace.min_size.size, 2)


def test_export_persistence_trace_numpy(knn):
    c = PLSCAN(metric="precomputed").fit(knn)
    arr = c.persistence_trace_.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (c._persistence_trace.min_size.size,)


def test_not_fitted_persistence_trace_attribute():
    c = PLSCAN()
    with pytest.raises(NotFittedError):
        c.persistence_trace_
