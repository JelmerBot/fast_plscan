"""Plot tests for persistence trace visualizations."""

import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from fast_plscan import PLSCAN


@image_comparison(
    baseline_images=["persistence_trace"], extensions=["png"], style="mpl20", tol=7.3
)
def test_persistence_trace(knn):
    plt.figure()
    PLSCAN(min_samples=7, metric="precomputed").fit(knn).persistence_trace_.plot()


@image_comparison(
    baseline_images=["persistence_trace_args"],
    extensions=["png"],
    style="mpl20",
    tol=7.3,
)
def test_persistence_trace_args(knn):
    plt.figure()
    PLSCAN(min_samples=7, metric="precomputed").fit(knn).persistence_trace_.plot(
        line_kws=dict(color="black", linewidth=0.5)
    )


@pytest.mark.parametrize("persistence_measure", ["size-distance", "size-density"])
def test_persistence_trace_bi_persistence(knn, persistence_measure):
    """Bi-persistence measures produce a valid 1D trace and must render without error."""
    c = PLSCAN(
        min_samples=7, metric="precomputed", persistence_measure=persistence_measure
    ).fit(knn)
    plt.figure()
    c.persistence_trace_.plot()
    plt.close()
    plt.figure()
    c.leaf_tree_.plot()
    plt.close()
