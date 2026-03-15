"""Plot tests for condensed tree visualizations."""

import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from fast_plscan import PLSCAN


@image_comparison(
    baseline_images=["condensed_tree_dist"], extensions=["png"], style="mpl20"
)
def test_condensed_tree_dist(knn):
    plt.figure()
    PLSCAN(min_samples=7, metric="precomputed").fit(knn).condensed_tree_.plot(
        select_clusters=True
    )


@image_comparison(
    baseline_images=["condensed_tree_dens"], extensions=["png"], style="mpl20"
)
def test_condensed_tree_dens(knn):
    plt.figure()
    PLSCAN(min_samples=7, metric="precomputed").fit(knn).condensed_tree_.plot(
        y="density", select_clusters=True
    )


@image_comparison(
    baseline_images=["condensed_tree_rank"], extensions=["png"], style="mpl20"
)
def test_condensed_tree_rank(knn):
    plt.figure()
    PLSCAN(min_samples=7, metric="precomputed").fit(knn).condensed_tree_.plot(
        y="ranks", select_clusters=True
    )


@image_comparison(
    baseline_images=["condensed_tree_args"], extensions=["png"], style="mpl20"
)
def test_condensed_tree_args(knn):
    plt.figure()
    PLSCAN(min_samples=7, metric="precomputed").fit(knn).condensed_tree_.plot(
        leaf_separation=0.5,
        cmap="turbo",
        colorbar=False,
        log_size=True,
        label_clusters=True,
        selection_palette="tab20",
        continuation_line_kws=dict(color="red"),
        connect_line_kws=dict(linewidth=0.4),
        colorbar_kws=dict(fraction=0.01),
        label_kws=dict(color="red"),
    )


def test_condensed_tree_invalid_y(knn):
    c = PLSCAN(min_samples=7, metric="precomputed").fit(knn)
    with pytest.raises(ValueError):
        c.condensed_tree_.plot(y="invalid")
