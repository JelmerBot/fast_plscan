"""Plot tests for leaf tree visualizations."""

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from fast_plscan import PLSCAN


@image_comparison(baseline_images=["leaf_tree"], extensions=["png"], style="mpl20")
def test_leaf_tree(knn):
    plt.figure()
    PLSCAN(min_samples=7, metric="precomputed").fit(knn).leaf_tree_.plot()


@image_comparison(baseline_images=["leaf_tree_args"], extensions=["png"], style="mpl20")
def test_leaf_tree_args(knn):
    plt.figure()
    PLSCAN(min_samples=7, metric="precomputed").fit(knn).leaf_tree_.plot(
        leaf_separation=0.5,
        width="density",
        cmap="turbo",
        colorbar=False,
        label_clusters=True,
        select_clusters=True,
        selection_palette="tab20",
        connect_line_kws=dict(linewidth=0.4),
        parent_line_kws=dict(color="red"),
        colorbar_kws=dict(fraction=0.01),
        label_kws=dict(color="red"),
    )
