"""Plot tests for leaf tree visualizations."""

import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from fast_plscan import PLSCAN


@image_comparison(
    baseline_images=["leaf_tree"], extensions=["png"], style="mpl20", tol=3.0
)
def test_leaf_tree(knn):
    plt.figure()
    PLSCAN(min_samples=7, metric="precomputed").fit(knn).leaf_tree_.plot(
        colorbar_kws=dict(fraction=0.01),
    )


@image_comparison(
    baseline_images=["leaf_tree_args"], extensions=["png"], style="mpl20", tol=3.3
)
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
        label_kws=dict(color="red"),
    )


@image_comparison(
    baseline_images=["leaf_tree_no_palette"], extensions=["png"], style="mpl20", tol=3.3
)
def test_leaf_tree_select_no_palette(knn):
    plt.figure()
    PLSCAN(min_samples=7, metric="precomputed").fit(knn).leaf_tree_.plot(
        select_clusters=True, selection_palette=None
    )


@image_comparison(
    baseline_images=["leaf_tree_no_clusters"],
    extensions=["png"],
    style="mpl20",
    tol=4.2,
)
def test_leaf_tree_no_selected_clusters(X):
    # All noise triggers best_size fallback path.
    plt.figure()
    PLSCAN(min_cluster_size=X.shape[0]).fit(X).leaf_tree_.plot(
        select_clusters=True, label_clusters=True
    )


def test_leaf_tree_invalid_width(knn):
    c = PLSCAN(min_samples=7, metric="precomputed").fit(knn)
    with pytest.raises(ValueError):
        c.leaf_tree_.plot(width="invalid")


@image_comparison(
    baseline_images=["leaf_tree_deep_hierarchy"],
    extensions=["png"],
    style="mpl20",
    tol=3.0,
)
def test_leaf_tree_deep_hierarchy(X):
    plt.figure()
    PLSCAN(min_cluster_size=3, min_samples=3).fit(X).leaf_tree_.plot(
        select_clusters=True
    )
