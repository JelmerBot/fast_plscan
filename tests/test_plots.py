"""Tests for the sklearn interface."""

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from plscan import PLSCAN


@image_comparison(
    baseline_images=["condensed_tree"], extensions=["png"], style="mpl20", tol=5.95
)
def test_condensed_tree(knn):
    plt.figure()
    PLSCAN(metric="precomputed").fit(knn).condensed_tree_.plot()


@image_comparison(
    baseline_images=["condensed_tree_args"], extensions=["png"], style="mpl20", tol=0.56
)
def test_condensed_tree_args(knn):
    plt.figure()
    PLSCAN(metric="precomputed").fit(knn).condensed_tree_.plot(
        leaf_separation=0.5,
        cmap="turbo",
        colorbar=False,
        log_size=True,
        distance_ranks=False,
        label_clusters=True,
        select_clusters=True,
        selection_palette="tab20",
        continuation_line_kws=dict(color="red"),
        connect_line_kws=dict(linewidth=0.4),
        colorbar_kws=dict(fraction=0.01),
        label_kws=dict(color="red"),
    )


@image_comparison(
    baseline_images=["leaf_tree"], extensions=["png"], style="mpl20", tol=1.85
)
def test_leaf_tree(knn):
    plt.figure()
    PLSCAN(metric="precomputed").fit(knn).leaf_tree_.plot()


@image_comparison(
    baseline_images=["leaf_tree_args"], extensions=["png"], style="mpl20", tol=0.86
)
def test_leaf_tree_args(knn):
    plt.figure()
    PLSCAN(metric="precomputed").fit(knn).leaf_tree_.plot(
        leaf_separation=0.5,
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


@image_comparison(
    baseline_images=["persistence_trace"], extensions=["png"], style="mpl20", tol=7.26
)
def test_persistence_trace(knn):
    plt.figure()
    PLSCAN(metric="precomputed").fit(knn).persistence_trace_.plot()


@image_comparison(
    baseline_images=["persistence_trace_args"],
    extensions=["png"],
    style="mpl20",
    tol=7.26,
)
def test_persistence_trace_args(knn):
    plt.figure()
    PLSCAN(metric="precomputed").fit(knn).persistence_trace_.plot(
        line_kws=dict(color="black", linewidth=0.5)
    )
