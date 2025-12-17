"""Tests for the sklearn interface."""

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from fast_plscan import PLSCAN


@image_comparison(
    baseline_images=["condensed_tree_dist"],
    extensions=["png"],
    style="mpl20",
    tol=12.71,  # branches can switch places without changing meaning
)
def test_condensed_tree_dist(knn):
    plt.figure()
    PLSCAN(min_samples=7, metric="precomputed").fit(knn).condensed_tree_.plot(
        select_clusters=True
    )


@image_comparison(
    baseline_images=["condensed_tree_dens"],
    extensions=["png"],
    style="mpl20",
    tol=13.25,  # branches can switch places without changing meaning
)
def test_condensed_tree_dens(knn):
    plt.figure()
    PLSCAN(min_samples=7, metric="precomputed").fit(knn).condensed_tree_.plot(
        y="density", select_clusters=True
    )


@image_comparison(
    baseline_images=["condensed_tree_rank"],
    extensions=["png"],
    style="mpl20",
    tol=20.45,  # branches can switch places without changing meaning
)
def test_condensed_tree_rank(knn):
    plt.figure()
    PLSCAN(min_samples=7, metric="precomputed").fit(knn).condensed_tree_.plot(
        y="ranks", select_clusters=True
    )


@image_comparison(
    baseline_images=["condensed_tree_args"],
    extensions=["png"],
    style="mpl20",
    tol=2.32,  # branches can switch places without changing meaning
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


@image_comparison(
    baseline_images=["leaf_tree"],
    extensions=["png"],
    style="mpl20",
    tol=5.60,  # correct MST orderings can change merge locations
)
def test_leaf_tree(knn):
    plt.figure()
    PLSCAN(min_samples=7, metric="precomputed").fit(knn).leaf_tree_.plot()


@image_comparison(
    baseline_images=["leaf_tree_args"],
    extensions=["png"],
    style="mpl20",
    tol=8.85,  # correct MST orderings can change merge locations
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
        colorbar_kws=dict(fraction=0.01),
        label_kws=dict(color="red"),
    )


@image_comparison(
    baseline_images=["persistence_trace"],
    extensions=["png"],
    style="mpl20",
    tol=16.81,  # correct MST orderings can change merge locations
)
def test_persistence_trace(knn):
    plt.figure()
    PLSCAN(min_samples=7, metric="precomputed").fit(knn).persistence_trace_.plot()


@image_comparison(
    baseline_images=["persistence_trace_args"],
    extensions=["png"],
    style="mpl20",
    tol=15.41,  # correct MST orderings can change merge locations
)
def test_persistence_trace_args(knn):
    plt.figure()
    PLSCAN(min_samples=7, metric="precomputed").fit(knn).persistence_trace_.plot(
        line_kws=dict(color="black", linewidth=0.5)
    )
