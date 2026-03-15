"""Public API for plotting and exporting condensed trees, leaf trees, and
persistence traces."""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import rankdata
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Ellipse
from matplotlib.colors import Colormap, BoundaryNorm
from typing import Any, Literal

from .._api import (
    LeafTree as LeafTreeTuple,
    CondensedTree as CondensedTreeTuple,
    PersistenceTrace as PersistenceTraceTuple,
    compute_distance_icicles,
    compute_density_icicles,
)


class LeafTree(object):
    """
    A tree describing which clusters exist and how they split along increasing
    minimum cluster size thresholds. See the documentation for the `to_*`
    conversion methods for details on the output formats!
    """

    def __init__(
        self,
        leaf_tree: LeafTreeTuple,
        condensed_tree: CondensedTreeTuple,
        selected_clusters: np.ndarray[tuple[int], np.dtype[np.uint32]],
        persistence_trace: PersistenceTraceTuple,
        num_points: int,
    ):
        """
        Parameters
        ----------
        leaf_tree
            The leaf tree object as produced internally.
        condensed_tree
            The condensed tree object as produced internally.
        selected_clusters
            The leaf tree parent IDs for the selected clusters.
        persistence_trace
            The persistence trace for the leaf tree.
        num_points
            The number of points in the leaf tree.
        """
        self._tree = leaf_tree
        self._condensed_tree = condensed_tree
        self._chosen_segments = {c: i for i, c in enumerate(selected_clusters)}
        self._persistence_trace = persistence_trace
        self._num_points = num_points

    def to_numpy(self) -> np.ndarray:
        """Returns a numpy structured array of the leaf tree.

        Each row represents a segment in the condensed tree, with the first row
        representing the phantom root.

        The `parent` column indicates the parent cluster ID for each segment.
        These IDs start from 0 and are row-indices into the leaf tree. The
        phantom root has itself as a parent to indicate that it is the root of
        the tree.

        The `min_distance` and `max_distance` columns form a right-open [birth,
        death) interval, indicating at which distance thresholds clusters exist.

        The `min_size` and `max_size` columns form a left-open (birth, death]
        interval, indicating at which min cluster size thresholds clusters are
        leaves. If `max_size` <= `min_size`, the cluster is not a leaf.
        """
        dtype = [
            ("parent", np.uint32),
            ("min_distance", np.float32),
            ("max_distance", np.float32),
            ("min_size", np.float32),
            ("max_size", np.float32),
        ]
        result = np.empty(self._tree.parent.shape[0], dtype=dtype)
        result["parent"] = self._tree.parent
        result["min_distance"] = self._tree.min_distance
        result["max_distance"] = self._tree.max_distance
        result["min_size"] = self._tree.min_size
        result["max_size"] = self._tree.max_size
        return result

    def to_pandas(self):
        """Return a pandas dataframe of the leaf tree.

        Each row represents a segment in the condensed tree, with the first row
        representing the phantom root.

        The `parent` column indicates the parent cluster ID for each segment.
        These IDs start from 0 and are row-indices into the leaf tree. The
        phantom root has itself as a parent to indicate that it is the root of
        the tree.

        The `min_distance` and `max_distance` columns form a right-open [birth,
        death) interval, indicating at which distance thresholds clusters exist.

        The `min_size` and `max_size` columns form a left-open (birth, death]
        interval, indicating at which min cluster size thresholds clusters are
        leaves. If `max_size` <= `min_size`, the cluster is not a leaf.
        """
        try:
            from pandas import DataFrame
        except ImportError:
            raise ImportError(
                "You must have pandas installed to export pandas DataFrames"
            )

        return DataFrame(
            dict(
                parent=self._tree.parent,
                min_distance=self._tree.min_distance,
                max_distance=self._tree.max_distance,
                min_size=self._tree.min_size,
                max_size=self._tree.max_size,
            )
        )

    def to_networkx(self):
        """Return a NetworkX DiGraph object representing the leaf tree.

        Edges have a `size` and `distance` attribute giving the cluster size
        threshold and distance at which the child connects to the parent. The
        `child` and `parent` values start from 0 and are row-indices into the
        leaf tree. The phantom root has itself as a parent to indicate that it
        is the root of the tree.

        Nodes have `min_size`, `max_size`, `min_distance`, `max_distance`
        attributes. The `min_distance` and `max_distance` columns form a
        right-open [birth, death) interval, indicating at which distance
        thresholds clusters exist. The `min_size` and `max_size` columns form a
        left-open (birth, death] interval, indicating at which min cluster size
        thresholds clusters are leaves. If `max_size` <= `min_size`, the cluster
        is not a leaf.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "You must have networkx installed to export networkx graphs"
            )

        g = nx.DiGraph(
            {(i + self._num_points, pt) for i, pt in enumerate(self._tree.parent)}
        )
        nx.set_edge_attributes(
            g,
            {
                (i + self._num_points, pt): dict(size=size, distance=dist)
                for i, (pt, size, dist) in enumerate(
                    zip(self._tree.parent, self._tree.max_size, self._tree.max_distance)
                )
            },
        )
        nx.set_node_attributes(
            g,
            {
                (i + self._num_points): dict(
                    min_size=min_size,
                    max_size=max_size,
                    min_distance=min_dist,
                    max_distance=max_dist,
                )
                for i, (min_size, max_size, min_dist, max_dist) in enumerate(
                    zip(
                        self._tree.min_size,
                        self._tree.max_size,
                        self._tree.min_distance,
                        self._tree.max_distance,
                    )
                )
            },
        )
        return g

    def plot(
        self,
        *,
        width: Literal["distance", "density"] = "distance",
        leaf_separation: float = 0.8,
        cmap: str | Colormap = "viridis_r",
        colorbar: bool = True,
        label_clusters: bool = False,
        select_clusters: bool = False,
        selection_palette: str | Colormap = "tab10",
        connect_line_kws: dict[str, Any] | None = None,
        parent_line_kws: dict[str, Any] | None = None,
        colorbar_kws: dict[str, Any] | None = None,
        label_kws: dict[str, Any] | None = None,
    ):
        """
        Creates an icicle plot of the leaf tree.

        Parameters
        ----------
        width
            Which cluster stability measure to use for the width of the
            segments. Can be one of "distance" or "density", determining whether
            distance or density persistences are used. The stability measure sum
            the persistences over all points in the cluster. These persistences
            change with the minimum cluster size threshold, as that threshold
            determines the lowest distance at which enough points are connected
            to be considered a cluster.
        leaf_separation
            A spacing parameter for icicle positioning.
        cmap
            The colormap to use for the segments.
        colorbar
            Whether to show a colorbar for the cluster size.
        label_clusters
            If True, the cluster labels are plotted on the icicle segments.
        select_clusters
            If True, the segments representing selected clusters are highlighted
            with ellipses.
        selection_palette
            A list of colors to highlight selected clusters.
        connect_line_kws
            Additional keyword arguments for the connecting lines between
            segments.
        parent_line_kws
            Additional keyword arguments for the parent lines connecting the
            segments to their parents.
        colorbar_kws
            Additional keyword arguments for the colorbar.
        label_kws
            Additional keyword arguments for the cluster labels.
        """

        # Compute the layout
        parents = np.empty_like(self._tree.parent)
        for idx, parent_idx in enumerate(self._tree.parent):
            parents[idx] = self._leaf_parent(parent_idx)
        x_coords = self._x_coords(parents) * leaf_separation

        # Prepare the labels
        _label_kws = dict(ha="center", va="bottom", fontsize=8)
        if label_kws is not None:
            _label_kws.update(label_kws)

        # vertical lines connecting death of leaf cluster to birth of parent cluster
        parent_lines = []
        _parent_line_kws = dict(linestyle=":", color="black", linewidth=0.5)
        if parent_line_kws is not None:
            _parent_line_kws.update(parent_line_kws)

        # horizontal lines connecting leaf cluster to its parent cluster
        connect_lines = []
        _connect_line_kws = dict(linestyle="-", color="black", linewidth=0.5)
        if connect_line_kws is not None:
            _connect_line_kws.update(connect_line_kws)

        # List cluster label for segments representing selected clusters
        ellipses = []
        if select_clusters:
            if selection_palette is None:
                ellipse_colors = ["r"]
            else:
                ellipse_colors = plt.get_cmap(selection_palette).colors

        if len(self._chosen_segments) == 0:
            best_size = self._tree.max_size[0] / 2
        else:
            best_size = max(
                self._tree.min_size[k] for k in self._chosen_segments.keys()
            )
        cmap = plt.get_cmap(cmap)
        cmap_norm = BoundaryNorm(np.linspace(1, 10, 10), cmap.N)
        min_size_traces, width_traces = self._compute_icicle_traces(width)
        non_empty_traces = [trace[0] for trace in width_traces if trace.size > 0]
        if len(non_empty_traces) == 0:
            max_width = 1
        else:
            max_width = max(non_empty_traces)

        bar = None
        for leaf_idx, (parent_idx, size_trace, width_trace) in enumerate(
            zip(parents[1:], min_size_traces[1:], width_traces[1:]), 1
        ):
            # skip segments that are not leaves
            if self._tree.max_size[leaf_idx] <= self._tree.min_size[leaf_idx]:
                continue

            # draw lines connecting the leaf cluster to its parent
            x = x_coords[leaf_idx]
            y_start = (
                self._tree.max_size[leaf_idx]
                if parent_idx > 0
                else self._tree.min_size[leaf_idx]
            )
            parent_lines.append(
                [
                    (x, y_start),
                    (x, self._tree.min_size[parent_idx]),
                ]
            )

            # don't draw anything else for root clusters
            if parent_idx == 0:
                continue

            # draw horizontal connecting line to parent
            segment_x = x
            if size_trace.size > 0:
                offset = width_trace[-1] / max_width * 0.25
                if x > x_coords[parent_idx]:
                    segment_x = x + offset
                else:
                    segment_x = x - offset
            connect_lines.append(
                [
                    (segment_x, self._tree.min_size[parent_idx]),
                    (x_coords[parent_idx], self._tree.min_size[parent_idx]),
                ]
            )

            # add Ellipse for selected segments
            if (
                label_clusters or select_clusters
            ) and leaf_idx in self._chosen_segments:
                max_size = self._tree.max_size[leaf_idx]
                if size_trace.shape[0] == 0:
                    min_size = self._tree.max_size[leaf_idx]
                else:
                    min_size = size_trace[0]
                center = (x_coords[leaf_idx], (max_size + min_size) / 2)
                height = max_size - min_size
                width = width_trace[0] if width_trace.size > 0 else 0
                width /= max_width
                ellipse = Ellipse(center, leaf_separation / 2 + width / 2, 1.2 * height)
                if label_clusters:
                    if leaf_idx in self._chosen_segments:
                        plt.text(
                            center[0],
                            best_size,
                            len(ellipses),
                            **_label_kws,
                        )
                if select_clusters:
                    ellipses.append(ellipse)

            # draw the icicle segment
            if size_trace.size > 0:
                xs = np.asarray([[x], [x]])
                widths = xs + width_trace / max_width * np.array([[-0.25], [0.25]])

                j = 0
                measure = np.empty_like(size_trace)
                measure_ranks = rankdata(
                    -self._persistence_trace.persistence, method="min"
                )
                for i, size in enumerate(self._persistence_trace.min_size):
                    while j < len(size_trace) and size_trace[j] < size:
                        measure[j] = measure_ranks[i - 1]
                        j += 1

                bar = plt.pcolormesh(
                    widths,
                    np.broadcast_to(size_trace, (2, len(size_trace))),
                    np.broadcast_to(measure, (2, len(size_trace))),
                    edgecolors="none",
                    linewidth=0,
                    cmap=cmap,
                    norm=cmap_norm,
                    shading="gouraud",
                )

        plt.gca().add_collection(LineCollection(parent_lines, **_parent_line_kws))
        plt.gca().add_collection(LineCollection(connect_lines, **_connect_line_kws))
        if select_clusters:
            plt.gca().add_collection(
                PatchCollection(
                    ellipses,
                    facecolor="none",
                    linewidth=2,
                    edgecolors=[
                        ellipse_colors[s % len(ellipse_colors)]
                        for s in range(len(ellipses))
                    ],
                )
            )

        # Plot the colorbar
        if colorbar and bar is not None:
            if colorbar_kws is None:
                colorbar_kws = dict(extend="max")

            if "fraction" in colorbar_kws:
                bbox = plt.gca().get_window_extent()
                ax_width, ax_height = bbox.width, bbox.height
                colorbar_kws["aspect"] = ax_height / (
                    ax_width * colorbar_kws["fraction"]
                )

            plt.colorbar(bar, label=f"Cut rank", **colorbar_kws)

        for side in ("right", "top", "bottom"):
            plt.gca().spines[side].set_visible(False)

        plt.xticks([])
        plt.xlim(x_coords.min() - leaf_separation, x_coords.max() + leaf_separation)
        plt.ylim(0, self._tree.min_size[0])
        plt.ylabel("Min cluster size")
        return x_coords

    def _leaf_parent(self, parent_idx: int):
        """Get the leaf-cluster parent of a leaf cluster."""
        while (
            self._tree.parent[parent_idx] > 0
            and self._tree.max_size[parent_idx] <= self._tree.min_size[parent_idx]
        ):
            parent_idx = self._tree.parent[parent_idx]
        return parent_idx

    def _compute_icicle_traces(self, width: Literal["distance", "density"]):
        # Lists the size--distance-persistence trace for each cluster
        if width == "distance":
            fun = compute_distance_icicles
        elif width == "density":
            fun = compute_density_icicles
        else:
            raise ValueError(f"Unknown width option '{width}'")
        sizes, traces = fun(self._tree, self._condensed_tree, self._num_points)

        # Compute stability and truncate to min_cluster_size lifetime
        upper_idx = [
            np.searchsorted(s, d, side="right")
            for d, s in zip(self._tree.max_size, sizes)
        ]
        stabilities = [
            (s * t + np.concatenate((np.cumsum(t[1:][::-1])[::-1], [0])))[:i]
            for s, t, i in zip(sizes, traces, upper_idx)
        ]
        sizes = [s[:i] for s, i in zip(sizes, upper_idx)]
        return sizes, stabilities

    @staticmethod
    def _min_point_per_segment(
        condensed_tree: CondensedTreeTuple,
        leaf_parent: np.ndarray,
        num_points: int,
    ) -> np.ndarray:
        """For each leaf-tree segment, the minimum data-point index in its subtree."""
        n_segments = len(leaf_parent)
        min_pts = np.full(n_segments, num_points, dtype=np.intp)
        mask = condensed_tree.child < num_points
        leaf_indices = condensed_tree.parent[mask].astype(np.intp) - num_points
        np.minimum.at(min_pts, leaf_indices, condensed_tree.child[mask])
        for segment_idx in range(n_segments - 1, 0, -1):
            parent_idx = int(leaf_parent[segment_idx])
            if min_pts[segment_idx] < min_pts[parent_idx]:
                min_pts[parent_idx] = min_pts[segment_idx]
        return min_pts

    def _x_coords(self, parents: np.ndarray[tuple[int], np.dtype[np.uint32]]):
        """Get the x-coordinates of the segments in the condensed tree."""
        children = dict()
        for child_idx, parent_idx in enumerate(parents[1:], 1):
            if (
                parent_idx > 0
                and self._tree.max_size[child_idx] <= self._tree.min_size[child_idx]
            ):
                continue
            if parent_idx not in children:
                children[parent_idx] = []
            children[parent_idx].append(child_idx)
        min_pts = LeafTree._min_point_per_segment(
            self._condensed_tree, self._tree.parent, self._num_points
        )
        for segments in children.values():
            segments.sort(key=lambda i: min_pts[i])
        x_coords = np.zeros(parents.shape[0])
        self._df_leaf_order(x_coords, children, 0, 0)
        return x_coords

    @classmethod
    def _df_leaf_order(
        cls,
        x_coords: np.ndarray[tuple[int], np.dtype[np.float64]],
        children: dict[int, list[int]],
        idx: int,
        count: int,
    ) -> tuple[list[tuple[int, float]], float, int]:
        """Depth-first (in-order) traversal to order the leaf clusters."""
        if idx not in children:
            x_coords[idx] = float(count)
            return count, count + 1

        segments = children[idx]
        collected = []
        for child in segments:
            child_xs, count = cls._df_leaf_order(x_coords, children, child, count)
            collected.append(child_xs)
        mid = (min(collected) + max(collected)) / 2
        x_coords[idx] = mid
        return mid, count
