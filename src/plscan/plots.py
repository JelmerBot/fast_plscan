"""Public API for plotting and exporting condensed trees, leaf trees, and
persistence traces."""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import rankdata
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Ellipse
from matplotlib.colors import Colormap, BoundaryNorm
from typing import Any, Literal

from ._api import (
    LeafTree as LeafTreeTuple,
    CondensedTree as CondensedTreeTuple,
    PersistenceTrace as PersistenceTraceTuple,
    compute_distance_icicles,
    compute_density_icicles,
)


class CondensedTree(object):
    """
    A tree/forest describing which clusters exist and how they split along
    descending distances. Unlike in HDBSCAN*, this version can represent a
    forest, rather than a single tree. See the documentation on the `to_*`
    conversion methods for details on the output formats!
    """

    def __init__(
        self,
        leaf_tree: LeafTreeTuple,
        condensed_tree: CondensedTreeTuple,
        selected_clusters: np.ndarray[tuple[int], np.dtype[np.uint32]],
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
            The condensed tree parent IDs for the selected clusters.
        num_points
            The number of points in the condensed tree.
        """
        self._leaf_tree = leaf_tree
        self._tree = condensed_tree
        self._chosen_segments = {c: i for i, c in enumerate(selected_clusters)}
        self._num_points = num_points

    def to_numpy(self) -> np.ndarray:
        """Returns a numpy structured array of the condensed tree.

        The columns are: parent, child, distance, density, child_size. The
        parent labelling starts at `num_points`, which represents a phantom
        root. All points connecting directly to the (multiple) tree roots have
        `num_points` as their parent. The labels for the tree roots themselves
        occur only as a parent and start from `num_points + 1`.

        Due to this construction, we cannot recover which points belong to which
        tree root (if there are multiple trees). In addition, the first parent
        value cannot be used to find `num_points`, as it can be the first
        tree-root value `num_points + 1`! All parents that do not occur as
        children should be considered a child of the phantom root.
        """
        dtype = [
            ("parent", np.uint32),
            ("child", np.uint32),
            ("distance", np.float32),
            ("density", np.float32),
            ("child_size", np.float32),
        ]
        result = np.empty(self._tree.parent.shape[0], dtype=dtype)
        result["parent"] = self._tree.parent
        result["child"] = self._tree.child
        result["distance"] = self._tree.distance
        result["density"] = np.exp(-self._tree.distance)
        result["child_size"] = self._tree.child_size
        return result

    def to_pandas(self):
        """
        Returns a pandas dataframe of the condensed tree.

        The columns are: parent, child, distance, density, child_size. The
        parent labelling starts at `num_points`, which represents a phantom
        root. All points connecting directly to the (multiple) tree roots have
        `num_points` as their parent. The labels for the tree roots themselves
        occur only as a parent and start from `num_points + 1`.

        Due to this construction, we cannot recover which points belong to which
        tree root (if there are multiple trees). In addition, the first parent
        value cannot be used to find `num_points`, as it can be the first
        tree-root value `num_points + 1`! All parents that do not occur as
        children should be considered a child of the phantom root.
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
                child=self._tree.child,
                distance=self._tree.distance,
                density=np.exp(-self._tree.distance),
                child_size=self._tree.child_size,
            )
        )

    def to_networkx(self):
        """Return a NetworkX DiGraph object representing the condensed tree.

        Edges have a `distance` and `density` attribute attached giving the
        distance and density at which the child node leaves the cluster.

        Nodes have a `size` attribute attached giving the number of (weighted)
        points that are in the cluster at the point of cluster creation (fewer
        points may be in the cluster at larger distance values).

        Edges connecting tree roots to the phantom root have no `distance`
        attribute, because that distance is not known. If there is a single tree
        root, the phantom root's maximum distance can be used instead.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "You must have networkx installed to export networkx graphs"
            )

        edges = [(pt, cd) for pt, cd in zip(self._tree.parent, self._tree.child)]
        g = nx.DiGraph(edges)
        nx.set_edge_attributes(
            g,
            {edge: dist for edge, dist in zip(edges, self._tree.distance)},
            "distance",
        )
        nx.set_edge_attributes(
            g,
            {edge: dens for edge, dens in zip(edges, np.exp(-self._tree.distance))},
            "density",
        )
        nx.set_node_attributes(
            g,
            {
                cd: sz
                for cd, sz in enumerate(zip(self._tree.child, self._tree.child_size))
            },
            "size",
        )
        for leaf_idx, parent in enumerate(self._leaf_tree.parent):
            if parent == 0:
                g.add_edge(self._num_points, leaf_idx + self._num_points)
        return g

    def plot(
        self,
        *,
        y: Literal["distance", "density", "ranks"] = "distance",
        leaf_separation: float = 0.8,
        cmap: str | Colormap = "viridis",
        colorbar: bool = True,
        log_size: bool = False,
        label_clusters: bool = False,
        select_clusters: bool = False,
        selection_palette: str | Colormap = "tab10",
        continuation_line_kws: dict[str, Any] | None = None,
        connect_line_kws: dict[str, Any] | None = None,
        colorbar_kws: dict[str, Any] | None = None,
        label_kws: dict[str, Any] | None = None,
    ):
        """
        Creates an icicle plot of the condensed tree.

        Parameters
        ----------
        y
            The y-axis variable to plot. Can be one of "distance", "density", or "ranks".
        leaf_separation
            A spacing parameter for icicle positioning.
        cmap
            The colormap to use for the segments.
        colorbar
            Whether to show a colorbar for the cluster size.
        log_size
            If True, the cluster sizes are plotted on a logarithmic scale.
        label_clusters
            If True, the cluster labels are plotted on the icicle segments.
        select_clusters
            If True, the segments representing selected clusters are highlighted
            with ellipses.
        selection_palette
            A list of colors to highlight selected clusters.
        continuation_line_kws
            Additional keyword arguments for the continuation lines indicating
            the continuation of root clusters.
        connect_line_kws
            Additional keyword arguments for the connecting lines between
            segments.
        colorbar_kws
            Additional keyword arguments for the colorbar.
        label_kws
            Additional keyword arguments for the cluster labels.
        """
        if y == "distance":
            distances = self._tree.distance
        elif y == "ranks":
            distances = rankdata(self._tree.distance, method="dense")
        elif y == "density":
            distances = np.exp(-self._tree.distance)
        else:
            raise ValueError(f"Unknown y value '{y}'")

        # Prepare trees
        max_size = self._leaf_tree.min_size[0]
        cluster_tree = CondensedTreeTuple(
            self._tree.parent[self._tree.cluster_rows],
            self._tree.child[self._tree.cluster_rows],
            distances[self._tree.cluster_rows].astype(
                np.float32, order="C", copy=False
            ),
            self._tree.child_size[self._tree.cluster_rows],
            np.array([], dtype=np.uint32),
        )

        # List segment info
        parents = self._leaf_tree.parent
        x_coords = self._x_coords(parents) * leaf_separation
        if y == "distance":
            death_dist = self._leaf_tree.max_distance
            birth_dist = self._leaf_tree.min_distance
        elif y == "ranks":
            death_dist = np.full(parents.shape, distances[0], dtype=np.float32)
            death_dist[cluster_tree.child - self._num_points] = cluster_tree.distance
            birth_dist = np.empty(parents.shape, dtype=np.float32)
            birth_dist[self._tree.parent - self._num_points] = distances
        elif y == "density":
            death_dist = np.exp(-self._leaf_tree.max_distance)
            birth_dist = np.exp(-self._leaf_tree.min_distance)
        order = np.argsort(self._tree.parent, kind="stable")
        if log_size:
            max_size = np.log(max_size)
            sizes = np.log(self._tree.child_size[order])
        else:
            sizes = self._tree.child_size[order]
        traces = np.split(
            np.vstack((distances[order], sizes)),
            np.flatnonzero(np.diff(self._tree.parent[order])) + 1,
            axis=1,
        )

        # Prepare the labels
        _label_kws = dict(ha="center", va="top", fontsize=8)
        if label_kws is not None:
            _label_kws.update(label_kws)

        # List cluster label for segments representing selected clusters
        if select_clusters:
            if selection_palette is None:
                ellipse_colors = ["r"]
            else:
                ellipse_colors = plt.get_cmap(selection_palette).colors

        # Process each segment
        bar = None
        ellipses = []
        connecting_lines = []
        continuation_lines = []
        # correct for cases where there are no direct phantom root child points!
        if self._tree.parent[0] != self._num_points and x_coords[1] == x_coords[0]:
            _i = 0
        else:
            _i = 1
        for segment_idx, (trace, parent_idx, segment_dist) in enumerate(
            zip(traces[_i:], parents[1:], death_dist[1:]), 1
        ):
            dist_trace, size_trace = self._prepare_trace(trace, segment_dist)
            if parent_idx == 0:
                if x_coords[0] == x_coords[segment_idx]:
                    # there is one root, plot its icicle.
                    root_dist_trace, root_size_trace = self._prepare_trace(
                        traces[0], death_dist[0]
                    )
                    bar = self._plot_icicle(
                        x_coords[segment_idx],
                        root_dist_trace,
                        root_size_trace + size_trace[0],
                        max_size,
                        plt.get_cmap(cmap),
                    )
                else:
                    # there are multiple roots, plot a continuation lines
                    continuation_lines.append(
                        [
                            (x_coords[segment_idx], dist_trace[0]),
                            (x_coords[segment_idx], segment_dist),
                        ]
                    )
            else:
                # horizontal connecting line to parent
                segment_x = x_coords[segment_idx]
                if size_trace.shape[0] > 0:
                    offset = size_trace[-1] / max_size * 0.25
                    if segment_x > x_coords[parent_idx]:
                        segment_x += offset
                    else:
                        segment_x -= offset
                connecting_lines.append(
                    [(segment_x, segment_dist), (x_coords[parent_idx], segment_dist)]
                )

                # plot the icicle
                if size_trace.shape[0] > 0:
                    bar = self._plot_icicle(
                        x_coords[segment_idx],
                        dist_trace,
                        size_trace,
                        max_size,
                        plt.get_cmap(cmap),
                    )

                # Add Ellipse for selected segments
                if (
                    label_clusters or select_clusters
                ) and segment_idx in self._chosen_segments:
                    max_dist = death_dist[segment_idx]
                    min_dist = birth_dist[segment_idx]
                    size = size_trace[0]
                    width = size / max_size
                    height = max_dist - min_dist
                    center = (x_coords[segment_idx], (min_dist + max_dist) / 2)
                    ellipse = Ellipse(
                        center, leaf_separation / 2 + width / 2, 1.4 * height
                    )
                    if label_clusters:
                        if segment_idx in self._chosen_segments:
                            plt.text(
                                x_coords[segment_idx],
                                ellipse.get_corners()[0][1],
                                len(ellipses),
                                **_label_kws,
                            )
                    if select_clusters:
                        ellipses.append(ellipse)

        # Plot the lines and ellipses
        _connect_line_kws = dict(linestyle="-", color="black", linewidth=0.5)
        if connect_line_kws is not None:
            _connect_line_kws.update(connect_line_kws)
        plt.gca().add_collection(LineCollection(connecting_lines, **_connect_line_kws))

        _continuation_line_kws = dict(linestyle=":", color="black", linewidth=1)
        if continuation_line_kws is not None:
            _continuation_line_kws.update(continuation_line_kws)
        plt.gca().add_collection(
            LineCollection(continuation_lines, **_continuation_line_kws)
        )

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
                colorbar_kws = dict()

            if "fraction" in colorbar_kws:
                bbox = plt.gca().get_window_extent()
                ax_width, ax_height = bbox.width, bbox.height
                colorbar_kws["aspect"] = ax_height / (
                    ax_width * colorbar_kws["fraction"]
                )

            plt.colorbar(
                bar,
                label=f"Cluster size {' (log)' if log_size else ''}",
                **colorbar_kws,
            )

        for side in ("right", "top", "bottom"):
            plt.gca().spines[side].set_visible(False)

        plt.xticks([])
        xlim = plt.xlim()
        plt.xlim([xlim[0] - 0.05 * xlim[1], 1.05 * xlim[1]])
        if y == "distance":
            plt.ylabel("Distance")
            plt.ylim(0, death_dist[0])
        elif y == "ranks":
            plt.ylabel("Distance rank")
            plt.ylim(0, death_dist[0])
        elif y == "density":
            plt.ylabel("Density")
            plt.ylim(1, death_dist[0])

    @classmethod
    def _plot_icicle(cls, x, dist_trace, size_trace, max_size, cmap):
        xs = np.array([[x], [x]])
        widths = xs + size_trace / max_size * np.array([[-0.25], [0.25]])
        return plt.pcolormesh(
            widths,
            np.broadcast_to(dist_trace, (2, dist_trace.shape[0])),
            np.broadcast_to(size_trace, (2, dist_trace.shape[0])),
            edgecolors="none",
            linewidth=0,
            vmin=0,
            vmax=max_size,
            cmap=cmap,
            shading="gouraud",
        )

    @classmethod
    def _prepare_trace(cls, trace, segment_dist):
        # extract distance--size traces and correct for the death distance
        size_trace = np.empty(trace.shape[1] + 1, dtype=np.float32)
        dist_trace = np.empty(trace.shape[1] + 1, dtype=np.float32)

        size_trace[:-1] = np.cumsum(trace[1, :][::-1])
        size_trace[-1] = size_trace[-2]
        dist_trace[:-1] = trace[0, :][::-1]
        dist_trace[-1] = segment_dist

        select = np.flatnonzero(np.diff(dist_trace, append=-1))
        dist_trace = dist_trace[select]
        size_trace = size_trace[select]
        return dist_trace, size_trace

    @classmethod
    def _x_coords(self, parents: np.ndarray[tuple[int], np.dtype[np.uint32]]):
        """Get the x-coordinates of the segments in the condensed tree."""
        children = dict()
        for child_idx, parent_idx in enumerate(parents[1:], 1):
            if parent_idx not in children:
                children[parent_idx] = []
            children[parent_idx].append(child_idx)

        x_coords = np.zeros(parents.shape[0])
        LeafTree._df_leaf_order(x_coords, children, 0, 0)
        return x_coords


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
        _num_points
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
                colorbar_kws = dict()

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


class PersistenceTrace(object):
    """
    A trace of the persistence of clusters in a condensed tree.
    """

    def __init__(self, trace: PersistenceTraceTuple):
        """
        Parameters
        ----------
        trace
            The total persistence trace as produced internally.
        """
        self._trace = trace

    def to_numpy(self) -> np.ndarray:
        """Returns a numpy array of the persistence trace.

        The total persistence is computed over the leaf-clusters' left-open
        (birth, death] intervals. `min_size` contains all unique birth minimum
        cluster size thresholds. It should not be confused with the
        `minimum_cluster_size` threshold, as `min_size` refers to the last value
        before a cluster becomes a leaf.
        """
        dtype = [
            ("min_size", np.float32),
            ("persistence", np.float32),
        ]
        result = np.empty(self._trace.min_size.shape[0], dtype=dtype)
        result["min_size"] = self._trace.min_size
        result["persistence"] = self._trace.persistence
        return result

    def to_pandas(self):
        """Returns a pandas dataframe representation of the persistence trace.

        The total persistence is computed over the leaf-clusters' left-open
        (birth, death] intervals. `min_size` contains all unique birth minimum
        cluster size thresholds. It should not be confused with the
        `minimum_cluster_size` threshold, as `min_size` refers to the last value
        before a cluster becomes a leaf.
        """
        try:
            from pandas import DataFrame
        except ImportError:
            raise ImportError(
                "You must have pandas installed to export pandas DataFrames"
            )

        return DataFrame(
            dict(min_size=self._trace.min_size, persistence=self._trace.persistence)
        )

    def plot(self, line_kws: dict[str, Any] | None = None):
        """
        Plots the total persistence trace.

        The x-axis shows the last minimum cluster size value before a cluster
        becomes a leaf! This matches the left-open (birth, death] interval used
        in the leaf tree and is needed to support weighted samples.

        Parameters
        ----------

        line_kws
            Additional keyword arguments for the line plot.
        """
        if line_kws is None:
            line_kws = dict()

        plt.plot(
            np.column_stack(
                (self._trace.min_size[:-1], self._trace.min_size[1:])
            ).reshape(-1),
            np.repeat(self._trace.persistence[:-1], 2),
            **line_kws,
        )
        plt.ylim([0, plt.ylim()[1]])
        plt.xlabel("Min cluster size in (birth, death]")
        plt.ylabel("Total persistence")
