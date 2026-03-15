"""Public API for plotting and exporting condensed trees, leaf trees, and
persistence traces."""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import rankdata
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Ellipse
from matplotlib.colors import Colormap
from typing import Any, Literal

from .leaf_tree import LeafTree
from .._api import LeafTree as LeafTreeTuple, CondensedTree as CondensedTreeTuple


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
