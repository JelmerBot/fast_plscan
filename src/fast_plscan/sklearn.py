"""The public scikit-learn interface."""

import numpy as np
from scipy.sparse import issparse, csr_array
from scipy.signal import find_peaks
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, _check_sample_weight
from sklearn.utils.validation import validate_data
from sklearn.utils._param_validation import Interval, StrOptions, InvalidParameterError
from sklearn.neighbors import KDTree, BallTree
from numbers import Real, Integral
from typing import Any

from ._helpers import distance_matrix_to_csr, knn_to_csr, remove_self_loops
from ._api import (
    SpanningTree,
    apply_distance_cut,
    apply_size_cut,
    Labelling,
    compute_cluster_labels,
    get_max_threads,
    set_num_threads,
)
from .api import (
    compute_mutual_spanning_tree,
    extract_mutual_spanning_forest,
    clusters_from_spanning_forest,
    compute_centroids_from_features,
    compute_exemplar_indices_from_trees,
    compute_medoid_indices_from_features,
    compute_medoid_indices_from_graph,
)
from . import plots


class PLSCAN(ClusterMixin, BaseEstimator):
    """PLSCAN computes HDBSCAN* leaf-clusters with an optimal minimum cluster
    size.

    The algorithm builds a hierarchy of leaf-clusters, showing which HDBSCAN*
    [1]_ clusters are leaves as the minimum cluster size varies (filtration).
    Then, it computes the total leaf-cluster persistence per minimum cluster
    size, and picks the minimum cluster size that maximizes that score.

    The leaf-cluster hierarchy in `leaf_tree_` can be plotted as an alternative
    to HDBSCAN*'s condensed cluster tree.

    Cluster segmentations for other high-persistence minimum cluster sizes can
    be computed using the `cluster_layers` method. This method finds the
    persistence peaks and returns their cluster labels and memberships.

    References
    ----------

    .. [1] Campello, R. J., Moulavi, D., & Sander, J. (2013, April).
       Density-based clustering based on hierarchical density estimates. In
       Pacific-Asia Conference on Knowledge Discovery and Data Mining (pp.
       160-172). Springer Berlin Heidelberg.

    """

    labels_: np.ndarray[tuple[int], np.dtype[np.int64]] = None
    """Cluster label for each point, shape ``(n_samples,)``.

    Points that do not belong to any cluster are assigned label ``-1``
    (noise). Cluster labels are zero-indexed non-negative integers.
    """
    probabilities_: np.ndarray[tuple[int], np.dtype[np.float32]] = None
    """Cluster membership probability for each point, shape ``(n_samples,)``.

    Values are in ``[0, 1]``. A value of ``1.0`` indicates full membership;
    lower values indicate that the point lies in a less dense region of its
    cluster and was assigned to it by the falling-out rule. Noise points
    (``labels_ == -1``) have probability ``0.0``.
    """
    selected_clusters_: np.ndarray[tuple[int], np.dtype[np.intp]] = None
    """Leaf-tree node indices of the selected clusters, shape ``(n_clusters,)``.

    Each entry is the index into the internal leaf tree that corresponds to one
    of the chosen leaf-clusters. The length equals ``labels_.max() + 1`` unless
    all points are noise, in which case the array may be empty.
    """
    core_distances_: np.ndarray[tuple[int], np.dtype[np.float32]] = None
    """Core distance for each point, shape ``(n_samples,)``.

    The core distance of a point is its distance to its
    ``min_samples``-th nearest neighbor (including itself). It is the
    per-point bandwidth for the mutual reachability distance. Not available
    (``None``) when the input was a precomputed MST edge list.
    """
    VALID_KDTREE_METRICS = [
        "euclidean",
        "l2",
        "manhattan",
        "cityblock",
        "l1",
        "chebyshev",
        "infinity",
        "minkowski",
        "p",
    ]
    """The distance metrics implemented for use with KDTrees.

    These metrics can be used with either ``space_tree="kd_tree"`` or
    ``space_tree="ball_tree"``: ``euclidean``, ``l2``, ``manhattan``,
    ``cityblock``, ``l1``, ``chebyshev``, ``infinity``, ``minkowski``, ``p``.
    The ``p`` and ``minkowski`` names are equivalent and require
    ``metric_kws={"p": <value>}`` with ``p >= 1``.
    """
    VALID_BALLTREE_METRICS = VALID_KDTREE_METRICS + [
        "seuclidean",
        "braycurtis",
        "canberra",
        "haversine",
        "mahalanobis",
        "hamming",
        "dice",
        "jaccard",
        "russellrao",
        "rogerstanimoto",
        "sokalsneath",
    ]
    """The distance metrics implemented for use with BallTrees.

    A superset of :py:attr:`VALID_KDTREE_METRICS` that adds metrics only
    available on BallTrees: ``seuclidean``, ``braycurtis``, ``canberra``,
    ``haversine``, ``mahalanobis``, and all boolean metrics (``hamming``,
    ``dice``, ``jaccard``, ``russellrao``, ``rogerstanimoto``,
    ``sokalsneath``).
    """

    _parameter_constraints = dict(
        min_samples=[Interval(Integral, 2, None, closed="left")],
        space_tree=[StrOptions({"auto", "kd_tree", "ball_tree"})],
        metric=[StrOptions({*VALID_BALLTREE_METRICS, "precomputed"})],
        min_cluster_size=[None, Interval(Real, 2.0, None, closed="left")],
        max_cluster_size=[Interval(Real, 2.0, None, closed="right")],
        persistence_measure=[
            StrOptions({"size", "distance", "density", "size-distance", "size-density"})
        ],
        num_threads=[None, Interval(Integral, 1, None, closed="left")],
    )

    def __init__(
        self,
        *,
        min_samples: int = 5,
        space_tree: str = "auto",
        metric: str = "euclidean",
        metric_kws: dict[str, Any] | None = None,
        min_cluster_size: float | None = None,
        max_cluster_size: float = np.inf,
        persistence_measure: str = "size",
        num_threads: int | None = None,
    ):
        """
        Parameters
        ----------
        min_samples
            The number of neighbors to use for computing core distances and the
            mutual reachability distances. Higher values produce smoother
            density profiles with fewer peaks. Minimum spanning tree inputs are
            assumed to contain mutual reachability distances and ignore this
            parameter.
        space_tree
            The type of tree to use for the search. Options are "auto",
            "kd_tree" and "ball_tree". If "auto", a "kd_tree" is used if that
            supports the selected metric. Space trees are not used when `metric`
            is "precomputed".
        metric
            The distance metric to use. See
            :py:attr:`.PLSCAN.VALID_KDTREE_METRICS` and
            :py:attr:`.PLSCAN.VALID_BALLTREE_METRICS` for available options. Use
            "precomputed" if the input to `.fit()` contains distances. See
            sklearn documentation for metric definitions.
        metric_kws
            Additional keyword arguments for the distance metric. For example,
            `p` for the Minkowski distance.
        min_cluster_size
            The minimum size limit for clusters, defaults to the value of
            min_samples. Values below min_samples are not allowed, as the
            leaf-clusters produced by those values can be incomplete and
            arbitrary.
        max_cluster_size
            The maximum size limit for clusters, by default np.inf.
        persistence_measure
            Selects a persistence measure. Valid options are "size", "distance",
            "density", "size-distance", and "size-density". The "size",
            "distance", and "density" options compute persistence as the range
            of size/distance/density values for which clusters are leaves. The
            "size-distance" and "size-density" options compute bi-persistence as
            the distance/density -- minimum cluster size areas for which clusters
            are leaves. Density is computed as exp(-dist).
        num_threads
            The number of threads to use for parallel computations, value must
            be positive. If None, OpenMP's default maximum thread count is used.
        """
        self.min_samples = min_samples
        self.space_tree = space_tree
        self.metric = metric
        self.metric_kws = metric_kws
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.persistence_measure = persistence_measure
        self.num_threads = num_threads

    def fit(
        self,
        X: np.ndarray[tuple[int, ...]] | tuple | csr_array,
        y: None = None,
        *,
        sample_weights: np.ndarray[tuple[int], np.dtype[np.float32]] | None = None,
        **fit_params,
    ):
        """
        Computes PLSCAN clusters and hierarchies for the input data. Several
        inputs are supported, including feature vectors, precomputed sorted
        (partial) minimums spanning trees, dense or sparse distance matrices,
        and k-nearest neighbors graphs.

        The input data does not have to form a single connected component, and
        the algorithm will select the minimum cluster size that maximizes the
        total persistence over all components. The components themselves are
        never selected as clusters.

        Parameters
        ----------
        X
            The input data. If `metric` is not set to "precomputed", the X must
            be a 2D array of shape (num_points, num_features). Missing values
            are not supported.

            If `metric` is set to "precomputed", the input is a (sparse)
            distance matrix in one of the following formats:

            1. tuple of (edges, num_points)
                A minimum spanning tree where `edges` is a 2D array in the
                format (parent, child, distance) and `num_points` is the number
                of points in the input data. There should be at most `num_points
                - 1` edges. Edges must be sorted by distance.
            2. tuple of (distances, indices)
                A k-nearest neighbors graph where `distances` is a 2D array of
                distances and `indices` is a 2D array of child indices. Rows
                must be sorted by distance. Negative indices indicate missing
                edges and must occur after all valid edges in their row.
            3. np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
                A condensed or full square distance matrix. The diagonal is
                filled with zeros before processing.
            4. csr_array:
                A sparse distance matrix in CSR format. Self-loops and explicit
                zeros are removed before processing.

            In all cases, distance values should be non-negative. In cases 2
            through 4, each point should have `min_samples` neighbors. Infinite
            distances, either as input or as a result of too few neighbors, may
            break plots and the bi-persistence computation.
        y
            Ignored, present for compatibility with scikit-learn.
        sample_weights
            Sample weights for the points in the sorted minimum spanning tree.
            If None, all samples are considered equally weighted.
        **fit_params
            Unused additional parameters for compatibility with scikit-learn.

        Returns
        -------
        self
            The fitted PLSCAN instance.
        """
        # Validate parameters
        self._validate_params()
        if self.min_cluster_size is None:
            min_cluster_size = self.min_samples
        else:
            min_cluster_size = self.min_cluster_size
            if min_cluster_size < self.min_samples:
                raise InvalidParameterError(
                    "Minimum cluster size must be at least equal to "
                    f"min_samples ({self.min_samples})."
                )
        if self.max_cluster_size <= min_cluster_size:
            raise InvalidParameterError(
                "Maximum cluster size must be greater than the minimum cluster size."
            )
        if self.metric in ["minkowski", "p"]:
            if self.metric_kws is None or "p" not in self.metric_kws:
                raise InvalidParameterError(
                    "Minkowski distance requires a `metric_kws` 'p' parameter."
                )
            if self.metric_kws["p"] < 1:
                raise InvalidParameterError(
                    "Minkowski distance requires a `metric_kws` 'p' parameter >= 1."
                )
        elif self.metric not in ["seuclidean", "mahalanobis"]:
            if self.metric_kws is not None and len(self.metric_kws) > 0:
                raise InvalidParameterError(
                    "Metric keyword arguments are only supported for the "
                    "'minkowski', 'seuclidean', and 'mahalanobis' metrics. "
                    f"Got `metric_kws` for metric '{self.metric}' instead."
                )

        if self.metric != "precomputed":
            if self.space_tree == "auto":
                space_tree = (
                    "kd_tree" if self.metric in KDTree.valid_metrics else "ball_tree"
                )
            else:
                space_tree = self.space_tree
                tree = KDTree if space_tree == "kd_tree" else BallTree
                if self.metric not in tree.valid_metrics:
                    raise InvalidParameterError(
                        f"Invalid metric '{self.metric}' for {space_tree}"
                    )

        if self.num_threads is not None:
            set_num_threads(self.num_threads)

        # Validate inputs
        if self.metric != "precomputed":
            X = validate_data(
                self,
                X,
                y=None,
                dtype=np.float32,
                ensure_min_samples=self.min_samples + 1,
            )
            self._X = X
            self._num_points = self._X.shape[0]
        else:
            self._X = None
            X, self._num_points, is_sorted, is_mst = self._check_input(X)

        if sample_weights is not None:
            sample_weights = _check_sample_weight(
                sample_weights,
                csr_array((self._num_points, self._num_points)),
                dtype=np.float32,
                ensure_non_negative=True,
            )
            if sample_weights.max() > min_cluster_size:
                raise ValueError(
                    "Sample weights must not exceed the minimum cluster size."
                )

        # Compute / extract MST
        if self.metric != "precomputed":
            self._mutual_graph = None
            self._minimum_spanning_tree, self._neighbors, self.core_distances_ = (
                compute_mutual_spanning_tree(
                    X,
                    space_tree=space_tree,
                    min_samples=self.min_samples,
                    metric=self.metric,
                    metric_kws=self.metric_kws,
                )
            )
        elif is_mst:
            self.core_distances_ = None
            self._mutual_graph = None
            self._neighbors = None
            self._minimum_spanning_tree = SpanningTree(
                X[:, 0].astype(np.uint32, copy=False),
                X[:, 1].astype(np.uint32, copy=False),
                X[:, 2].astype(np.float32, copy=False),
            )
        else:
            self._neighbors = None
            (
                self._minimum_spanning_tree,
                self._mutual_graph,
                self.core_distances_,
            ) = extract_mutual_spanning_forest(
                X, min_samples=self.min_samples, is_sorted=is_sorted
            )

        # Compute clusters from MST
        (
            (self.labels_, self.probabilities_),
            self.selected_clusters_,
            self._persistence_trace,
            self._leaf_tree,
            self._condensed_tree,
            self._linkage_tree,
        ) = clusters_from_spanning_forest(
            self._minimum_spanning_tree,
            self._num_points,
            sample_weights=sample_weights,
            min_cluster_size=min_cluster_size,
            max_cluster_size=self.max_cluster_size,
            persistence_measure=self.persistence_measure,
        )

        # Reset the number of threads back to the default
        if self.num_threads is not None:
            set_num_threads(get_max_threads())
        return self

    @property
    def persistence_trace_(self) -> plots.PersistenceTrace:
        """The total persistence signal over all minimum cluster sizes.

        Each point on the trace records the summed persistence of all
        leaf-clusters that are alive at a given minimum cluster size value.
        The optimal minimum cluster size — used by :py:meth:`fit` — is the
        one that maximises this signal.

        For standard persistence measures (``"size"``, ``"distance"``,
        ``"density"``), the trace is 1-D (one value per minimum cluster size).
        For bi-persistence measures (``"size-distance"``,
        ``"size-density"``), the trace integrates over the secondary axis,
        still yielding a 1-D signal. The object supports
        :py:meth:`~fast_plscan.plots.PersistenceTrace.plot`,
        :py:meth:`~fast_plscan.plots.PersistenceTrace.to_numpy`, and
        :py:meth:`~fast_plscan.plots.PersistenceTrace.to_pandas`.
        """
        check_is_fitted(self, "_persistence_trace")
        return plots.PersistenceTrace(self._persistence_trace)

    @property
    def leaf_tree_(self) -> plots.LeafTree:
        """The leaf-cluster hierarchy across all minimum cluster sizes.

        The leaf tree tracks which HDBSCAN* condensed-tree segments are
        *leaves* (i.e. have no child clusters) as the minimum cluster size
        parameter varies. It is an alternative visualisation to
        :py:attr:`condensed_tree_` that makes the PLSCAN filtration explicit.

        Selected clusters (those corresponding to the fitted
        ``min_cluster_size``) are highlighted when plotted. The object
        supports
        :py:meth:`~fast_plscan.plots.LeafTree.plot`,
        :py:meth:`~fast_plscan.plots.LeafTree.to_networkx`,
        :py:meth:`~fast_plscan.plots.LeafTree.to_numpy`, and
        :py:meth:`~fast_plscan.plots.LeafTree.to_pandas`.
        """
        check_is_fitted(self, ("_leaf_tree"))
        return plots.LeafTree(
            self._leaf_tree,
            self._condensed_tree,
            self.selected_clusters_,
            self._persistence_trace,
            self._num_points,
        )

    @property
    def condensed_tree_(self) -> plots.CondensedTree:
        """The HDBSCAN* condensed cluster tree.

        The condensed tree is built by collapsing the full single-linkage
        dendrogram: only splits where at least one child has at least
        ``min_samples`` members are retained. Each remaining segment
        represents a cluster that persists across a range of distance
        (epsilon) values.

        Selected clusters (the fitted leaf-cluster segmentation) are
        highlighted when plotted. The object supports
        :py:meth:`~fast_plscan.plots.CondensedTree.plot`,
        :py:meth:`~fast_plscan.plots.CondensedTree.to_networkx`,
        :py:meth:`~fast_plscan.plots.CondensedTree.to_numpy`, and
        :py:meth:`~fast_plscan.plots.CondensedTree.to_pandas`.
        """
        check_is_fitted(self, ("_condensed_tree"))
        return plots.CondensedTree(
            self._leaf_tree,
            self._condensed_tree,
            self.selected_clusters_,
            self._num_points,
        )

    @property
    def single_linkage_tree_(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """The full single-linkage dendrogram in scipy linkage format.

        A float64 array of shape ``(n_samples - 1, 4)`` (or fewer rows when
        the input contains multiple connected components). Columns are:

        - ``0`` — parent node index
        - ``1`` — child node index (leaf indices are in ``[0, n_samples)``)
        - ``2`` — merge distance (mutual reachability distance)
        - ``3`` — size of the child subtree

        This format is compatible with :py:func:`scipy.cluster.hierarchy.dendrogram`
        and related scipy functions.
        """
        check_is_fitted(self, ("_linkage_tree"))
        return np.column_stack(
            (
                self._linkage_tree.parent,
                self._linkage_tree.child,
                self._minimum_spanning_tree.distance,
                self._linkage_tree.child_size,
            )
        )

    @property
    def minimum_spanning_tree_(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """The mutual reachability minimum spanning tree (or forest).

        A float64 array of shape ``(n_edges, 3)`` with columns:

        - ``0`` — parent point index
        - ``1`` — child point index
        - ``2`` — mutual reachability edge weight

        Edges are sorted by distance in ascending order. When the input data
        forms multiple connected components (e.g. a sparse distance graph with
        disconnected subgraphs), the result is a spanning *forest* with fewer
        than ``n_samples - 1`` edges.
        """
        check_is_fitted(self, "_minimum_spanning_tree")
        return np.column_stack(tuple(self._minimum_spanning_tree))

    def compute_centroids(
        self,
        labels: np.ndarray[tuple[int], np.dtype[np.int64]] | None = None,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        """Return the probability-weighted centroid of each cluster.

        For each cluster the centroid is the membership-probability-weighted
        mean of the feature vectors of all cluster members (noise points with
        label ``-1`` are excluded). The result has shape
        ``(n_clusters, n_features)``.

        Only available for feature-vector input (``metric != "precomputed"``).
        Raises :py:exc:`ValueError` for precomputed inputs and
        :py:exc:`~sklearn.exceptions.NotFittedError` before fitting.

        Parameters
        ----------
        labels
            An optional integer array of shape ``(n_samples,)`` with cluster
            labels. When ``None`` (default), the fitted ``labels_`` are used.

        Returns
        -------
        centroids
            Float32 array of shape ``(n_clusters, n_features)``.
        """
        check_is_fitted(self, "_minimum_spanning_tree")
        if self._X is None:
            raise ValueError(
                "compute_centroids is only available for feature-vector input "
                "(metric != 'precomputed')."
            )
        if labels is None:
            labels = self.labels_
        return compute_centroids_from_features(self._X, self.probabilities_, labels)

    def compute_medoid_indices(
        self,
        labels: np.ndarray[tuple[int], np.dtype[np.int64]] | None = None,
    ) -> np.ndarray[tuple[int], np.dtype[np.intp]]:
        """Return the index of the medoid point for each cluster.

        For each cluster the medoid is the cluster member whose
        probability-weighted sum of pairwise within-cluster distances is
        smallest — the point ``i*`` in cluster ``c`` that minimises:

        .. math::

            i^* = \\operatorname{arg\\,min}_{i \\in c}
                  \\sum_{j \\in c} p_j \\cdot d(x_i, x_j)

        where :math:`p_j` is the cluster-membership probability of point
        :math:`j` and :math:`d` is the fitted distance metric.

        For feature-vector inputs, exact pairwise distances are used. For
        precomputed sparse distance inputs, the average mutual reachability
        distances are used to compensate for variations in the sparsity of the
        input graph.

        Returns one index per cluster into the original input, making it simple
        to retrieve any attribute of the medoid point.

        Only available for feature-vector or (sparse) precomputed distance
        inputs. Raises :py:exc:`ValueError` for precomputed MST inputs and
        :py:exc:`~sklearn.exceptions.NotFittedError` before fitting.

        Parameters
        ----------
        labels
            An optional integer array of shape ``(n_samples,)`` with cluster
            labels. When ``None`` (default), the fitted ``labels_`` are used.

        Returns
        -------
        medoid_indices
            Integer array of shape ``(n_clusters,)`` with the index of the
            medoid point for each cluster.
        """
        check_is_fitted(self, "_minimum_spanning_tree")
        if self._X is None and self._mutual_graph is None:
            raise ValueError(
                "compute_medoid_indices is not available for precomputed MST input. "
                "Provide a distance matrix or kNN graph instead."
            )
        if labels is None:
            labels = self.labels_

        if self._X is not None:
            medoid_indices = compute_medoid_indices_from_features(
                self._X,
                self.core_distances_,
                self.probabilities_,
                labels,
                metric=self.metric,
                metric_kws=self.metric_kws,
            )
        else:
            medoid_indices = compute_medoid_indices_from_graph(
                self._mutual_graph,
                self.probabilities_,
                labels,
            )

        return medoid_indices

    def compute_exemplar_indices(
        self,
        labels: np.ndarray[tuple[int], np.dtype[np.int64]] | None = None,
    ) -> list[np.ndarray]:
        """Return the exemplar point indices for each cluster.

        For each leaf-cluster segment, exemplars are the points whose dropout
        distance in the condensed tree equals the segment's minimum distance --
        the densest points within that segment.

        Parameters
        ----------
        labels
            An optional integer array of shape ``(n_samples,)`` with cluster
            labels for each point. When ``None`` (default), the fitted
            ``labels_`` are used. Labels must be dense and zero-indexed
            (cluster labels are ``0, 1, ..., n_clusters - 1``), with ``-1``
            for noise points.

        Returns
        -------
        exemplars_per_cluster
            A list of length ``n_clusters``. Each entry is an integer array
            of point indices that are exemplars for the corresponding cluster.
            Returns an empty list when no clusters exist (all noise).

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        ValueError
            If the shape of ``labels`` does not match the number of samples.
        """
        check_is_fitted(self, "_leaf_tree")
        if labels is None:
            labels = self.labels_
        if len(labels) != self._num_points:
            raise ValueError("labels must match the number of samples")

        n_unique = int(np.unique(labels[labels >= 0]).size)
        n_clusters = int(labels.max()) + 1
        if n_clusters <= 0:
            return []
        if n_unique != n_clusters:
            raise ValueError(
                "labels must be dense: cluster indices must be contiguous from 0."
            )
        return compute_exemplar_indices_from_trees(
            self._leaf_tree,
            self._condensed_tree,
            labels,
            self._num_points,
        )

    def cluster_layers(
        self,
        max_peaks: int | None = None,
        min_size: float | None = None,
        max_size: float | None = None,
        height: float = 0.0,
        threshold: float = 0.0,
        **kwargs,
    ) -> list[tuple[np.float32, Labelling]]:
        """Return cluster labels and probabilities at each persistence peak.

        The persistence trace (:py:attr:`persistence_trace_`) may have multiple
        local maxima, each corresponding to a stable minimum cluster size at
        which a distinct set of leaf-clusters is well-separated. This method
        detects those peaks via :py:func:`scipy.signal.find_peaks` and returns
        cluster segmentations for each one, ordered from lowest to highest
        minimum cluster size.

        Parameters
        ----------
        max_peaks
            Maximum number of peaks to return. If ``None``, all detected peaks
            are returned. When specified, only the ``max_peaks`` peaks with the
            highest persistence are kept. Applied after all other filters.
        min_size
            Discard peaks whose minimum cluster size is below this value.
        max_size
            Discard peaks whose minimum cluster size is above this value.
        height
            Minimum persistence height a peak must exceed to be included,
            default ``0.0``. Equivalent to the ``height`` parameter of
            :py:func:`scipy.signal.find_peaks`.
        threshold
            Minimum required persistence drop on at least one side of the peak,
            default ``0.0``. Equivalent to the ``threshold`` parameter of
            :py:func:`scipy.signal.find_peaks`.
        **kwargs
            Additional keyword arguments forwarded directly to
            :py:func:`scipy.signal.find_peaks`. Note that the persistence
            signal is sampled at irregularly spaced minimum cluster size values,
            so sample-distance parameters (e.g. ``distance``) do not correspond
            to a uniform spacing.

        Returns
        -------
        list of (min_cluster_size, labels, probabilities)
            One entry per detected peak. Each entry is a 3-tuple of:

            - ``min_cluster_size`` — the :py:class:`numpy.float32` minimum
              cluster size at the peak.
            - ``labels`` — int64 array of shape ``(n_samples,)`` giving the
              cluster label for each point (``-1`` for noise).
            - ``probabilities`` — float32 array of shape ``(n_samples,)`` with
              cluster membership probabilities.

            Returns an empty list when no peaks are found.
        """
        check_is_fitted(self, "_persistence_trace")
        # Pad persistence with zero so maxima at the edges can be detected as peaks
        x, y = self._persistence_trace
        zero = np.array([0], dtype=y.dtype)
        signal = np.hstack((zero, y, zero))
        peaks = find_peaks(signal, height=height, threshold=threshold, **kwargs)[0] - 1

        if min_size is not None:
            peaks = peaks[x[peaks] >= min_size]
        if max_size is not None:
            peaks = peaks[x[peaks] <= max_size]
        if max_peaks is not None and len(peaks) > 0:
            peak_idx = -min(max_peaks, len(peaks))
            limit = np.partition(y[peaks], peak_idx)[peak_idx]
            peaks = peaks[y[peaks] >= limit]
        return [(x[peak], *self.min_cluster_size_cut(x[peak])) for peak in peaks]

    def distance_cut(self, epsilon: float) -> Labelling:
        """Return a DBSCAN*-style clustering at a fixed distance threshold.

        Selects all leaf-clusters whose birth distance is at most ``epsilon``
        and returns labels and membership probabilities for those clusters.
        This is equivalent to running DBSCAN* with ``eps = epsilon`` and
        ``min_samples`` equal to the fitted value: points that fall outside
        every selected cluster are labelled as noise (``-1``).

        Parameters
        ----------
        epsilon
            Distance threshold. Only leaf-clusters with birth distance
            ``≤ epsilon`` are selected. Use ``epsilon = 0`` to select no
            clusters (all noise) and ``epsilon = np.inf`` to select all
            leaf-clusters.

        Returns
        -------
        labels
            int64 array of shape ``(n_samples,)``. Cluster indices are
            zero-based; noise points are ``-1``.
        probabilities
            float32 array of shape ``(n_samples,)`` with cluster membership
            probabilities in ``[0, 1]``.
        """
        check_is_fitted(self, "_leaf_tree")
        selected_clusters = apply_distance_cut(self._leaf_tree, epsilon)
        return compute_cluster_labels(
            self._leaf_tree, self._condensed_tree, selected_clusters, self._num_points
        )

    def min_cluster_size_cut(self, cut_size: float) -> Labelling:
        """Return the clustering produced by a specific minimum cluster size.

        Selects all leaf-clusters that are alive at ``cut_size`` in the
        left-open interval ``(birth, death]``, i.e. clusters whose birth size
        is strictly less than ``cut_size`` and whose death size is at least
        ``cut_size``. This is the same selection rule used internally by
        :py:meth:`fit` for the automatically chosen minimum cluster size.

        Use :py:attr:`persistence_trace_` to identify candidate cut sizes, or
        use :py:meth:`cluster_layers` to obtain clusterings for all persistence
        peaks at once.

        Parameters
        ----------
        cut_size
            Minimum cluster size threshold. Must be ``≥ 2.0``.

        Returns
        -------
        labels
            int64 array of shape ``(n_samples,)``. Cluster indices are
            zero-based; noise points are ``-1``.
        probabilities
            float32 array of shape ``(n_samples,)`` with cluster membership
            probabilities in ``[0, 1]``.
        """
        check_is_fitted(self, "_leaf_tree")
        selected_clusters = apply_size_cut(self._leaf_tree, cut_size)
        return compute_cluster_labels(
            self._leaf_tree, self._condensed_tree, selected_clusters, self._num_points
        )

    def _check_input(self, X):
        """Checks and converts the input to a CSR sparse matrix."""
        # Check kNN / MST inputs
        if isinstance(X, tuple):
            if len(X) < 2:
                raise ValueError(
                    "Input tuple must have at least 2 elements, "
                    f"got {len(X)} instead."
                )
            if isinstance(X[1], np.ndarray):
                return knn_to_csr(*self._check_knn(X)), X[0].shape[0], True, False
            else:
                edges, num_points = self._check_mst(X)
                return edges, num_points, True, True

        # Check distance matrix input
        X = check_array(
            X,
            accept_sparse="csr",
            ensure_2d=False,
            ensure_non_negative=True,
            ensure_all_finite=False,
            ensure_min_samples=self.min_samples + 1,
            input_name="X",
        )

        # Check input is square
        copy = True
        if X.ndim == 1:
            X = squareform(X)
            copy = False
        elif X.shape[0] != X.shape[1]:
            raise ValueError(
                "Distance matrix must be square, got shape " f"{X.shape} instead."
            )

        # Convert to valid CSR format
        if issparse(X):
            X = remove_self_loops(X)
        else:
            X = distance_matrix_to_csr(X, copy=copy)
        return X, X.shape[0], False, False

    def _check_knn(self, X):
        """Checks if a kNN graph is valid."""
        if len(X) != 2:
            raise ValueError(
                "kNN input must be a tuple of (distances, indices), "
                f"got {len(X)} elements instead."
            )
        distances, indices = X
        if distances.shape != indices.shape:
            raise ValueError(
                "kNN distances and indices must have the same shape, "
                f"got {distances.shape} and {indices.shape}."
            )
        distances = check_array(
            distances,
            ensure_non_negative=True,
            ensure_all_finite=False,
            ensure_min_features=self.min_samples + 1,
            ensure_min_samples=self.min_samples + 1,
            input_name="kNN distances",
        )
        indices = check_array(
            indices,
            ensure_all_finite=True,
            ensure_min_features=self.min_samples + 1,
            ensure_min_samples=self.min_samples + 1,
            input_name="kNN indices",
        )
        return distances, indices

    def _check_mst(self, X):
        if len(X) != 2:
            raise ValueError(
                "MST input must be a tuple of (edges, num_points), "
                f"got {len(X)} elements instead."
            )
        edges, num_points = X
        if num_points < self.min_samples + 1:
            raise ValueError(
                f"Number of points in MST must be at least {self.min_samples + 1}, "
                f"got {num_points} instead."
            )
        edges = check_array(edges, ensure_non_negative=True, input_name="MST edges")
        if edges.shape[1] != 3:
            raise ValueError(
                "MST edges must have shape (n_edges, 3), " f"got {edges.shape} instead."
            )
        if edges.shape[0] > num_points - 1:
            raise ValueError(
                "MST edges must not contain more than n_points - 1 edges, "
                f"got {edges.shape[0]} edges for {num_points} points."
            )
        return edges, num_points
