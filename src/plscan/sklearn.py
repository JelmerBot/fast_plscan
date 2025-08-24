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
from typing import Self, Any

from .api import (
    compute_mutual_spanning_tree,
    extract_mutual_spanning_forest,
    clusters_from_spanning_forest,
)
from ._lib import (
    Labelling,
    SpanningTree,
    apply_distance_cut,
    apply_size_cut,
    compute_cluster_labels,
    distance_matrix_to_csr,
    knn_to_csr,
    remove_self_loops,
    get_max_threads,
    set_num_threads
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
    """The computed cluster labels."""
    probabilities_: np.ndarray[tuple[int], np.dtype[np.float32]] = None
    """The computed cluster membership probabilities."""
    selected_clusters_: np.ndarray[tuple[int], np.dtype[np.intp]] = None
    """The computed leaf tree indices of the selected clusters."""
    core_distances_: np.ndarray[tuple[int], np.dtype[np.float32]] = None
    """The computed core distances. These are the distances to the
    `min_samples`-th nearest neighbor."""
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
    """The distance metrics implemented for use with KDTrees."""
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
    "The distance metrics implemented for use with BallTrees."

    _parameter_constraints = dict(
        min_samples=[Interval(Integral, 2, None, closed="left")],
        space_tree=[StrOptions({"auto", "kd_tree", "ball_tree"})],
        metric=[StrOptions({*VALID_BALLTREE_METRICS, "precomputed"})],
        min_cluster_size=[None, Interval(Real, 2.0, None, closed="left")],
        max_cluster_size=[Interval(Real, 2.0, None, closed="right")],
        use_bi_persistence=["boolean"],
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
        use_bi_persistence: bool = False,
        num_threads: int | None = None,
    ):
        """
        Parameters
        ----------
        min_samples
            The number of neighbors to use for computing core distances and the
            mutual reachability distances. Higher values produce smoother density
            profiles with fewer peaks. Minimum spanning tree inputs are assumed to
            contain mutual reachability distances and ignore this parameter.
        space_tree
            The type of tree to use for the search. Options are "auto", "kd_tree"
            and "ball_tree". If "auto", a "kd_tree" is used if that supports the
            selected metric. Space trees are not used when `metric` is
            "precomputed".
        metric
            The distance metric to use. See :py:attr:`.PLSCAN.VALID_KDTREE_METRICS`
            and :py:attr:`.PLSCAN.VALID_BALLTREE_METRICS` for available options. Use
            "precomputed" if the input to `.fit()` contains distances. See sklearn
            documentation for metric definitions.
        metric_kws
            Additional keyword arguments for the distance metric. For example, `p`
            for the Minkowski distance.
        min_cluster_size
            The minimum size limit for clusters, defaults to the value of
            min_samples. Values below min_samples are not allowed, as the
            leaf-clusters produced by those values can be incomplete and arbitrary.
        max_cluster_size
            The maximum size limit for clusters, by default np.inf.
        use_bi_persistence
            Whether to use total bi-persistence or total size-persistence for
            selecting the optimal minimum cluster size.
        num_threads
            The number of threads to use for parallel computations, value must be
            positive. If None, OpenMP's default maximum thread count is used.
        """
        self.min_samples = min_samples
        self.space_tree = space_tree
        self.metric = metric
        self.metric_kws = metric_kws
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.use_bi_persistence = use_bi_persistence
        self.num_threads = num_threads

    def fit(
        self: Self,
        X: np.ndarray[tuple[int, ...]] | tuple | csr_array,
        y: None = None,
        *,
        sample_weights: np.ndarray[tuple[int], np.dtype[np.float32]] | None = None,
        **fit_params,
    ) -> Self:
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
        else:
            if self.metric_kws is not None and len(self.metric_kws) > 0:
                raise InvalidParameterError(
                    "Metric keyword arguments are only supported for Minkowski "
                    "distance. Got `metric_kws` for metric "
                    f"{self.metric} instead."
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
                ensure_min_samples=self.min_samples,
            )
            self._num_points = X.shape[0]
        else:
            X, self._num_points, is_sorted, is_mst = self._check_input(X)

        if sample_weights is not None:
            sample_weights = _check_sample_weight(
                sample_weights,
                csr_array((self._num_points, self._num_points)),
                dtype=np.float32,
                ensure_non_negative=True,
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
            use_bi_persistence=self.use_bi_persistence,
        )

        # Reset the number of threads back to the default
        if self.num_threads is not None:
            set_num_threads(get_max_threads())
        return self

    @property
    def persistence_trace_(self: Self) -> plots.PersistenceTrace:
        """
        A trace of the total (bi-)persistence per minimum cluster size. sizes
        represent births in (birth, death] intervals.
        """
        check_is_fitted(self, "_persistence_trace")
        return plots.PersistenceTrace(self._persistence_trace)

    @property
    def leaf_tree_(self: Self) -> plots.LeafTree:
        """
        The minimum cluster size leaf-cluster tree showing which condensed tree
        segments are leaves at each minimum cluster size value. The object has
        as plotting function and conversion methods for networkx, pandas, and
        numpy.
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
    def condensed_tree_(self: Self) -> plots.CondensedTree:
        """
        The condensed cluster tree showing which distance-contour clusters exist
        in the data. The object has as plotting function and conversion methods
        for networkx, pandas, and numpy.
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
        self: Self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """
        A single linkage dendrogram in scipy format. The first column represents
        the link's parent, the second column represents the link's child, and
        the third column represents the link's distance.
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
        self: Self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """
        A minimum spanning tree in scipy format. The first column represents the
        edge's parent, the second column represents the edge's child, and the
        third column represents the edge's distance. May be a spanning forest if
        the input contained multiple connected components.
        """
        check_is_fitted(self, "_minimum_spanning_tree")
        return np.column_stack(tuple(self._minimum_spanning_tree))

    def cluster_layers(
        self: Self,
        n_peaks: int | None = None,
        min_size: float | None = None,
        max_size: float | None = None,
        height: float = 0.0,
        threshold: float = 0.0,
        **kwargs,
    ) -> list[tuple[np.float32, Labelling]]:
        """
        Computes cluster labels and membership probabilities for the peaks in
        the persistence trace.

        Parameters
        ----------
        n_peaks
            The number of peaks to return. If None, all peaks are returned. If
            specified, the n_peaks most persistent peaks are returned. The
            selection is performed after all other thresholds.
        min_size
            The minimum cluster size to consider for the cluster layers. If
            None, all clusters are considered.
        max_size
            The maximum cluster size to consider for the cluster layers. If
            None, all clusters are considered.
        height
            Suppress peak with a persistence below this value, default 0.0.
        threshold
            Suppress peak with a persistence change below this value, default
            0.0.
        **kwargs
            Additional parameters for the `scipy.signal.find_peaks` function.
            Note that the persistence signal is defined on irregularly spaced
            minimum cluster size values. So the parameters relating to the
            distance between peaks in samples (e.g., `distance`) do not provide
            a uniform meaning.

        Returns
        -------
        peaks
            Cluster labels and membership probabilities for the detected peaks.
            Each item contains the minimum cluster size, cluster labels, and
            membership probabilities for the corresponding peak.

        """
        check_is_fitted(self, "_persistence_trace")
        x, y = self._persistence_trace
        peaks = find_peaks(y, height=height, threshold=threshold, **kwargs)[0]

        if min_size is not None:
            peaks = peaks[x[peaks] >= min_size]
        if max_size is not None:
            peaks = peaks[x[peaks] <= max_size]
        if n_peaks is not None:
            peak_idx = -min(n_peaks, len(peaks))
            limit = np.partition(y[peaks], peak_idx)[peak_idx]
            peaks = peaks[y[peaks] >= limit]
        return [(x[peak], *self.min_cluster_size_cut(x[peak])) for peak in peaks]

    def distance_cut(self: Self, epsilon: float) -> Labelling:
        """
        Computes (DBSCAN*-like) cluster labels and membership probabilities at
        the given distance threshold (epsilon).

        Parameters
        ----------
        birth_size
            The birth size threshold for the cluster labels and membership
            probabilities.

        Returns
        -------
        labelling
            Effectively a tuple of cluster labels and membership probability
            vectors.
        """
        check_is_fitted(self, "_leaf_tree")
        selected_clusters = apply_distance_cut(self._leaf_tree, epsilon)
        return compute_cluster_labels(
            self._leaf_tree, self._condensed_tree, selected_clusters, self._num_points
        )

    def min_cluster_size_cut(self: Self, cut_size: float) -> Labelling:
        """
        Computes cluster labels and membership probabilities at the given cut
        size threshold (cut_size) in a left-open (birth, death] size interval.

        Parameters
        ----------
        cut_size
            The birth size threshold for the cluster labels and membership
            probabilities.

        Returns
        -------
        labelling
            Effectively a tuple of cluster labels and membership probability
            vectors.
        """
        check_is_fitted(self, "_leaf_tree")
        selected_clusters = apply_size_cut(self._leaf_tree, cut_size)
        return compute_cluster_labels(
            self._leaf_tree, self._condensed_tree, selected_clusters, self._num_points
        )

    def _check_input(self: Self, X):
        """Checks and converts the input to a CSR sparse matrix."""
        # Check kNN / MST inputs
        if isinstance(X, tuple):
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

    def _check_knn(self: Self, X):
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

    def _check_mst(self: Self, X):
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
