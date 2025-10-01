#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <omp.h>

#include <array>

#include "array.h"
#include "condensed_tree.h"
#include "distances.h"
#include "labelling.h"
#include "leaf_tree.h"
#include "linkage_tree.h"
#include "persistence_trace.h"
#include "space_tree.h"
#include "spanning_tree.h"
#include "sparse_graph.h"

namespace nb = nanobind;

// --- Implementation details

namespace {

// Helper functions

template <Metric metric>
nb::object wrap_dist(nb::dict const metric_kws) {
  return nb::cpp_function([dist = get_dist<metric>(metric_kws)](
                              array_ref<float const> const x,
                              array_ref<float const> const y
                          ) { return dist(to_view(x), to_view(y)); });
}

// Public bindings

void add_condensed_bindings(nb::module_ &m) {
  nb::class_<CondensedTree>(m, "CondensedTree")
      .def(
          "__init__",
          [](CondensedTree *t, nb::handle parent, nb::handle child,
             nb::handle distance, nb::handle child_size,
             nb::handle cluster_rows) {
            // Support np.memmap and np.ndarray input types for sklearn
            // pickling. The output of np.asarray can cast to nanobind ndarrays.
            auto const asarray = nb::module_::import_("numpy").attr("asarray");
            new (t) CondensedTree(
                nb::cast<array_ref<uint32_t const>>(asarray(parent), false),
                nb::cast<array_ref<uint32_t const>>(asarray(child), false),
                nb::cast<array_ref<float const>>(asarray(distance), false),
                nb::cast<array_ref<float const>>(asarray(child_size), false),
                nb::cast<array_ref<uint32_t const>>(
                    asarray(cluster_rows), false
                )
            );
          },
          nb::arg("parent"), nb::arg("child"), nb::arg("distance"),
          nb::arg("child_size"), nb::arg("cluster_rows"),
          R"(
            Parameters
            ----------
            parent
                An array of parent cluster indices. Clusters are labelled
                with indices starting from the number of points.
            child
                An array of child node and cluster indices. Clusters are labelled
                with indices starting from the number of points.
            distance
                The distance at which the child side connects to the parent side.
            child_size
                The (weighted) size in the child side of the link.
            cluster_rows
                The row indices with a cluster as child.
          )"
      )
      .def_ro(
          "parent", &CondensedTree::parent, nb::rv_policy::reference,
          "A 1D array of parent cluster indices."
      )
      .def_ro(
          "child", &CondensedTree::child, nb::rv_policy::reference,
          "A 1D array of child cluster indices."
      )
      .def_ro(
          "distance", &CondensedTree::distance, nb::rv_policy::reference,
          "A 1D array of distances."
      )
      .def_ro(
          "child_size", &CondensedTree::child_size, nb::rv_policy::reference,
          "A 1D array of child sizes."
      )
      .def_ro(
          "cluster_rows", &CondensedTree::cluster_rows,
          nb::rv_policy::reference, "A 1D array of cluster row indices."
      )
      .def(
          "__iter__",
          [](CondensedTree const &self) {
            return nb::make_tuple(
                       self.parent, self.child, self.distance, self.child_size,
                       self.cluster_rows
            )
                .attr("__iter__")();
          }
      )
      .def(
          "__reduce__",
          [](CondensedTree &self) {
            return nb::make_tuple(
                nb::type<CondensedTree>(),
                nb::make_tuple(
                    self.parent, self.child, self.distance, self.child_size,
                    self.cluster_rows
                )
            );
          }
      )
      .doc() = "CondensedTree contains a pruned dendrogram.";

  m.def(
      "compute_condensed_tree", &compute_condensed_tree,
      nb::arg("linkage_tree"), nb::arg("minimum_spanning_tree"),
      nb::arg("num_points"), nb::arg("min_cluster_size") = 5.0f,
      nb::arg("sample_weights") = nb::none(),
      R"(
        Prunes a linkage tree to create a condensed tree.

        Parameters
        ----------
        linkage_tree
            The input linkage tree. Must originate from and have the same size
            as the spanning tree.
        spanning_tree
            The input minimum spanning tree (sorted).
        min_cluster_size
            The minimum size of clusters to be included in the condensed tree.
            Default is 5.0.
        sample_weights
            The data point sample weights. If not provided, all
            points get an equal weight. Must have a value for each data point!

        Returns
        -------
        condensed_tree
            A CondensedTree with parent, child, distance, child_size,
            and cluster_rows arrays. The child_size array contains the
            (weighted) size of the child cluster, which is the sum of the
            sample weights for all points in the child cluster. The cluster_rows
            array contains the row indices with clusters as child.
        )"
  );
}

void add_distance_bindings(nb::module_ &m) {
  m.def(
      "get_dist",
      [](char const *const metric, nb::kwargs const metric_kws) {
        // Must match Metric enumeration order!
        constexpr static std::array lookup{
            wrap_dist<Metric::Euclidean>,   wrap_dist<Metric::Cityblock>,
            wrap_dist<Metric::Chebyshev>,   wrap_dist<Metric::Minkowski>,
            wrap_dist<Metric::Hamming>,     wrap_dist<Metric::Braycurtis>,
            wrap_dist<Metric::Canberra>,    wrap_dist<Metric::Haversine>,
            wrap_dist<Metric::SEuclidean>,  wrap_dist<Metric::Mahalanobis>,
            wrap_dist<Metric::Dice>,        wrap_dist<Metric::Jaccard>,
            wrap_dist<Metric::Russellrao>,  wrap_dist<Metric::Rogerstanimoto>,
            wrap_dist<Metric::Sokalsneath>,
        };

        auto const idx = parse_metric(metric);
        if (idx >= lookup.size())
          throw nb::value_error(  //
              nb::str("Missing python wrapper for '{}'").format(metric).c_str()
          );

        return lookup[idx](metric_kws);
      },
      nb::arg("metric"), nb::arg("metric_kws"),
      R"(
        Retrieves the specified distance metric callback.

        Parameters
        ----------
        metric
          The name of the metric to use. See :py:attr:`~plscan.PLSCAN.VALID_BALLTREE_METRICS`
          for a list of valid metrics.
        **metric_kws
          p: The order of the Minkowski distance. Required if `metric` is "minkowski".
          V: The variance vector for the standardized Euclidean distance. Required if
          `metric` is "seuclidean".
          VI: The inverse covariance matrix for the Mahalanobis distance. Required if
          `metric` is "mahalanobis".

        Returns
        -------
        dist
            The distance function callback. Its input arrays must be c-contiguous.
      )"
  );
}

void add_labelling_bindings(nb::module_ &m) {
  nb::class_<Labelling>(m, "Labelling")
      .def(
          "__init__",
          [](Labelling *t, nb::handle label, nb::handle probability) {
            // Support np.memmap and np.ndarray input types for sklearn
            // pickling. The output of np.asarray can cast to nanobind ndarrays.
            auto const asarray = nb::module_::import_("numpy").attr("asarray");
            new (t) Labelling(
                nb::cast<array_ref<int32_t const>>(asarray(label), false),
                nb::cast<array_ref<float const>>(asarray(probability), false)
            );
          },
          nb::arg("label"), nb::arg("probability"),
          R"(
            Parameters
            ----------
            label
                The data point cluster labels.
            persistence
                The data point cluster membership probabilities.
          )"
      )
      .def_ro(
          "label", &Labelling::label, nb::rv_policy::reference,
          "A 1D array with cluster labels."
      )
      .def_ro(
          "probability", &Labelling::probability, nb::rv_policy::reference,
          "A 1D array with cluster membership probabilities."
      )
      .def(
          "__iter__",
          [](Labelling const &self) {
            return nb::make_tuple(self.label, self.probability)
                .attr("__iter__")();
          }
      )
      .def(
          "__reduce__",
          [](Labelling const &self) {
            return nb::make_tuple(
                nb::type<Labelling>(),
                nb::make_tuple(self.label, self.probability)
            );
          }
      )
      .doc() = "Labelling contains the cluster labels and probabilities.";

  m.def(
      "compute_cluster_labels", &compute_cluster_labels, nb::arg("leaf_tree"),
      nb::arg("condensed_tree"), nb::arg("selected_clusters").noconvert(),
      nb::arg("num_points"),
      R"(
        Computes cluster labels and membership probabilities for the points.

        Parameters
        ----------
        leaf_tree
            The input leaf tree.
        condensed_tree
            The input condensed tree.
        selected_clusters
            The condensed_tree parent IDs of the selected clusters.
        num_points
            The number of points in the condensed tree.

        Returns
        -------
        labelling
            The Labelling containing arrays for the cluster labels and
            membership probabilities. Labels -1 indicate points classified as
            noise.
      )"
  );
}

void add_leaf_tree_bindings(nb::module_ &m) {
  nb::class_<LeafTree>(m, "LeafTree")
      .def(
          "__init__",
          [](LeafTree *t, nb::handle parent, nb::handle min_distance,
             nb::handle max_distance, nb::handle min_size,
             nb::handle max_size) {
            // Support np.memmap and np.ndarray input types for sklearn
            // pickling. The output of np.asarray can cast to nanobind
            // ndarrays.
            auto const asarray = nb::module_::import_("numpy").attr("asarray");
            new (t) LeafTree(
                nb::cast<array_ref<uint32_t const>>(asarray(parent), false),
                nb::cast<array_ref<float const>>(asarray(min_distance), false),
                nb::cast<array_ref<float const>>(asarray(max_distance), false),
                nb::cast<array_ref<float const>>(asarray(min_size), false),
                nb::cast<array_ref<float const>>(asarray(max_size), false)
            );
          },
          nb::arg("parent"), nb::arg("min_distance"), nb::arg("max_distance"),
          nb::arg("min_size"), nb::arg("max_size"),
          R"(
            Parameters
            ----------
            parent
                The parent cluster IDs.
            min_distance
                The minimum distance at which the cluster ID exists in the
                condensed tree. The leaf-cluster represented by the cluster ID
                may exist at smaller distances because it can contain points of
                its descendants.
            max_distance
                The distance at which the cluster connects to its parent in the
                condensed tree and leaf tree.
            min_size
                The min_cluster_size at which the cluster becomes a leaf.
            max_size
                The min_cluster_size at which the cluster stops being a leaf.
          )"
      )
      .def_ro(
          "parent", &LeafTree::parent, nb::rv_policy::reference,
          "A 1D array with parent cluster IDs."
      )
      .def_ro(
          "min_distance", &LeafTree::min_distance, nb::rv_policy::reference,
          "A 1D array with minimum leaf cluster distances. I.e., the minimum "
          "distance at which the cluster ID exists in the condensed tree. The "
          "leaf-cluster represented by the cluster ID may exist at smaller "
          "distances because it can contain points of its children."
      )
      .def_ro(
          "max_distance", &LeafTree::max_distance, nb::rv_policy::reference,
          "A 1D array with maximum leaf cluster distances."
      )
      .def_ro(
          "min_size", &LeafTree::min_size, nb::rv_policy::reference,
          "A 1D array with minimum leaf cluster sizes."
      )
      .def_ro(
          "max_size", &LeafTree::max_size, nb::rv_policy::reference,
          "A 1D array with maximum leaf cluster sizes."
      )
      .def(
          "__iter__",
          [](LeafTree const &self) {
            return nb::make_tuple(
                       self.parent, self.min_distance, self.max_distance,
                       self.min_size, self.max_size
            )
                .attr("__iter__")();
          }
      )
      .def(
          "__reduce__",
          [](LeafTree const &self) {
            return nb::make_tuple(
                nb::type<LeafTree>(),
                nb::make_tuple(
                    self.parent, self.min_distance, self.max_distance,
                    self.min_size, self.max_size
                )
            );
          }
      )
      .doc() = R"(
        LeafTree lists information for the clusters in a condensed tree.

        Indexing with [cluster_id - num_points] gives information for the
        cluster with cluster_id.
      )";

  m.def(
      "compute_leaf_tree", &compute_leaf_tree, nb::arg("condensed_tree"),
      nb::arg("num_points"), nb::arg("min_cluster_size") = 5.0f,
      R"(
        Computes a leaf tree from a condensed tree.

        Parameters
        ----------
        condensed_tree
            The input condensed tree.
        num_points
            The number of points in the dataset.
        min_cluster_size
            The minimum size of clusters to be included in the leaf tree.

        Returns
        -------
        leaf_tree
            A LeafTree containing parent, min_distance, max_distance, min_size,
            and max_size arrays. The min/max distance arrays contain each
            cluster's distance range. The min_size and max_size arrays contain
            the minimum and maximum min_cluster_size thresholds for which the
            clusters are leaves, respectively. Some clusters never become
            leaves, indicated by a min_size larger than the max_size.
      )"
  );

  m.def(
      "apply_size_cut", &apply_size_cut, nb::arg("leaf_tree"),
      nb::arg("cut_size"),
      R"(
        Finds the cluster IDs for leaf-clusters that exist at the
        given cut_size threshold. The threshold is interpreted as a
        birth value in a left-open (birth, death] size interval.

        Parameters
        ----------
        leaf_tree
            The input leaf tree.
        size_cut
            The size threshold for selecting clusters. The threshold is
            interpreted as a birth value in a left-open (birth, death] size
            interval.

        Returns
        -------
        selected_clusters
            The cluster IDs for leaf-clusters that exist at the
            given cut_size threshold.
      )"
  );

  m.def(
      "apply_distance_cut", &apply_distance_cut, nb::arg("leaf_tree"),
      nb::arg("cut_distance"),
      R"(
        Finds the cluster IDs for clusters that exist at the given cut distance
        threshold.

        Parameters
        ----------
        leaf_tree
            The input leaf tree.
        distance_cut
            The distance threshold for selecting clusters.

        Returns
        -------
        selected_clusters
            The cluster IDs for clusters that exist at the given distance
            threshold.
      )"
  );
}

void add_linkage_tree_bindings(nb::module_ &m) {
  nb::class_<LinkageTree>(m, "LinkageTree")
      .def(
          nb::init<
              array_ref<uint32_t const>, array_ref<uint32_t const>,
              array_ref<uint32_t const>, array_ref<float const>>(),
          nb::arg("parent").noconvert(), nb::arg("child").noconvert(),
          nb::arg("child_count").noconvert(), nb::arg("child_size").noconvert(),
          R"(
            Parameters
            ----------
            parent
                An array of parent node and cluster indices. Clusters are
                labelled with indices starting from the number of points.
            child
                An array of child node and cluster indices. Clusters are labelled
                with indices starting from the number of points.
            child_count
                The number of points contained in the child side of the link.
            child_size
                The (weighted) size in the child side of the link.
          )"
      )
      .def_ro(
          "parent", &LinkageTree::parent, nb::rv_policy::reference,
          "A 1D array with parent values. "
      )
      .def_ro(
          "child", &LinkageTree::child, nb::rv_policy::reference,
          "A 1D array with child values. "
      )
      .def_ro(
          "child_count", &LinkageTree::child_count, nb::rv_policy::reference,
          "A 1D array with child_count values. "
      )
      .def_ro(
          "child_size", &LinkageTree::child_size, nb::rv_policy::reference,
          "A 1D array with child_size values. "
      )
      .def(
          "__iter__",
          [](LinkageTree const &self) {
            return nb::make_tuple(
                       self.parent, self.child, self.child_count,
                       self.child_size
            )
                .attr("__iter__")();
          }
      )
      .def(
          "__reduce__",
          [](LinkageTree const &self) {
            return nb::make_tuple(
                nb::type<LinkageTree>(),
                nb::make_tuple(
                    self.parent, self.child, self.child_count, self.child_size
                )
            );
          }
      )
      .doc() = "LinkageTree contains a single-linkage dendrogram.";

  m.def(
      "compute_linkage_tree", &compute_linkage_tree,
      nb::arg("minimum_spanning_tree"), nb::arg("num_points"),
      nb::arg("sample_weights") = nb::none(),
      R"(
        Constructs a LinkageTree containing a single-linkage
        dendrogram.

        Parameters
        ----------
        minimum_spanning_tree
            The SpanningTree containing the (sorted/partial) minimum spanning
            tree.
        num_points
            The number of data points in the data set.
        sample_weights
            The data point sample weights. If not provided, all points
            get an equal weight.

        Returns
        -------
        tree
            A LinkageTree containing the parent, child, child_count,
            and child_size arrays of the single-linkage dendrogram.
            Count refers to the number of data points in the child
            cluster. Size refers to the (weighted) size of the child
            cluster, which is the sum of the sample weights for all
            points in the child cluster.
      )"
  );
}

void add_persistence_trace_bindings(nb::module_ &m) {
  m.doc() = "Module for persistence trace computation in PLSCAN.";
  nb::class_<PersistenceTrace>(m, "PersistenceTrace")
      .def(
          "__init__",
          [](PersistenceTrace *t, nb::handle min_size, nb::handle persistence) {
            // Support np.memmap and np.ndarray input types for sklearn
            // pickling. The output of np.asarray can cast to nanobind ndarrays.
            auto const asarray = nb::module_::import_("numpy").attr("asarray");
            new (t) PersistenceTrace(
                nb::cast<array_ref<float const>>(asarray(min_size), false),
                nb::cast<array_ref<float const>>(asarray(persistence), false)
            );
          },
          nb::arg("min_size"), nb::arg("persistence"),
          R"(
            Parameters
            ----------
            min_size
                The minimum cluster sizes at which leaf-clusters start to exist.
            persistence
                The persistence sum for the leaf-clusters that exist at the
                minimum cluster sizes.
          )"
      )
      .def_ro(
          "min_size", &PersistenceTrace::min_size, nb::rv_policy::reference,
          "A 1D array with minimum cluster sizes."
      )
      .def_ro(
          "persistence", &PersistenceTrace::persistence,
          nb::rv_policy::reference, "A 1D array with total persistence values."
      )
      .def(
          "__iter__",
          [](PersistenceTrace const &self) {
            return nb::make_tuple(self.min_size, self.persistence)
                .attr("__iter__")();
          }
      )
      .def(
          "__reduce__",
          [](PersistenceTrace const &self) {
            return nb::make_tuple(
                nb::type<PersistenceTrace>(),
                nb::make_tuple(self.min_size, self.persistence)
            );
          }
      )
      .doc() = "PersistenceTrace lists the persistences per min_cluster_size.";

  m.def(
      "compute_size_persistence", &compute_size_persistence,
      nb::arg("leaf_tree"),
      R"(
        Computes the total min_cluster_size persistence trace.

        Parameters
        ----------
        leaf_tree
            The input leaf tree.

        Returns
        -------
        persistence_trace
            A PersistenceTrace containing arrays for the minimum cluster size
            and total persistence values. The min_size array contains all unique
            min_cluster_sizes at which clusters become leaves. The persistence
            array contains the total persistence of leaf clusters at those
            minimum size thresholds.
      )"
  );

  m.def(
      "compute_size_distance_bi_persistence",
      &compute_size_distance_bi_persistence, nb::arg("leaf_tree"),
      nb::arg("condensed_tree"), nb::arg("num_points"),
      R"(
        Computes a bi-persistence trace for over min_cluster_size and
        mutual reachability distances.

        Parameters
        ----------
        leaf_tree
            The input leaf tree.
        condensed_tree
            The input condensed tree.
        num_points
            The number of points in the condensed tree.

        Returns
        -------
        persistence_trace
            A PersistenceTrace containing arrays for the minimum cluster size
            and total persistence values. The min_size array contains all unique
            min_cluster_sizes at which clusters become leaves. The persistence
            array contains the total bi-persistence of leaf clusters at those
            minimum size thresholds.
      )"
  );

  m.def(
      "compute_size_density_bi_persistence",
      &compute_size_density_bi_persistence, nb::arg("leaf_tree"),
      nb::arg("condensed_tree"), nb::arg("num_points"),
      R"(
        Computes a bi-persistence trace for over min_cluster_size and
        mutual reachability densities.

        Parameters
        ----------
        leaf_tree
            The input leaf tree.
        condensed_tree
            The input condensed tree.
        num_points
            The number of points in the condensed tree.

        Returns
        -------
        persistence_trace
            A PersistenceTrace containing arrays for the minimum cluster size
            and total persistence values. The min_size array contains all unique
            min_cluster_sizes at which clusters become leaves. The persistence
            array contains the total bi-persistence of leaf clusters at those
            minimum size thresholds.
      )"
  );

  m.def(
      "compute_distance_icicles", &compute_distance_icicles,
      nb::arg("leaf_tree"), nb::arg("condensed_tree"), nb::arg("num_points"),
      R"(
        Computes size--distance traces for the LeafTree plot.

        Parameters
        ----------
        leaf_tree
            The input leaf tree.
        condensed_tree
            The input condensed tree.
        num_points
            The number of points in the condensed tree.

        Returns
        -------
        sizes
            The icicle min cluster sizes (births in (birth, death])).
        stabilities
            The icicle stabilities.
      )"
  );

  m.def(
      "compute_density_icicles", &compute_density_icicles, nb::arg("leaf_tree"),
      nb::arg("condensed_tree"), nb::arg("num_points"),
      R"(
        Computes size--density traces for the LeafTree plot.

        Parameters
        ----------
        leaf_tree
            The input leaf tree.
        condensed_tree
            The input condensed tree.
        num_points
            The number of points in the condensed tree.

        Returns
        -------
        sizes
            The icicle min cluster sizes (births in (birth, death])).
        stabilities
            The icicle stabilities.
      )"
  );
}

void add_space_tree_bindings(nb::module_ &m) {
  nb::class_<NodeData>(m, "NodeData")
      .def(
          nb::init<int64_t, int64_t, int64_t, double>(), nb::arg("idx_start"),
          nb::arg("idx_end"), nb::arg("is_leaf"), nb::arg("radius"),
          R"(
            Parameters
            ----------
            idx_start
                The starting index of the node in the tree.
            idx_end
                The ending index of the node in the tree.
            is_leaf
                A flag indicating whether the node is a leaf (1) or not (0).
            radius
                The radius of the node, used in BallTrees to define the
                hypersphere that contains the points in the node.
          )"
      )
      .def_ro(
          "idx_start", &NodeData::idx_start,
          "The starting index of the node in the tree."
      )
      .def_ro(
          "idx_end", &NodeData::idx_end,
          "The ending index of the node in the tree."
      )
      .def_ro(
          "is_leaf", &NodeData::is_leaf,
          "A flag indicating whether the node is a leaf (1) or not (0)."
      )
      .def_ro(
          "radius", &NodeData::radius,
          "The radius of the node, used in BallTrees to define the hypersphere "
          "that contains the points in the node."
      )
      .def(
          "__iter__",
          [](NodeData const &self) {
            return nb::make_tuple(
                       self.idx_start, self.idx_end, self.is_leaf, self.radius
            )
                .attr("__iter__")();
          }
      )
      .def(
          "__reduce__",
          [](NodeData const &self) {
            return nb::make_tuple(
                nb::type<NodeData>(),
                nb::make_tuple(
                    self.idx_start, self.idx_end, self.is_leaf, self.radius
                )
            );
          }
      )
      .doc() = "NodeData represents nodes in sklearn KDTrees and BallTrees.";

  nb::class_<SpaceTree>(m, "SpaceTree")
      .def(
          nb::init<
              ndarray_ref<float const, 2>, array_ref<int64_t const>,
              array_ref<double const>, ndarray_ref<float const, 3>>(),
          nb::arg("data").noconvert(), nb::arg("idx_array").noconvert(),
          nb::arg("node_data").noconvert(), nb::arg("node_bounds").noconvert(),
          R"(
            Parameters
            ----------
            data
                The data feature vectors.
            idx_array
                The tree's index array mapping points to tree nodes.
            node_data
                A float64 view into the structured :py:class:`~NodeData` array.
            node_bounds
                The node bounds, a 3D array (x, num_nodes, num_features),
                representing the min and max bounds of each node in the feature
                space.
          )"
      )
      .def_ro(
          "data", &SpaceTree::data, nb::rv_policy::reference,
          "A 2D array with feature vectors."
      )
      .def_ro(
          "idx_array", &SpaceTree::idx_array, nb::rv_policy::reference,
          "A 1D array mapping nodes to data points."
      )
      .def_ro(
          "node_data", &SpaceTree::node_data, nb::rv_policy::reference,
          "A 1D float64 view into a node data array."
      )
      .def_ro(
          "node_bounds", &SpaceTree::node_bounds, nb::rv_policy::reference,
          "A 3D array with node bounds."
      )
      .def(
          "__iter__",
          [](SpaceTree const &self) {
            return nb::make_tuple(
                       self.data, self.idx_array, self.node_data,
                       self.node_bounds
            )
                .attr("__iter__")();
          }
      )
      .def(
          "__reduce__",
          [](SpaceTree const &self) {
            return nb::make_tuple(
                nb::type<SpaceTree>(),
                nb::make_tuple(
                    self.data, self.idx_array, self.node_data, self.node_bounds
                )
            );
          }
      )
      .doc() = "SpaceTree represents sklearn KDTrees and BallTrees.";

  m.def(
      "check_node_data", &check_node_data, nb::arg("node_data").noconvert(),
      R"(
        Converts float64 node_data view to a list of NodeData objects.
        This function is used in tests to check whether the node data
        conversion works correctly!

        Parameters
        ----------
        node_data
            A flat float64 array view containing :py:class:`~NodeData`.

        Returns
        -------
        copied_data
            A list of :py:class:`~NodeData` objects created from the input
            array.
      )"
  );

  m.def(
      "kdtree_query", &kdtree_query, nb::arg("tree"), nb::arg("num_neighbors"),
      nb::arg("metric"), nb::arg("metric_kws"),
      R"(
        Performs a k-nearest neighbors query on a SpaceTree.

        Parameters
        ----------
        tree
            The SpaceTree to query (must be a KDTree!).
        num_neighbors
            The number of nearest neighbors to find.
        metric
            The distance metric to use. Supported metrics are listed in
            :py:attr:`~plscan.PLSCAN.VALID_KDTREE_METRICS`.
        metric_kws
            Additional keyword arguments for the distance function, such as
            the Minkowski distance parameter `p` for the "minkowski" metric.

        Returns
        -------
        knn
            A sparse graph containing the distance-sorted nearest neighbors for
            each point.
      )"
  );

  m.def(
      "balltree_query", &balltree_query, nb::arg("tree"),
      nb::arg("num_neighbors"), nb::arg("metric"), nb::arg("metric_kws"),
      R"(
        Performs a k-nearest neighbors query on a SpaceTree.

        Parameters
        ----------
        tree
            The SpaceTree to query (must be a BallTree!).
        num_neighbors
            The number of nearest neighbors to find.
        metric
            The distance metric to use. Supported metrics are listed in
            :py:attr:`~plscan.PLSCAN.VALID_BALLTREE_METRICS`.
        metric_kws
            Additional keyword arguments for the distance function, such as
            the Minkowski distance parameter `p` for the "minkowski" metric.

        Returns
        -------
        knn
            A sparse graph containing the distance-sorted nearest neighbors for
            each point.
      )"
  );
}

void add_spanning_tree_bindings(nb::module_ &m) {
  nb::class_<SpanningTree>(m, "SpanningTree")
      .def(
          "__init__",
          [](SpanningTree *t, nb::handle parent, nb::handle child,
             nb::handle distance) {
            // Support np.memmap and np.ndarray input types for sklearn
            // pickling. The output of np.asarray can cast to nanobind ndarrays.
            auto const asarray = nb::module_::import_("numpy").attr("asarray");
            new (t) SpanningTree(
                nb::cast<array_ref<uint32_t const>>(asarray(parent), false),
                nb::cast<array_ref<uint32_t const>>(asarray(child), false),
                nb::cast<array_ref<float const>>(asarray(distance), false)
            );
          },
          nb::arg("parent"), nb::arg("child"), nb::arg("distance"),
          R"(
            Parameters
            ----------
            parent
                An array of parent node indices.
            child
                An array of child node indices.
            distance
                An array of distances between the nodes.
          )"
      )
      .def_ro(
          "parent", &SpanningTree::parent, nb::rv_policy::reference,
          "A 1D array with parent values."
      )
      .def_ro(
          "child", &SpanningTree::child, nb::rv_policy::reference,
          "A 1D array with child values."
      )
      .def_ro(
          "distance", &SpanningTree::distance, nb::rv_policy::reference,
          "A 1D array with distance values."
      )
      .def(
          "__iter__",
          [](SpanningTree const &self) {
            return nb::make_tuple(self.parent, self.child, self.distance)
                .attr("__iter__")();
          }
      )
      .def(
          "__reduce__",
          [](SpanningTree const &self) {
            return nb::make_tuple(
                nb::type<SpanningTree>(),
                nb::make_tuple(self.parent, self.child, self.distance)
            );
          }
      )
      .doc() = "SpanningTree contains a sorted minimum spanning tree (MST).";

  m.def(
      "extract_spanning_forest", &extract_spanning_forest, nb::arg("graph"),
      R"(
        Extracts a minimum spanning forest from a sparse graph.

        Parameters
        ----------
        graph
            The input sparse graph.

        Returns
        -------
        spanning_tree
            The computed spanning forest.
        )"
  );

  m.def(
      "compute_spanning_tree_kdtree", &compute_spanning_tree_kdtree,
      nb::arg("tree"), nb::arg("knn"), nb::arg("core_distances"),
      nb::arg("metric"), nb::arg("metric_kws"),
      R"(
        Computes a minimum spanning tree (MST) using a k-d tree.

        Parameters
        ----------
        tree
            The kdtree structure.
        knn
            The k-nearest neighbors graph.
        core_distances
            The core distances for each point.
        metric
            The distance metric to use. Supported metrics are listed in
            :py:attr:`~plscan.PLSCAN.VALID_KDTREE_METRICS`.
        metric_kws
            Additional keyword arguments for the distance function, such as
            the Minkowski distance parameter `p` for the "minkowski" metric.

        Returns
        -------
        spanning_tree
            The computed minimum spanning tree.
        )"
  );

  m.def(
      "compute_spanning_tree_balltree", &compute_spanning_tree_balltree,
      nb::arg("tree"), nb::arg("knn"), nb::arg("core_distances"),
      nb::arg("metric"), nb::arg("metric_kws"),
      R"(
        Computes a minimum spanning tree (MST) using a ball tree.

        Parameters
        ----------
        tree
            The balltree structure.
        knn
            The k-nearest neighbors graph.
        core_distances
            The core distances for each point.
        metric
            The distance metric to use. Supported metrics are listed in
            :py:attr:`~plscan.PLSCAN.VALID_BALLTREE_METRICS`.
        metric_kws
            Additional keyword arguments for the distance function, such as
            the Minkowski distance parameter `p` for the "minkowski" metric.

        Returns
        -------
        spanning_tree
            The computed minimum spanning tree.
        )"
  );
}

void add_sparse_graph_bindings(nb::module_ &m) {
  nb::class_<SparseGraph>(m, "SparseGraph")
      .def(
          "__init__",
          [](SparseGraph *t, nb::handle data, nb::handle indices,
             nb::handle indptr) {
            // Support np.memmap and np.ndarray input types for sklearn
            // pickling. The output of np.asarray can cast to nanobind ndarrays.
            auto const asarray = nb::module_::import_("numpy").attr("asarray");
            new (t) SparseGraph(
                nb::cast<array_ref<float const>>(asarray(data), false),
                nb::cast<array_ref<int32_t const>>(asarray(indices), false),
                nb::cast<array_ref<int32_t const>>(asarray(indptr), false)
            );
          },
          nb::arg("data"), nb::arg("indices"), nb::arg("indptr"),
          R"(
          Parameters
          ----------
          data
              An array of distances.
          indices
              An array of column indices.
          indptr
              The CSR indptr array.
          )"
      )
      .def_ro(
          "data", &SparseGraph::data, nb::rv_policy::reference,
          "A 1D array with data values."
      )
      .def_ro(
          "indices", &SparseGraph::indices, nb::rv_policy::reference,
          "A 1D array with indices values."
      )
      .def_ro(
          "indptr", &SparseGraph::indptr, nb::rv_policy::reference,
          "A 1D array with indptr values."
      )
      .def(
          "__iter__",
          [](SparseGraph const &self) {
            return nb::make_tuple(self.data, self.indices, self.indptr)
                .attr("__iter__")();
          }
      )
      .def(
          "__reduce__",
          [](SparseGraph const &self) {
            return nb::make_tuple(
                nb::type<SparseGraph>(),
                nb::make_tuple(self.data, self.indices, self.indptr)
            );
          }
      )
      .doc() = "SparseGraph contains a (square) distance matrix in CSR format.";

  m.def(
      "extract_core_distances", &extract_core_distances, nb::arg("graph"),
      nb::arg("min_samples") = 5, nb::arg("is_sorted") = false,
      R"(
          Extracts core distances from a sparse graph.

          Parameters
          ----------
          graph
                The sparse graph to extract core distances from.
          min_samples
                The number of nearest neighbors to consider for core distance.
          is_sorted
                Whether the rows of the graph are sorted by distance.

          Returns
          -------
          core_distances
                An array of core distances.
        )"
  );

  m.def(
      "compute_mutual_reachability", &compute_mutual_reachability,
      nb::arg("graph"), nb::arg("core_distances"),
      R"(
          Applies core distances to a sparse graph to compute mutual
          reachability.

          Parameters
          ----------
          graph
                The sparse graph to extract core distances from.
          core_distances
                An array of core distances, one for each point in the graph.

          Returns
          -------
          mutual_graph
                A new sparse graph with mutual reachability distances. Rows are
                sorted by mutual reachability distance.
        )"
  );
}

void add_threading_bindings(nb::module_ &m) {
  m.def(
      "get_max_threads", &omp_get_max_threads,
      R"(Returns the default number of OpenMP threads used.)"
  );

  m.def(
      "set_num_threads", &omp_set_num_threads, nb::arg("num_threads"),
      R"(
          Sets the default number of OpenMP threads to use.

          Parameters
          ----------
          num_threads : int
                The number of threads to set.
        )"
  );
}

}  // namespace

// --- Module definition

NB_MODULE(_api, m) {
  m.doc() = "C++ API bindings for PLSCAN using nanobind.";
  add_condensed_bindings(m);
  add_distance_bindings(m);
  add_labelling_bindings(m);
  add_leaf_tree_bindings(m);
  add_linkage_tree_bindings(m);
  add_persistence_trace_bindings(m);
  add_space_tree_bindings(m);
  add_spanning_tree_bindings(m);
  add_sparse_graph_bindings(m);
  add_threading_bindings(m);
}
