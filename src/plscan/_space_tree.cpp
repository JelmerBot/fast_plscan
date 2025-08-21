#include "_space_tree.h"

#include <nanobind/stl/vector.h>

#include <functional>

#include "_distances.h"
#include "_sparse_graph.h"

// --- General space tree query

template <typename rdist_fun_t, typename min_rdist_fun_t>
class RowQueryState {
  SpaceTreeView const tree;
  std::span<float const> const point;
  std::span<float> const row_dists;
  std::span<int32_t> const row_indices;
  rdist_fun_t rdist_fun;
  min_rdist_fun_t min_rdist_fun;

 public:
  RowQueryState(
      SparseGraphWriteView const knn, SpaceTreeView const tree,
      rdist_fun_t rdist_fun, min_rdist_fun_t min_rdist_fun,
      int64_t const point_idx
  )
      : tree(tree),
        point(row_view(tree.data, point_idx)),
        row_dists(knn.data.subspan(
            knn.indptr[point_idx],
            knn.indptr[point_idx + 1] - knn.indptr[point_idx]
        )),
        row_indices(knn.indices.subspan(
            knn.indptr[point_idx],
            knn.indptr[point_idx + 1] - knn.indptr[point_idx]
        )),
        rdist_fun(std::move(rdist_fun)),
        min_rdist_fun(std::move(min_rdist_fun)) {}

  void perform_query() const {
    constexpr size_t node_idx = 0ul;
    float const lower_bound = min_rdist_fun(tree, point, node_idx);
    recursive_query(lower_bound, node_idx);
    deheap_sort();
  }

 private:
  void recursive_query(float const lower_bound, size_t const node_idx) const {
    if (lower_bound > row_dists[0])
      return;

    if (auto const &[idx_start, idx_end, is_leaf, _] = tree.node_data[node_idx];
        is_leaf)
      process_leaf(idx_start, idx_end);
    else
      traverse_node(node_idx);
  }

  void process_leaf(int64_t const idx_start, int64_t const idx_end) const {
    for (int64_t _i = idx_start; _i < idx_end; ++_i) {
      int64_t const idx = tree.idx_array[_i];
      if (float const dist = rdist_fun(point, row_view(tree.data, idx));
          dist < row_dists[0])
        heap_push(dist, static_cast<int32_t>(idx));
    }
  }

  void traverse_node(size_t const node_idx) const {
    size_t left = node_idx * 2 + 1;
    size_t right = left + 1;

    float left_lower_bound = min_rdist_fun(tree, point, left);
    float right_lower_bound = min_rdist_fun(tree, point, right);
    if (left_lower_bound > right_lower_bound) {
      std::swap(left, right);
      std::swap(left_lower_bound, right_lower_bound);
    }

    recursive_query(left_lower_bound, left);
    recursive_query(right_lower_bound, right);
  }

  void heap_push(float const dist, int32_t const neighbor) const {
    size_t idx = 0ul;
    size_t const num_neighbors = row_dists.size();

    // Replace the largest value at index 0 with the next largest.
    // stop at the to-be-inserted distance value!
    while (true) {
      size_t left = idx * 2 + 1;
      if (left >= num_neighbors)
        break;

      // Find the largest distance child
      size_t right = left + 1;
      float left_dist = row_dists[left];
      if (float right_dist = right >= num_neighbors ? 0.0f : row_dists[right];
          left_dist <= right_dist) {
        std::swap(left, right);
        std::swap(left_dist, right_dist);
      }

      // Shift values if child is larger than the new distance
      if (left_dist <= dist)
        break;
      row_dists[idx] = row_dists[left];
      row_indices[idx] = row_indices[left];
      idx = left;
    }

    // Place the new distance and index at the current position
    row_dists[idx] = dist;
    row_indices[idx] = neighbor;
  }

  void deheap_sort() const {
    size_t const num_neighbors = row_dists.size();
    for (size_t _i = 1ul; _i <= num_neighbors; ++_i) {
      size_t const idx = num_neighbors - _i;
      std::swap(row_dists[0], row_dists[idx]);
      std::swap(row_indices[0], row_indices[idx]);
      siftdown(idx);
    }
  }

  void siftdown(size_t const idx) const {
    std::span<float> sub_dists = row_dists.subspan(0, idx);
    std::span<int32_t> sub_indices = row_indices.subspan(0, idx);

    size_t element = 0ul;
    while (element * 2 + 1 < sub_dists.size()) {
      size_t const left = element * 2 + 1;
      size_t const right = left + 1;
      size_t largest = element;

      if (sub_dists[largest] < sub_dists[left])
        largest = left;

      if (right < sub_dists.size() and sub_dists[largest] < sub_dists[right])
        largest = right;

      if (largest == element)
        break;

      std::swap(sub_dists[element], sub_dists[largest]);
      std::swap(sub_indices[element], sub_indices[largest]);
      element = largest;
    }
  }
};

template <typename rdist_fun_t, typename min_rdist_fun_t>
void parallel_query(
    SparseGraphWriteView const knn, SpaceTreeView const tree,
    rdist_fun_t &&rdist_fun, min_rdist_fun_t &&min_rdist_fun
) {
  nb::gil_scoped_release guard{};
  // clang-format off
  #pragma omp parallel for default(none) shared(knn, tree, rdist_fun, min_rdist_fun)  // clang-format on
  for (int64_t point_idx = 0; point_idx < tree.data.shape(0); ++point_idx) {
    RowQueryState state{knn, tree, rdist_fun, min_rdist_fun, point_idx};
    state.perform_query();
  }
}

using parallel_query_fun_t =
    void (*)(SparseGraphWriteView, SpaceTreeView, nb::dict);

// --- KDTree specific query

template <Metric metric>
void run_parallel_kdtree_query(
    SparseGraphWriteView const knn, SpaceTreeView const tree,
    nb::dict const metric_kws
) {
  parallel_query(
      knn, tree, get_rdist<metric>(metric_kws),
      get_kdtree_min_rdist<metric>(metric_kws)
  );
}

parallel_query_fun_t get_kdtree_executor(char const *const metric) {
  static std::map<Metric, parallel_query_fun_t> lookup = {
      {Metric::Euclidean, run_parallel_kdtree_query<Metric::Euclidean>},
      {Metric::Cityblock, run_parallel_kdtree_query<Metric::Cityblock>},
      {Metric::Chebyshev, run_parallel_kdtree_query<Metric::Chebyshev>},
      {Metric::Minkowski, run_parallel_kdtree_query<Metric::Minkowski>},
  };

  if (auto const it = lookup.find(parse_metric(metric)); it != lookup.end())
    return it->second;

  throw nb::value_error(
      nb::str("Unsupported metric for KDTree query: {}").format(metric).c_str()
  );
}

SparseGraph kdtree_query(
    SpaceTree const tree, uint32_t const num_neighbors,
    char const *const metric, nb::dict const metric_kws
) {
  auto [knn_view, knn_cap] = SparseGraph::allocate_knn(
      tree.data.shape(0), num_neighbors
  );
  get_kdtree_executor(metric)(knn_view, tree.view(), metric_kws);
  return {knn_view, knn_cap};
}

// --- Ball specific query

template <Metric metric>
void run_parallel_balltree_query(
    SparseGraphWriteView const knn, SpaceTreeView const tree,
    nb::dict const metric_kws
) {
  parallel_query(
      knn, tree, get_rdist<metric>(metric_kws),
      get_balltree_min_rdist<metric>(metric_kws)
  );
}

parallel_query_fun_t get_balltree_executor(char const *const metric) {
  static std::map<Metric, parallel_query_fun_t> lookup = {
      {Metric::Euclidean, run_parallel_balltree_query<Metric::Euclidean>},
      {Metric::Cityblock, run_parallel_balltree_query<Metric::Cityblock>},
      {Metric::Chebyshev, run_parallel_balltree_query<Metric::Chebyshev>},
      {Metric::Minkowski, run_parallel_balltree_query<Metric::Minkowski>},
      {Metric::Hamming, run_parallel_balltree_query<Metric::Hamming>},
      {Metric::Braycurtis, run_parallel_balltree_query<Metric::Braycurtis>},
      {Metric::Canberra, run_parallel_balltree_query<Metric::Canberra>},
      {Metric::Haversine, run_parallel_balltree_query<Metric::Haversine>},
      {Metric::SEuclidean, run_parallel_balltree_query<Metric::SEuclidean>},
      {Metric::Mahalanobis, run_parallel_balltree_query<Metric::Mahalanobis>},
      {Metric::Dice, run_parallel_balltree_query<Metric::Dice>},
      {Metric::Jaccard, run_parallel_balltree_query<Metric::Jaccard>},
      {Metric::Russellrao, run_parallel_balltree_query<Metric::Russellrao>},
      {Metric::Rogerstanimoto,
       run_parallel_balltree_query<Metric::Rogerstanimoto>},
      {Metric::Sokalsneath, run_parallel_balltree_query<Metric::Sokalsneath>}
  };

  if (auto const it = lookup.find(parse_metric(metric)); it != lookup.end())
    return it->second;

  throw nb::value_error(  //
      nb::str("Unsupported metric for BallTree query: {}")
          .format(metric)
          .c_str()
  );
}

SparseGraph balltree_query(
    SpaceTree const tree, uint32_t const num_neighbors,
    char const *const metric, nb::dict const metric_kws
) {
  auto [knn_view, knn_cap] = SparseGraph::allocate_knn(
      tree.data.shape(0), num_neighbors
  );
  get_balltree_executor(metric)(knn_view, tree.view(), metric_kws);
  return {knn_view, knn_cap};
}

// --- Test for space tree python to c++ translation

std::vector<NodeData> check_node_data(array_ref<double const> const node_data) {
  auto range = convert_node_data(node_data);
  return {range.begin(), range.end()};
}

// --- Module definitions

NB_MODULE(_space_tree, m) {
  m.doc() = "Module for space tree computations in PLSCAN.";

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
          "A 1D array mapping nodes to data points (np.int64)."
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