#include "space_tree.h"

#include <array>
#include <functional>

// --- Implementation details

namespace {

// Reinterprets 64bit float array view as an array of NodeData objects
std::span<NodeData const> convert_node_data(
    array_ref<double const> const &node_data
) {
  return {
      reinterpret_cast<NodeData const *>(node_data.data()), node_data.size() / 4
  };
}

// General space tree query

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

// KDTree specific query

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
  // Must match Metric enumeration order!
  constexpr static std::array lookup = {
      run_parallel_kdtree_query<Metric::Euclidean>,
      run_parallel_kdtree_query<Metric::Cityblock>,
      run_parallel_kdtree_query<Metric::Chebyshev>,
      run_parallel_kdtree_query<Metric::Minkowski>,
  };

  auto const idx = parse_metric(metric);
  if (idx >= lookup.size())
    throw nb::value_error(  //
        nb::str("Missing KDTree query for '{}'").format(metric).c_str()
    );

  return lookup[idx];
}

// Ball specific query

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
  // Must match Metric enumeration order!
  constexpr static std::array lookup = {
      run_parallel_balltree_query<Metric::Euclidean>,
      run_parallel_balltree_query<Metric::Cityblock>,
      run_parallel_balltree_query<Metric::Chebyshev>,
      run_parallel_balltree_query<Metric::Minkowski>,
      run_parallel_balltree_query<Metric::Hamming>,
      run_parallel_balltree_query<Metric::Braycurtis>,
      run_parallel_balltree_query<Metric::Canberra>,
      run_parallel_balltree_query<Metric::Haversine>,
      run_parallel_balltree_query<Metric::SEuclidean>,
      run_parallel_balltree_query<Metric::Mahalanobis>,
      run_parallel_balltree_query<Metric::Dice>,
      run_parallel_balltree_query<Metric::Jaccard>,
      run_parallel_balltree_query<Metric::Russellrao>,
      run_parallel_balltree_query<Metric::Rogerstanimoto>,
      run_parallel_balltree_query<Metric::Sokalsneath>,
  };

  auto const idx = parse_metric(metric);
  if (idx >= lookup.size())
    throw nb::value_error(  //
        nb::str("Missing BallTree query for '{}'").format(metric).c_str()
    );

  return lookup[idx];
}

}  // namespace

// --- Function API

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

std::vector<NodeData> check_node_data(array_ref<double const> const node_data) {
  auto range = convert_node_data(node_data);
  return {range.begin(), range.end()};
}

// --- Class API

SpaceTreeView SpaceTree::view() const {
  return {
      data.view(), to_view(idx_array), convert_node_data(node_data),
      node_bounds.view()
  };
}
