#ifndef PLSCAN_API_SPACE_TREE_H
#define PLSCAN_API_SPACE_TREE_H

#include <span>

#include "_array.h"
#include "_distances.h"

struct NodeData {
  int64_t idx_start;
  int64_t idx_end;
  int64_t is_leaf;
  double radius;
};

// Reinterprets 64bit float array view as an array of NodeData objects
inline std::span<NodeData const> convert_node_data(
    array_ref<double const> const &node_data
) {
  return {
      reinterpret_cast<NodeData const *>(node_data.data()), node_data.size() / 4
  };
}

struct SpaceTreeView {
  nb::ndarray_view<float const, 2, 'C'> const data;
  std::span<int64_t const> const idx_array;
  std::span<NodeData const> const node_data;
  nb::ndarray_view<float const, 3, 'C'> const node_bounds;
};

struct SpaceTree {
  ndarray_ref<float const, 2> data;
  array_ref<int64_t const> idx_array;
  array_ref<double const> node_data;
  ndarray_ref<float const, 3> node_bounds;

  [[nodiscard]] SpaceTreeView view() const {
    return {
        data.view(), to_view(idx_array), convert_node_data(node_data),
        node_bounds.view()
    };
  }
};

// --- KDTree distance point-to-node lower bound functions

template <Metric metric, typename... Args>
  requires MinkowskiMetric<metric>
float kdtree_min_rdist(
    SpaceTreeView const tree, std::span<float const> const point,
    size_t const node_idx, Args... args
) {
  float rdist = 0.0f;
  for (size_t idx = 0; idx < point.size(); ++idx) {
    float const d_lo = tree.node_bounds(0, node_idx, idx) - point[idx];
    float const d_hi = point[idx] - tree.node_bounds(1, node_idx, idx);
    float const diff = 0.5f * (d_lo + std::abs(d_lo) + d_hi + std::abs(d_hi));
    apply_minkowski<metric>(rdist, diff, args...);
  }
  return rdist;
}

template <Metric metric>
  requires MinkowskiMetric<metric>
auto get_kdtree_min_rdist(nb::dict const kwargs) {
  if constexpr (metric == Metric::Minkowski) {
    return [p = nb::cast<float>(kwargs["p"])](
               SpaceTreeView const tree, std::span<float const> const point,
               size_t const node_idx
           ) { return kdtree_min_rdist<metric>(tree, point, node_idx, p); };
  } else
    return kdtree_min_rdist<metric>;
}

// --- BallTree distance point-to-node lower bound functions

template <Metric metric>
auto get_balltree_min_rdist(nb::dict const kwargs) {
  return [dist_fun = get_dist<metric>(kwargs),
          to_rdist = get_dist_to_rdist<metric>(kwargs)](  //
             SpaceTreeView const tree, std::span<float const> const point,
             size_t const node_idx
         ) {
    // Work with full distances to match the ball tree radius and retain
    // precision. Converting the radius to rdist instead does not produce
    // accurate enough results.
    std::span const centre = {&tree.node_bounds(0, node_idx, 0), point.size()};
    double const diff = dist_fun(point, centre);
    double const radius = tree.node_data[node_idx].radius;
    float const dist = static_cast<float>(std::max(0.0, diff - radius));
    // Convert to rdist for space tree traversal!
    return to_rdist(dist);
  };
}

#endif  // PLSCAN_API_SPACE_TREE_H
