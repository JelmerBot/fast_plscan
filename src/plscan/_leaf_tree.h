#ifndef PLSCAN_LEAF_TREE_H
#define PLSCAN_LEAF_TREE_H

#include <span>

#include "_array.h"

// Non-owning view of a leaf tree
struct LeafTreeView {
  std::span<uint32_t> parent;
  std::span<float> min_distance;
  std::span<float> max_distance;
  std::span<float> min_size;
  std::span<float> max_size;

  [[nodiscard]] size_t size() const {
    return parent.size();
  }
};

struct LeafTree {
  array_ref<uint32_t> const parent;
  // [min_dist, max_dist)
  array_ref<float> const min_distance;
  array_ref<float> const max_distance;
  // (min_size, max_size]
  array_ref<float> const min_size;
  array_ref<float> const max_size;

  LeafTree() = default;
  LeafTree(LeafTree &&) = default;
  LeafTree(LeafTree const &) = default;

  // Python side constructor.
  LeafTree(
      array_ref<uint32_t> const parent, array_ref<float> const min_distance,
      array_ref<float> const max_distance, array_ref<float> const min_size,
      array_ref<float> const max_size
  )
      : parent(parent),
        min_distance(min_distance),
        max_distance(max_distance),
        min_size(min_size),
        max_size(max_size){}

  // C++ side constructor.
  explicit LeafTree(size_t const num_cluster_ids)
      : parent(new_array<uint32_t>(num_cluster_ids)),
        min_distance(new_array<float>(num_cluster_ids)),
        max_distance(new_array<float>(num_cluster_ids)),
        min_size(new_array<float>(num_cluster_ids)),
        max_size(new_array<float>(num_cluster_ids)) {}

  [[nodiscard]] LeafTreeView view() const {
    return {
        to_view(parent),   to_view(min_distance), to_view(max_distance),
        to_view(min_size), to_view(max_size),
    };
  }

  [[nodiscard]] size_t size() const {
    return parent.size();
  }
};

#endif  // PLSCAN_LEAF_TREE_H
