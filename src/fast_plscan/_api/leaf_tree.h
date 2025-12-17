#ifndef PLSCAN_LEAF_TREE_H
#define PLSCAN_LEAF_TREE_H

#include "array.h"
#include "condensed_tree.h"

// Ownership for a LeafTreeWriteView.
struct LeafTreeCapsule {
  nb::capsule parent;
  nb::capsule min_distance;
  nb::capsule max_distance;
  nb::capsule min_size;
  nb::capsule max_size;
};

// Non-owning view of a leaf tree
struct LeafTreeWriteView {
  std::span<uint32_t> const parent;
  std::span<float> const min_distance;
  std::span<float> const max_distance;
  std::span<float> const min_size;
  std::span<float> const max_size;

  [[nodiscard]] size_t size() const;
};

// Non-owning view of a leaf tree
struct LeafTreeView {
  std::span<uint32_t const> const parent;
  std::span<float const> const min_distance;
  std::span<float const> const max_distance;
  std::span<float const> const min_size;
  std::span<float const> const max_size;

  [[nodiscard]] size_t size() const;
};

struct LeafTree {
  array_ref<uint32_t const> parent;
  // [min_dist, max_dist)
  array_ref<float const> min_distance;
  array_ref<float const> max_distance;
  // (min_size, max_size]
  array_ref<float const> min_size;
  array_ref<float const> max_size;

  LeafTree() = default;
  LeafTree(LeafTree &&) = default;
  LeafTree(LeafTree const &) = default;
  LeafTree &operator=(LeafTree &&) = default;
  LeafTree &operator=(LeafTree const &) = default;

  // Python side constructor.
  LeafTree(
      array_ref<uint32_t const> parent, array_ref<float const> min_distance,
      array_ref<float const> max_distance, array_ref<float const> min_size,
      array_ref<float const> max_size
  );

  // C++ side constructor.
  LeafTree(LeafTreeWriteView view, LeafTreeCapsule cap);

  // Allocate buffers to fill later.
  static std::pair<LeafTreeWriteView, LeafTreeCapsule> allocate(
      size_t num_clusters
  );

  [[nodiscard]] LeafTreeView view() const;
  [[nodiscard]] size_t size() const;
};

// --- Function API

LeafTree compute_leaf_tree(
    CondensedTree condensed_tree, size_t num_points, float min_size
);

array_ref<uint32_t const> apply_size_cut(
    LeafTree const &leaf_tree, float cut_size
);

array_ref<uint32_t const> apply_distance_cut(
    LeafTree const &leaf_tree, float cut_distance
);

#endif  // PLSCAN_LEAF_TREE_H
