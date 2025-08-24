#ifndef PLSCAN_LEAF_TREE_H
#define PLSCAN_LEAF_TREE_H

#include <span>

#include "_array.h"

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

  [[nodiscard]] size_t size() const {
    return parent.size();
  }
};

// Non-owning view of a leaf tree
struct LeafTreeView {
  std::span<uint32_t const> const parent;
  std::span<float const> const min_distance;
  std::span<float const> const max_distance;
  std::span<float const> const min_size;
  std::span<float const> const max_size;

  [[nodiscard]] size_t size() const {
    return parent.size();
  }
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
      array_ref<uint32_t const> const parent,
      array_ref<float const> const min_distance,
      array_ref<float const> const max_distance,
      array_ref<float const> const min_size,
      array_ref<float const> const max_size
  )
      : parent(parent),
        min_distance(min_distance),
        max_distance(max_distance),
        min_size(min_size),
        max_size(max_size) {}

  // C++ side constructor.
  LeafTree(LeafTreeWriteView const view, LeafTreeCapsule cap)
      : parent(  //
            to_array(view.parent, std::move(cap.parent), view.parent.size())
        ),
        min_distance(to_array(
            view.min_distance, std::move(cap.min_distance),
            view.min_distance.size()
        )),
        max_distance(to_array(
            view.max_distance, std::move(cap.max_distance),
            view.max_distance.size()
        )),
        min_size(to_array(
            view.min_size, std::move(cap.min_size), view.min_size.size()
        )),
        max_size(to_array(
            view.max_size, std::move(cap.max_size), view.max_size.size()
        )) {}

  // Allocate buffers to fill later.
  static auto allocate(size_t const num_clusters) {
    auto [parent, parent_cap] = new_buffer<uint32_t>(num_clusters);
    auto [min_distance, min_distance_cap] = new_buffer<float>(num_clusters);
    auto [max_distance, max_distance_cap] = new_buffer<float>(num_clusters);
    auto [min_size, min_size_cap] = new_buffer<float>(num_clusters);
    auto [max_size, max_size_cap] = new_buffer<float>(num_clusters);
    return std::make_pair(
        LeafTreeWriteView{
            parent, min_distance, max_distance, min_size, max_size
        },
        LeafTreeCapsule{
            std::move(parent_cap), std::move(min_distance_cap),
            std::move(max_distance_cap), std::move(min_size_cap),
            std::move(max_size_cap)
        }
    );
  }

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
