#ifndef PLSCAN_API_LINKAGE_H
#define PLSCAN_API_LINKAGE_H

#include <span>

#include "_array.h"

// Ownership for a LinkageTreeView.
struct LinkageTreeCapsule {
  nb::capsule parent;
  nb::capsule child;
  nb::capsule child_count;
  nb::capsule child_size;
};

// Non-owning view of a leaf tree
struct LinkageTreeView {
  std::span<uint32_t> parent;
  std::span<uint32_t> child;
  std::span<uint32_t> child_count;
  std::span<float> child_size;

  [[nodiscard]] size_t size() const {
    return parent.size();
  }
};

struct LinkageTree {
  array_ref<uint32_t> const parent;
  array_ref<uint32_t> const child;
  array_ref<uint32_t> const child_count;
  array_ref<float> const child_size;

  LinkageTree() = default;
  LinkageTree(LinkageTree &&) = default;
  LinkageTree(LinkageTree const &) = default;

  // Python side constructor.
  LinkageTree(
      array_ref<uint32_t> const parent, array_ref<uint32_t> const child,
      array_ref<uint32_t> const child_count, array_ref<float> const child_size
  )
      : parent(parent),
        child(child),
        child_count(child_count),
        child_size(child_size){}

  // C++ side constructor that converts buffers to potentially smaller arrays
  LinkageTree(
      LinkageTreeView const view, LinkageTreeCapsule cap, size_t const num_edges
  )
      : parent(to_array(view.parent, std::move(cap.parent), num_edges)),
        child(to_array(view.child, std::move(cap.child), num_edges)),
        child_count(
            to_array(view.child_count, std::move(cap.child_count), num_edges)
        ),
        child_size(
            to_array(view.child_size, std::move(cap.child_size), num_edges)
        ) {}

  // Allocate buffers to fill and resize later.
  static auto allocate(size_t const num_edges) {
    size_t const buffer_size = 2 * num_edges;
    auto [parent, parent_cap] = new_buffer<uint32_t>(buffer_size);
    auto [child, child_cap] = new_buffer<uint32_t>(buffer_size);
    auto [count, count_cap] = new_buffer<uint32_t>(buffer_size);
    auto [size, size_cap] = new_buffer<float>(buffer_size);
    return std::make_pair(
        LinkageTreeView{parent, child, count, size},
        LinkageTreeCapsule{parent_cap, child_cap, count_cap, size_cap}
    );
  }

  [[nodiscard]] LinkageTreeView view() const {
    return {
        to_view(parent),
        to_view(child),
        to_view(child_count),
        to_view(child_size),
    };
  }

  [[nodiscard]] size_t size() const {
    return parent.size();
  }
};

#endif  // PLSCAN_API_LINKAGE_H
