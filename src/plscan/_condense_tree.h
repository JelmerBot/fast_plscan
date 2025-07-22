#ifndef PLSCAN_API_CONDENSED_H
#define PLSCAN_API_CONDENSED_H

#include <span>

#include "_array.h"

// Ownership for a CondensedTreeView.
struct CondensedTreeCapsule {
  nb::capsule parent;
  nb::capsule child;
  nb::capsule distance;
  nb::capsule child_size;
  nb::capsule cluster_rows;
};

// Non-owning view of a condensed tree
struct CondensedTreeView {
  std::span<uint32_t> parent;
  std::span<uint32_t> child;
  std::span<float> distance;
  std::span<float> child_size;
  std::span<uint32_t> cluster_rows;

  [[nodiscard]] size_t size() const {
    return parent.size();
  }
};

struct CondensedTree {
  array_ref<uint32_t> const parent;
  array_ref<uint32_t> const child;
  array_ref<float> const distance;
  array_ref<float> const child_size;
  array_ref<uint32_t> const cluster_rows;

  CondensedTree() = default;
  CondensedTree(CondensedTree &&) = default;
  CondensedTree(CondensedTree const &) = default;

  // Python side constructor.
  CondensedTree(
      array_ref<uint32_t> const parent, array_ref<uint32_t> const child,
      array_ref<float> const distance, array_ref<float> const child_size,
      array_ref<uint32_t> const cluster_rows
  )
      : parent(parent),
        child(child),
        distance(distance),
        child_size(child_size),
        cluster_rows(cluster_rows){}

  // C++ side constructor that converts buffers to potentially smaller arrays
  CondensedTree(
      CondensedTreeView const view, CondensedTreeCapsule cap,
      size_t const num_edges, size_t const num_clusters
  )
      : parent(to_array(view.parent, std::move(cap.parent), num_edges)),
        child(to_array(view.child, std::move(cap.child), num_edges)),
        distance(to_array(view.distance, std::move(cap.distance), num_edges)),
        child_size(
            to_array(view.child_size, std::move(cap.child_size), num_edges)
        ),
        cluster_rows(to_array(
            view.cluster_rows, std::move(cap.cluster_rows), num_clusters
        )) {}

  // Allocate buffers to fill and resize later.
  static auto allocate(size_t const num_edges) {
    size_t const buffer_size = 2 * num_edges;
    auto [parent, parent_cap] = new_buffer<uint32_t>(buffer_size);
    auto [child, child_cap] = new_buffer<uint32_t>(buffer_size);
    auto [dist, dist_cap] = new_buffer<float>(buffer_size);
    auto [size, size_cap] = new_buffer<float>(buffer_size);
    auto [rows, rows_cap] = new_buffer<uint32_t>(num_edges);
    return std::make_pair(
        CondensedTreeView{parent, child, dist, size, rows},
        CondensedTreeCapsule{
            parent_cap, child_cap, dist_cap, size_cap, rows_cap
        }
    );
  }

  [[nodiscard]] CondensedTreeView view() const {
    return {
        to_view(parent),     to_view(child),        to_view(distance),
        to_view(child_size), to_view(cluster_rows),
    };
  }

  [[nodiscard]] size_t size() const {
    return parent.size();
  }
};

#endif  // PLSCAN_API_CONDENSED_H
