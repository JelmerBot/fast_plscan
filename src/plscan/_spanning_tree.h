#ifndef PLSCAN_API_SPANNING_TREE_H
#define PLSCAN_API_SPANNING_TREE_H

#include <span>

#include "_array.h"

// Ownership for a SpanningTreeView.
struct SpanningTreeCapsule {
  nb::capsule parent;
  nb::capsule child;
  nb::capsule distance;
};

// Non-owning view of a spanning tree
struct SpanningTreeWriteView {
  std::span<uint32_t> const parent;
  std::span<uint32_t> const child;
  std::span<float> const distance;

  [[nodiscard]] size_t size() const {
    return parent.size();
  }
};

// Non-owning view of a spanning tree
struct SpanningTreeView {
  std::span<uint32_t const> const parent;
  std::span<uint32_t const> const child;
  std::span<float const> const distance;

  [[nodiscard]] size_t size() const {
    return parent.size();
  }
};

struct SpanningTree {
  array_ref<uint32_t const> parent;
  array_ref<uint32_t const> child;
  array_ref<float const> distance;

  SpanningTree() = default;
  SpanningTree(SpanningTree &&) = default;
  SpanningTree(SpanningTree const &) = default;
  SpanningTree &operator=(SpanningTree &&) = default;
  SpanningTree &operator=(SpanningTree const &) = default;

  // Python side constructor.
  SpanningTree(
      array_ref<uint32_t const> const parent,
      array_ref<uint32_t const> const child,
      array_ref<float const> const distance
  )
      : parent(parent), child(child), distance(distance){};

  // C++ side constructor that converts buffers to potentially smaller arrays.
  SpanningTree(
      SpanningTreeWriteView const view, SpanningTreeCapsule cap,
      size_t const num_edges
  )
      : parent(to_array(view.parent, std::move(cap.parent), num_edges)),
        child(to_array(view.child, std::move(cap.child), num_edges)),
        distance(to_array(view.distance, std::move(cap.distance), num_edges)) {}

  // Allocate buffers to fill and resize later.
  static auto allocate(size_t const num_edges) {
    auto [parents, parent_cap] = new_buffer<uint32_t>(num_edges);
    auto [children, child_cap] = new_buffer<uint32_t>(num_edges);
    auto [distances, distance_cap] = new_buffer<float>(num_edges);
    return std::make_pair(
        SpanningTreeWriteView{parents, children, distances},
        SpanningTreeCapsule{
            std::move(parent_cap), std::move(child_cap), std::move(distance_cap)
        }
    );
  }

  [[nodiscard]] SpanningTreeView view() const {
    return {to_view(parent), to_view(child), to_view(distance)};
  }

  [[nodiscard]] size_t size() const {
    return parent.size();
  }
};

#endif  // PLSCAN_API_SPANNING_TREE_H
