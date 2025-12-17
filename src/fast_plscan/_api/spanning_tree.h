#ifndef PLSCAN_API_SPANNING_TREE_H
#define PLSCAN_API_SPANNING_TREE_H

#include "array.h"
#include "space_tree.h"
#include "sparse_graph.h"

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

  [[nodiscard]] size_t size() const;
};

// Non-owning view of a spanning tree
struct SpanningTreeView {
  std::span<uint32_t const> const parent;
  std::span<uint32_t const> const child;
  std::span<float const> const distance;

  [[nodiscard]] size_t size() const;
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
      array_ref<uint32_t const> parent, array_ref<uint32_t const> child,
      array_ref<float const> distance
  );

  // C++ side constructor that converts buffers to potentially smaller arrays.
  SpanningTree(
      SpanningTreeWriteView view, SpanningTreeCapsule cap, size_t num_edges
  );

  // Allocate buffers to fill and resize later.
  static std::pair<SpanningTreeWriteView, SpanningTreeCapsule> allocate(
      size_t num_edges
  );

  [[nodiscard]] SpanningTreeView view() const;
  [[nodiscard]] size_t size() const;
};

// --- Function API

SpanningTree extract_spanning_forest(SparseGraph graph);

SpanningTree compute_spanning_tree_kdtree(
    SpaceTree tree, SparseGraph knn, array_ref<float> core_distances,
    char const *metric, nb::dict metric_kws
);

SpanningTree compute_spanning_tree_balltree(
    SpaceTree tree, SparseGraph knn, array_ref<float> core_distances,
    char const *metric, nb::dict metric_kws
);

#endif  // PLSCAN_API_SPANNING_TREE_H
