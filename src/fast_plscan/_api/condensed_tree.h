#ifndef PLSCAN_API_CONDENSED_H
#define PLSCAN_API_CONDENSED_H

#include <optional>

#include "array.h"
#include "linkage_tree.h"
#include "spanning_tree.h"

// Ownership for a CondensedTreeWriteView.
struct CondensedTreeCapsule {
  nb::capsule parent;
  nb::capsule child;
  nb::capsule distance;
  nb::capsule child_size;
  nb::capsule cluster_rows;
};

// Mutable non-owning view of a condensed tree
struct CondensedTreeWriteView {
  std::span<uint32_t> const parent;
  std::span<uint32_t> const child;
  std::span<float> const distance;
  std::span<float> const child_size;
  std::span<uint32_t> const cluster_rows;

  [[nodiscard]] size_t size() const;
};

// Readonly non-owning view of a condensed tree
struct CondensedTreeView {
  std::span<uint32_t const> const parent;
  std::span<uint32_t const> const child;
  std::span<float const> const distance;
  std::span<float const> const child_size;
  std::span<uint32_t const> const cluster_rows;

  [[nodiscard]] size_t size() const;
};

// Reference counted owning condensed tree
struct CondensedTree {
  array_ref<uint32_t const> parent;
  array_ref<uint32_t const> child;
  array_ref<float const> distance;
  array_ref<float const> child_size;
  array_ref<uint32_t const> cluster_rows;

  CondensedTree() = default;
  CondensedTree(CondensedTree &&) = default;
  CondensedTree(CondensedTree const &) = default;
  CondensedTree &operator=(CondensedTree &&) = default;
  CondensedTree &operator=(CondensedTree const &) = default;

  // Python side constructor.
  CondensedTree(
      array_ref<uint32_t const> parent, array_ref<uint32_t const> child,
      array_ref<float const> distance, array_ref<float const> child_size,
      array_ref<uint32_t const> cluster_rows
  );

  // C++ side constructor that converts buffers to potentially smaller arrays
  CondensedTree(
      CondensedTreeWriteView view, CondensedTreeCapsule cap, size_t num_edges,
      size_t num_clusters
  );

  // Allocate buffers to fill and resize later.
  static std::pair<CondensedTreeWriteView, CondensedTreeCapsule> allocate(
      size_t num_edges
  );

  [[nodiscard]] CondensedTreeView view() const;
  [[nodiscard]] size_t size() const;
};

// --- Function API

CondensedTree compute_condensed_tree(
    LinkageTree linkage, SpanningTree mst, size_t num_points, float min_size,
    std::optional<array_ref<float>> sample_weights
);

#endif  // PLSCAN_API_CONDENSED_H
