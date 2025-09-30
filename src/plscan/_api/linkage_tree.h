#ifndef PLSCAN_API_LINKAGE_H
#define PLSCAN_API_LINKAGE_H

#include <optional>

#include "array.h"
#include "spanning_tree.h"

// Ownership for a LinkageTreeView.
struct LinkageTreeCapsule {
  nb::capsule parent;
  nb::capsule child;
  nb::capsule child_count;
  nb::capsule child_size;
};

// Non-owning view of a leaf tree
struct LinkageTreeWriteView {
  std::span<uint32_t> const parent;
  std::span<uint32_t> const child;
  std::span<uint32_t> const child_count;
  std::span<float> const child_size;

  [[nodiscard]] size_t size() const;
};

// Non-owning view of a leaf tree
struct LinkageTreeView {
  std::span<uint32_t const> const parent;
  std::span<uint32_t const> const child;
  std::span<uint32_t const> const child_count;
  std::span<float const> const child_size;

  [[nodiscard]] size_t size() const;
};

struct LinkageTree {
  array_ref<uint32_t const> parent;
  array_ref<uint32_t const> child;
  array_ref<uint32_t const> child_count;
  array_ref<float const> child_size;

  LinkageTree() = default;
  LinkageTree(LinkageTree &&) = default;
  LinkageTree(LinkageTree const &) = default;
  LinkageTree &operator=(LinkageTree &&) = default;
  LinkageTree &operator=(LinkageTree const &) = default;

  // Python side constructor.
  LinkageTree(
      array_ref<uint32_t const> parent, array_ref<uint32_t const> child,
      array_ref<uint32_t const> child_count, array_ref<float const> child_size
  );

  // C++ side constructor that converts buffers to potentially smaller arrays
  LinkageTree(
      LinkageTreeWriteView view, LinkageTreeCapsule cap, size_t num_edges
  );

  // Allocate buffers to fill and resize later.
  static std::pair<LinkageTreeWriteView, LinkageTreeCapsule> allocate(
      size_t num_edges
  );

  [[nodiscard]] LinkageTreeView view() const;
  [[nodiscard]] size_t size() const;
};

// --- Function API

LinkageTree compute_linkage_tree(
    SpanningTree mst, size_t num_points,
    std::optional<array_ref<float>> sample_weights
);

#endif  // PLSCAN_API_LINKAGE_H
