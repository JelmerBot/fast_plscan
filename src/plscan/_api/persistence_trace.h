#ifndef PLSCAN_PERSISTENCE_TRACE_H
#define PLSCAN_PERSISTENCE_TRACE_H

#include <vector>

#include "array.h"
#include "condensed_tree.h"
#include "leaf_tree.h"

// Ownership for a PersistenceTraceView.
struct PersistenceTraceCapsule {
  nb::capsule min_size;
  nb::capsule persistence;
};

// Non-owning view of a leaf tree
struct PersistenceTraceWriteView {
  std::span<float> const min_size;
  std::span<float> const persistence;

  [[nodiscard]] size_t size() const;
};

// Non-owning view of a leaf tree
struct PersistenceTraceView {
  std::span<float const> const min_size;
  std::span<float const> const persistence;

  [[nodiscard]] size_t size() const;
};

struct PersistenceTrace {
  array_ref<float const> min_size;
  array_ref<float const> persistence;

  PersistenceTrace() = default;
  PersistenceTrace(PersistenceTrace &&) = default;
  PersistenceTrace(PersistenceTrace const &) = default;
  PersistenceTrace &operator=(PersistenceTrace &&) = default;
  PersistenceTrace &operator=(PersistenceTrace const &) = default;

  // Python side constructor.
  PersistenceTrace(
      array_ref<float const> min_size, array_ref<float const> persistence
  );

  // C++ side constructor that converts buffers to potentially smaller arrays.
  PersistenceTrace(
      PersistenceTraceWriteView view, PersistenceTraceCapsule cap,
      size_t num_traces
  );

  // Allocate buffers to fill and resize later.
  static std::pair<PersistenceTraceWriteView, PersistenceTraceCapsule> allocate(
      size_t num_traces
  );

  [[nodiscard]] PersistenceTraceView view() const;
  [[nodiscard]] size_t size() const;
};

// --- Function API

PersistenceTrace compute_size_persistence(LeafTree leaf_tree);

PersistenceTrace compute_size_distance_bi_persistence(
    LeafTree leaf_tree, CondensedTree condensed_tree, size_t num_points
);

PersistenceTrace compute_size_density_bi_persistence(
    LeafTree leaf_tree, CondensedTree condensed_tree, size_t num_points
);

std::pair<std::vector<array_ref<float>>, std::vector<array_ref<float>>>
compute_distance_icicles(
    LeafTree leaf_tree, CondensedTree condensed_tree, size_t num_points
);

std::pair<std::vector<array_ref<float>>, std::vector<array_ref<float>>>
compute_density_icicles(
    LeafTree leaf_tree, CondensedTree condensed_tree, size_t num_points
);

#endif  // PLSCAN_PERSISTENCE_TRACE_H
