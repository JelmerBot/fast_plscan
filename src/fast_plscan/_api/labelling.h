#ifndef PLSCAN_API_LABELLING_H
#define PLSCAN_API_LABELLING_H

#include "array.h"
#include "condensed_tree.h"
#include "leaf_tree.h"

// Ownership for a LabellingWriteView.
struct LabellingCapsule {
  nb::capsule label;
  nb::capsule probability;
};

// Non-owning view of a labelling
struct LabellingWriteView {
  std::span<int64_t> const label;
  std::span<float> const probability;

  [[nodiscard]] size_t size() const;
};

// Non-owning view of a labelling
struct LabellingView {
  std::span<int64_t const> const label;
  std::span<float const> const probability;

  [[nodiscard]] size_t size() const;
};

struct Labelling {
  array_ref<int64_t const> label;
  array_ref<float const> probability;

  Labelling() = default;
  Labelling(Labelling &&) = default;
  Labelling(Labelling const &) = default;
  Labelling &operator=(Labelling &&) = default;
  Labelling &operator=(Labelling const &) = default;

  // Python side constructor.
  Labelling(array_ref<int32_t const> label, array_ref<float const> probability);

  // C++ side constructor
  Labelling(LabellingWriteView view, LabellingCapsule cap);

  // Allocate buffers to fill later.
  static std::pair<LabellingWriteView, LabellingCapsule> allocate(
      size_t num_points
  );

  [[nodiscard]] LabellingView view() const;
  [[nodiscard]] size_t size() const;
};

// --- Function API

Labelling compute_cluster_labels(
    LeafTree leaf_tree, CondensedTree condensed_tree,
    array_ref<uint32_t const> selected_clusters, size_t num_points
);

#endif  // PLSCAN_API_LABELLING_H
