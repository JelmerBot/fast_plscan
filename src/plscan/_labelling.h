#ifndef PLSCAN_API_LABELLING_H
#define PLSCAN_API_LABELLING_H

#include <span>

#include "_array.h"

struct LabellingView {
  std::span<int32_t> label;
  std::span<float> probability;

  [[nodiscard]] size_t size() const {
    return label.size();
  }
};

struct Labelling {
  array_ref<int32_t> const label;
  array_ref<float> const probability;

  Labelling() = default;
  Labelling(Labelling &&) = default;
  Labelling(Labelling const &) = default;

  // Python side constructor.
  Labelling(array_ref<int32_t> const label, array_ref<float> const probability)
      : label(label), probability(probability) {}

  // C++ side constructor
  explicit Labelling(size_t const num_points)
      : label(new_array<int32_t>(num_points)),
        probability(new_array<float>(num_points)) {}

  [[nodiscard]] LabellingView view() const {
    return {to_view(label), to_view(probability)};
  }

  [[nodiscard]] size_t size() const {
    return label.size();
  }
};

#endif  // PLSCAN_API_LABELLING_H
