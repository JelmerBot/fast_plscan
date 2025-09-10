#ifndef PLSCAN_API_LABELLING_H
#define PLSCAN_API_LABELLING_H

#include <span>

#include "_array.h"

// Ownership for a LabellingWriteView.
struct LabellingCapsule {
  nb::capsule label;
  nb::capsule probability;
};

// Non-owning view of a labelling
struct LabellingWriteView {
  std::span<int64_t> const label;
  std::span<float> const probability;

  [[nodiscard]] size_t size() const {
    return label.size();
  }
};

// Non-owning view of a labelling
struct LabellingView {
  std::span<int64_t const> const label;
  std::span<float const> const probability;

  [[nodiscard]] size_t size() const {
    return label.size();
  }
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
  Labelling(
      array_ref<int32_t const> const label,
      array_ref<float const> const probability
  )
      : label(label), probability(probability) {}

  // C++ side constructor
  Labelling(LabellingWriteView const view, LabellingCapsule cap)
      : label(to_array(view.label, std::move(cap.label), view.label.size())),
        probability(to_array(
            view.probability, std::move(cap.probability),
            view.probability.size()
        )) {}

  // Allocate buffers to fill later.
  static auto allocate(size_t const num_points) {
    auto [label, label_cap] = new_buffer<int64_t>(num_points);
    auto [prob, prob_cap] = new_buffer<float>(num_points);
    return std::make_pair(
        LabellingWriteView{label, prob},
        LabellingCapsule{std::move(label_cap), std::move(prob_cap)}
    );
  }

  [[nodiscard]] LabellingView view() const {
    return {to_view(label), to_view(probability)};
  }

  [[nodiscard]] size_t size() const {
    return label.size();
  }
};

#endif  // PLSCAN_API_LABELLING_H
