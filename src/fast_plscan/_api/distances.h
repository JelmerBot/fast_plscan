#ifndef PLSCAN_API_DISTANCES_H
#define PLSCAN_API_DISTANCES_H

#include <cmath>
#include <vector>

#include "array.h"

// --- Supported metrics

// Enum value is used as index into lookup arrays. Do not change order without
// updating all lookup sites. All KDTree metrics must be listed first, followed
// by all BallTree metrics, and finally non-tree metrics.
enum class Metric {
  Euclidean,
  Cityblock,
  Chebyshev,
  Minkowski,  // boundary for KDTree metrics
  Hamming,
  Braycurtis,
  Canberra,
  Haversine,
  SEuclidean,
  Mahalanobis,
  Dice,
  Jaccard,
  Russellrao,
  Rogerstanimoto,
  Sokalsneath  // boundary for BallTree metrics
};

template <Metric metric>
concept KDTreeMetric = metric <= Metric::Minkowski;

template <Metric metric>
concept BallTreeMetric = metric <= Metric::Sokalsneath;

// --- Parameter pack helpers

auto first_in_pack(auto &first, auto &...) {
  return first;
}

// --- Function API

std::underlying_type_t<Metric> parse_metric(char const *metric);

// Minkowski distance implementations

template <Metric metric, typename... Args>
  requires KDTreeMetric<metric>
void apply_minkowski(float &rdist, float const diff, Args... args) {
  if constexpr (metric == Metric::Euclidean)
    rdist += diff * diff;
  else if constexpr (metric == Metric::Cityblock)
    rdist += std::abs(diff);
  else if constexpr (metric == Metric::Chebyshev)
    rdist = std::max(std::abs(diff), rdist);
  else  // Metric::Minkowski
    rdist += std::pow(std::abs(diff), first_in_pack(args...));
}

// Rdist overloads for different metrics

template <Metric metric, typename... Args>
  requires KDTreeMetric<metric>
float rdist(
    std::span<float const> const a, std::span<float const> const b, Args... args
) {
  float rdist = 0.0f;
  for (size_t col = 0; col < a.size(); ++col)
    apply_minkowski<metric>(rdist, a[col] - b[col], args...);
  return rdist;
}

template <Metric metric>
  requires(metric == Metric::SEuclidean)
float rdist(
    std::span<float const> const a, std::span<float const> const b,
    std::span<float const> const V
) {
  float rdist = 0.0f;
  for (size_t col = 0; col < a.size(); ++col) {
    float const diff = a[col] - b[col];
    rdist += diff * diff / V[col];
  }
  return rdist;
}

template <Metric metric>
  requires(metric == Metric::Mahalanobis)
float rdist(
    std::span<float const> const a, std::span<float const> const b,
    nb::ndarray_view<float const, 2, 'C'> const VI
) {
  std::vector<float> buffer(a.size());
  for (size_t col = 0; col < a.size(); ++col)
    buffer[col] = a[col] - b[col];

  float rdist = 0.0f;
  for (size_t row = 0; row < a.size(); ++row) {
    float value = 0.0f;
    for (size_t col = 0; col < a.size(); ++col)
      value += VI(row, col) * buffer[col];
    rdist += value * buffer[row];
  }
  return rdist;
}

template <Metric metric>
  requires(metric == Metric::Hamming)
float rdist(std::span<float const> const a, std::span<float const> const b) {
  float num_differences = 0.0f;
  for (size_t col = 0; col < a.size(); ++col)
    if (a[col] != b[col])
      ++num_differences;
  return num_differences / static_cast<float>(a.size());
}

template <Metric metric>
  requires(metric == Metric::Braycurtis)
float rdist(std::span<float const> const a, std::span<float const> const b) {
  float diffs = 0.0f;
  float sums = 0.0f;
  for (size_t col = 0; col < a.size(); ++col) {
    diffs += std::abs(a[col] - b[col]);
    // sklearn uses this definition for their balltree!
    sums += std::abs(a[col]) + std::abs(b[col]);
  }
  return sums <= 0.0f ? 0.0f : diffs / sums;
}

template <Metric metric>
  requires(metric == Metric::Canberra)
float rdist(std::span<float const> const a, std::span<float const> const b) {
  float dist = 0.0f;
  for (size_t col = 0; col < a.size(); ++col)
    if (float const sums = std::abs(a[col]) + std::abs(b[col]); sums != 0.0f)
      dist += std::abs(a[col] - b[col]) / sums;
  return dist;
}

template <Metric metric>
  requires(metric == Metric::Haversine)
float rdist(std::span<float const> const a, std::span<float const> const b) {
  float const sin_0 = std::sin((a[0] - b[0]) * 0.5f);
  float const sin_1 = std::sin((a[1] - b[1]) * 0.5f);
  return sin_0 * sin_0 + std::cos(a[0]) * std::cos(b[0]) * sin_1 * sin_1;
}

template <Metric metric>
  requires(metric == Metric::Dice)
float rdist(std::span<float const> const a, std::span<float const> const b) {
  float num_diff = 0;
  float num_both_true = 0;
  for (size_t col = 0; col < a.size(); ++col) {
    bool const a_true = a[col] != 0.0f;
    bool const b_true = b[col] != 0.0f;
    num_diff += static_cast<float>(a_true != b_true);
    num_both_true += static_cast<float>(a_true and b_true);
  }
  return num_diff / (2.0f * num_both_true + num_diff);
}

template <Metric metric>
  requires(metric == Metric::Jaccard)
float rdist(std::span<float const> const a, std::span<float const> const b) {
  float num_or = 0;
  float num_and = 0;
  for (size_t col = 0; col < a.size(); ++col) {
    bool const a_true = a[col] != 0.0f;
    bool const b_true = b[col] != 0.0f;
    num_or += static_cast<float>(a_true or b_true);
    num_and += static_cast<float>(a_true and b_true);
  }
  return (num_or - num_and) / num_or;
}

template <Metric metric>
  requires(metric == Metric::Russellrao)
float rdist(std::span<float const> const a, std::span<float const> const b) {
  float num_both = 0.0f;
  for (size_t col = 0; col < a.size(); ++col)
    num_both += static_cast<float>(a[col] != 0.0f and b[col] != 0.0f);

  auto const size = static_cast<float>(a.size());
  return (size - num_both) / size;
}

template <Metric metric>
  requires(metric == Metric::Rogerstanimoto)
float rdist(std::span<float const> const a, std::span<float const> const b) {
  float num_diff = 0.0f;
  for (size_t col = 0; col < a.size(); ++col) {
    bool const a_true = a[col] != 0.0f;
    bool const b_true = b[col] != 0.0f;
    num_diff += static_cast<float>(a_true != b_true);
  }

  auto const size = static_cast<float>(a.size());
  return 2.0f * num_diff / (size + num_diff);
}

template <Metric metric>
  requires(metric == Metric::Sokalsneath)
float rdist(std::span<float const> const a, std::span<float const> const b) {
  float num_diff = 0.0f;
  float num_both = 0.0f;
  for (size_t col = 0; col < a.size(); ++col) {
    bool const a_true = a[col] != 0.0f;
    bool const b_true = b[col] != 0.0f;
    num_diff += static_cast<float>(a_true != b_true);
    num_both += static_cast<float>(a_true and b_true);
  }
  return num_diff / (0.5f * num_both + num_diff);
}

// Getter for (r)dist functions dealing with potential keyword arguments

template <Metric metric>
concept SquaredMetrics = metric == Metric::Euclidean or
                         metric == Metric::SEuclidean or
                         metric == Metric::Mahalanobis;

template <Metric metric>
auto get_rdist(nb::dict const metric_kws) {
  if constexpr (metric == Metric::Minkowski)
    return [p = nb::cast<float>(metric_kws["p"])](
               std::span<float const> const a, std::span<float const> const b
           ) { return rdist<metric>(a, b, p); };
  else if constexpr (metric == Metric::SEuclidean)
    return  //
        [V = to_view(nb::cast<array_ref<float const>>(metric_kws["V"], false))](
            std::span<float const> const a, std::span<float const> const b
        ) { return rdist<Metric::SEuclidean>(a, b, V); };
  else if constexpr (metric == Metric::Mahalanobis)
    return
        [VI = nb::cast<ndarray_ref<float const, 2>>(metric_kws["VI"]).view()](
            std::span<float const> const a, std::span<float const> const b
        ) { return rdist<Metric::Mahalanobis>(a, b, VI); };
  else
    return rdist<metric>;
}

template <Metric metric>
auto get_dist_to_rdist(nb::dict const metric_kws) {
  if constexpr (SquaredMetrics<metric>)
    return [](float const dist) { return dist * dist; };
  else if constexpr (metric == Metric::Minkowski)
    return [p = nb::cast<float>(metric_kws["p"])](float const dist) {
      return std::pow(dist, p);
    };
  else if constexpr (metric == Metric::Haversine)
    return [](float const dist) {
      float const sin_half = std::sin(dist * 0.5f);
      return sin_half * sin_half;
    };
  else
    // The compiler optimizes this to a no-op!
    return [](float const dist) { return dist; };
}

template <Metric metric>
auto get_rdist_to_dist(nb::dict const metric_kws) {
  if constexpr (SquaredMetrics<metric>)
    return [](float const dist) { return std::sqrt(dist); };
  else if constexpr (metric == Metric::Minkowski)
    return [p = nb::cast<float>(metric_kws["p"])](float const dist) {
      return std::pow(dist, 1.0f / p);
    };
  else if constexpr (metric == Metric::Haversine)
    return [](float const dist) { return 2 * std::asin(std::sqrt(dist)); };
  else
    // The compiler optimizes this to a no-op!
    return [](float const dist) { return dist; };
}

template <Metric metric>
auto get_dist(nb::dict const metric_kws) {
  return [rdist_fun = get_rdist<metric>(metric_kws),
          to_dist = get_rdist_to_dist<metric>(metric_kws)](
             std::span<float const> const a, std::span<float const> const b
         ) { return to_dist(rdist_fun(a, b)); };
}

#endif  // PLSCAN_API_DISTANCES_H
