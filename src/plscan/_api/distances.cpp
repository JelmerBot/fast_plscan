#include "distances.h"

#include <cstring>
#include <map>

#include "array.h"

// --- Implementation details

namespace {

struct strless {
  bool operator()(char const *const a, char const *const b) const {
    return std::strcmp(a, b) < 0;
  }
};

}  // namespace

// --- Function API

std::underlying_type_t<Metric> parse_metric(char const *const metric) {
  static std::map<char const *const, Metric, strless> const metric_map = {
      {"l2", Metric::Euclidean},
      {"euclidean", Metric::Euclidean},
      {"l1", Metric::Cityblock},
      {"manhattan", Metric::Cityblock},
      {"cityblock", Metric::Cityblock},
      {"chebyshev", Metric::Chebyshev},
      {"infinity", Metric::Chebyshev},
      {"p", Metric::Minkowski},
      {"minkowski", Metric::Minkowski},
      {"hamming", Metric::Hamming},
      {"braycurtis", Metric::Braycurtis},
      {"canberra", Metric::Canberra},
      {"haversine", Metric::Haversine},
      {"seuclidean", Metric::SEuclidean},
      {"mahalanobis", Metric::Mahalanobis},
      {"dice", Metric::Dice},
      {"jaccard", Metric::Jaccard},
      {"russellrao", Metric::Russellrao},
      {"rogerstanimoto", Metric::Rogerstanimoto},
      {"sokalsneath", Metric::Sokalsneath}
  };

  auto const it = metric_map.find(metric);
  if (it == metric_map.end())
    throw nb::value_error(  //
        nb::str("Unsupported metric: {}").format(metric).c_str()
    );

  return static_cast<std::underlying_type_t<Metric>>(it->second);
}
