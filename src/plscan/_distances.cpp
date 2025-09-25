#include "_distances.h"

#include <array>

#include "_array.h"

template <Metric metric>
nb::object wrap_dist(nb::dict const metric_kws) {
  return nb::cpp_function([dist = get_dist<metric>(metric_kws)](
                              array_ref<float const> const x,
                              array_ref<float const> const y
                          ) { return dist(to_view(x), to_view(y)); });
}

NB_MODULE(_distances, m) {
  m.doc() = "Module for distance computation in PLSCAN.";
  m.def(
      "get_dist",
      [](char const *const metric, nb::kwargs const metric_kws) {
        // Must match Metric enumeration order!
        constexpr std::array lookup{
            wrap_dist<Metric::Euclidean>,   wrap_dist<Metric::Cityblock>,
            wrap_dist<Metric::Chebyshev>,   wrap_dist<Metric::Minkowski>,
            wrap_dist<Metric::Hamming>,     wrap_dist<Metric::Braycurtis>,
            wrap_dist<Metric::Canberra>,    wrap_dist<Metric::Haversine>,
            wrap_dist<Metric::SEuclidean>,  wrap_dist<Metric::Mahalanobis>,
            wrap_dist<Metric::Dice>,        wrap_dist<Metric::Jaccard>,
            wrap_dist<Metric::Russellrao>,  wrap_dist<Metric::Rogerstanimoto>,
            wrap_dist<Metric::Sokalsneath>,
        };

        auto const idx = parse_metric(metric);
        if (idx >= lookup.size())
          throw nb::value_error(  //
              nb::str("Missing python wrapper for '{}'").format(metric).c_str()
          );

        return lookup[idx](metric_kws);
      },
      nb::arg("metric"), nb::arg("metric_kws"),
      R"(
        Retrieves the specified distance metric callback.

        Parameters
        ----------
        metric
          The name of the metric to use. See :py:attr:`~plscan.PLSCAN.VALID_BALLTREE_METRICS` 
          for a list of valid metrics.
        **metric_kws
          p: The order of the Minkowski distance. Required if `metric` is "minkowski".
          V: The variance vector for the standardized Euclidean distance. Required if 
          `metric` is "seuclidean".
          VI: The inverse covariance matrix for the Mahalanobis distance. Required if 
          `metric` is "mahalanobis".

        Returns
        -------
        dist
            The distance function callback. Its input arrays must be c-contiguous.
      )"
  );
}
