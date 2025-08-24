#include "_distances.h"

#include "_array.h"

template <Metric metric>
nb::object wrap_dist(nb::dict const metric_kws) {
  return nb::cpp_function([dist = get_dist<metric>(metric_kws)](
                              array_ref<float const> const x,
                              array_ref<float const> const y
                          ) { return dist(to_view(x), to_view(y)); });
}

NB_MODULE(_distances_ext, m) {
  m.doc() = "Module for distance computation in PLSCAN.";
  m.def(
      "get_dist",
      [](char const *const metric, nb::kwargs const metric_kws) {
        static std::map<Metric, nb::object (*)(nb::dict)> lookup = {
            {Metric::Euclidean, wrap_dist<Metric::Euclidean>},
            {Metric::Cityblock, wrap_dist<Metric::Cityblock>},
            {Metric::Chebyshev, wrap_dist<Metric::Chebyshev>},
            {Metric::Minkowski, wrap_dist<Metric::Minkowski>},
            {Metric::Hamming, wrap_dist<Metric::Hamming>},
            {Metric::Braycurtis, wrap_dist<Metric::Braycurtis>},
            {Metric::Canberra, wrap_dist<Metric::Canberra>},
            {Metric::Haversine, wrap_dist<Metric::Haversine>},
            {Metric::SEuclidean, wrap_dist<Metric::SEuclidean>},
            {Metric::Mahalanobis, wrap_dist<Metric::Mahalanobis>},
            {Metric::Dice, wrap_dist<Metric::Dice>},
            {Metric::Jaccard, wrap_dist<Metric::Jaccard>},
            {Metric::Russellrao, wrap_dist<Metric::Russellrao>},
            {Metric::Rogerstanimoto, wrap_dist<Metric::Rogerstanimoto>},
            {Metric::Sokalsneath, wrap_dist<Metric::Sokalsneath>}
        };
        if (auto const it = lookup.find(parse_metric(metric));
            it != lookup.end())
          return it->second(metric_kws);

        throw nb::value_error(  //
            nb::str("Unsupported metric: {}").format(metric).c_str()
        );
      },
      nb::arg("metric"), nb::arg("metric_kws"),
      nb::sig(                //
          "def get_dist("     //
          " metric: str, *,"  //
          " p: float,"        //
          " V: np.ndarray[tuple[int], np.dtype[np.float3//2]],"
          " VI: np.ndarray[tuple[int, int], np.dtype[np.//float32]]"
          ") -> typing.Callable[["  //
          " np.ndarray[tuple[int], np.dtype[np.float32]]//,"
          " np.ndarray[tuple[int], np.dtype[np.float32]]//"
          "], float]"  //
      ),
      R"(
        Retrieves the specified distance metric callback.

        Parameters
        ----------
        metric
          The name of the metric to use. See :py:attr:`~plscan.PLSCAN.VALID_BALLTREE_METRICS` 
          for a list of valid metrics.
        p
          The order of the Minkowski distance. Required if `metric` is "minkowski".
        V
          The variance vector for the standardized Euclidean distance. Required
          if `metric` is "seuclidean".
        VI
          The inverse covariance matrix for the Mahalanobis distance. Required
          if `metric` is "mahalanobis".

        Returns
        -------
        dist
            The distance function callback. Its input arrays must be c-contiguous.
      )"
  );
}
