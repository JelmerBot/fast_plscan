#include "_distances.h"

#include "_array.h"

NB_MODULE(_distances, m) {
  m.doc() = "Module for distance computation in PLSCAN.";

  m.def(
      "euclidean",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Euclidean>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      R"(
        Euclidean distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector.
        v
            The second input vector.

        Returns
        -------
        distance
            The Euclidean distance between the two vectors.
        )"
  );

  m.def(
      "cityblock",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Cityblock>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      R"(
        Cityblock distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector.
        v
            The second input vector.

        Returns
        -------
        distance
            The Cityblock distance between the two vectors.
        )"
  );

  m.def(
      "chebyshev",
      [](array_ref<float> const u, array_ref<float> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Chebyshev>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      R"(
        Chebyshev distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector.
        v
            The second input vector.

        Returns
        -------
        distance
            The Chebyshev distance between the two vectors.
        )"
  );

  m.def(
      "minkowski",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Minkowski>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      R"(
        Minkowski distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector.
        v
            The second input vector.
        p
            The order of the Minkowski distance. Keyword only!

        Returns
        -------
        distance
            The Minkowski distance between the two vectors.
        )"
  );

  m.def(
      "hamming",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Hamming>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      R"(
        Hamming distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector.
        v
            The second input vector.

        Returns
        -------
        distance
            The Hamming distance between the two vectors.
        )"
  );

  m.def(
      "braycurtis",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Braycurtis>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      R"(
        Braycurtis distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector.
        v
            The second input vector.

        Returns
        -------
        distance
            The Braycurtis distance between the two vectors.
        )"
  );

  m.def(
      "canberra",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Canberra>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      R"(
        Canberra distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector.
        v
            The second input vector.

        Returns
        -------
        distance
            The Canberra distance between the two vectors.
        )"
  );
  m.def(
      "haversine",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Haversine>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      R"(
        Haversine distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector. Only the first
            two values are used, which must be latitude and longitude in
            radians.
        v
            The second input vector.Only the first
            two values are used, which must be latitude and longitude in
            radians.

        Returns
        -------
        distance
            The Haversine distance between the two vectors.
        )"
  );

  m.def(
      "seuclidean",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::SEuclidean>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      R"(
        SEuclidean distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector.
        v
            The second input vector.
        V
            The variance vector for the SEuclidean distance. This is used to scale 
            the distance based on the variance of the coordinates.

        Returns
        -------
        distance
            The SEuclidean distance between the two vectors.
        )"
  );

  m.def(
      "mahalanobis",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Mahalanobis>(metric_kws)(
            to_view(u), to_view(v)
        );
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      R"(
        Mahalanobis distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector.
        v
            The second input vector.
        VI
            The inverse covariance matrix for the Mahalanobis distance.

        Returns
        -------
        distance : float
            The Mahalanobis distance between the two vectors.
        )"
  );

  m.def(
      "dice",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Dice>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      R"(
        Dice distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector.
        v
            The second input vector.

        Returns
        -------
        distance
            The Dice distance between the two vectors.
        )"
  );

  m.def(
      "jaccard",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Jaccard>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      R"(
        Jaccard distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector.
        v
            The second input vector.

        Returns
        -------
        distance
            The Jaccard distance between the two vectors.
        )"
  );

  m.def(
      "russellrao",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Russellrao>(metric_kws)(to_view(u), to_view(v));
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      R"(
        Russellrao distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector.
        v
            The second input vector.

        Returns
        -------
        distance
            The Russellrao distance between the two vectors.
        )"
  );

  m.def(
      "rogerstanimoto",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Rogerstanimoto>(metric_kws)(
            to_view(u), to_view(v)
        );
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      R"(
        Rogerstanimoto distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector.
        v
            The second input vector.

        Returns
        -------
        distance
            The Rogerstanimoto distance between the two vectors.
        )"
  );

  m.def(
      "sokalsneath",
      [](array_ref<float const> const u, array_ref<float const> const v,
         nb::kwargs const metric_kws) {
        return get_dist<Metric::Sokalsneath>(metric_kws)(
            to_view(u), to_view(v)
        );
      },
      nb::arg("u").noconvert(), nb::arg("v").noconvert(), nb::arg("metric_kws"),
      R"(
        Sokalsneath distance between two 1D vectors.

        Parameters
        ----------
        u
            The first input vector.
        v
            The second input vector.

        Returns
        -------
        distance
            The Sokalsneath distance between the two vectors.
        )"
  );
}