#include "_sparse_graph.h"

#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

// --- Extract core distances

void fill_distances_unsorted(
    std::span<float> distances, SparseGraphView const graph,
    int32_t const min_samples
) {
  nb::gil_scoped_release guard{};
  // Copy data so we can do in-place partitioning
  std::vector<float> data(graph.data.size());
  std::ranges::copy(graph.data, data.begin());
  // clang-format off
  #pragma omp parallel for default(none) shared(graph, data, distances, min_samples)  // clang-format on
  for (int32_t row = 0; row < graph.size(); ++row) {
    int32_t const start = graph.indptr[row];
    if (int32_t const end = graph.indptr[row + 1]; end - start <= min_samples)
      distances[row] = std::numeric_limits<float>::infinity();
    else {
      int32_t const mid = start + min_samples;
      std::nth_element(
          std::next(data.begin(), start), std::next(data.begin(), mid),
          std::next(data.begin(), end)
      );
      distances[row] = data[mid];
    }
  }
}

void fill_distances_sorted(
    std::span<float> distances, SparseGraphView const graph,
    int32_t const min_samples
) {
  nb::gil_scoped_release guard{};
  // clang-format off
  #pragma omp parallel for default(none) shared(graph, min_samples, distances)  // clang-format on
  for (int32_t row = 0; row < graph.size(); ++row)
    if (int32_t const start = graph.indptr[row];
        graph.indptr[row + 1] - start <= min_samples)
      distances[row] = std::numeric_limits<float>::infinity();
    else
      distances[row] = graph.data[start + min_samples];
}

array_ref<float> extract_core_distances(
    SparseGraph graph, int32_t const min_samples, bool const is_sorted
) {
  size_t const num_points = graph.size();
  auto core_distances = new_array<float>(num_points);
  if (is_sorted)
    fill_distances_sorted(to_view(core_distances), graph.view(), min_samples);
  else
    fill_distances_unsorted(to_view(core_distances), graph.view(), min_samples);
  return core_distances;
}

// --- Compute mutual reachability

void apply_core_distances(
    SparseGraphWriteView graph, std::span<float> const core_distances,
    std::span<float> data, std::span<int32_t> indices
) {
  nb::gil_scoped_release guard{};

  std::vector<uint32_t> order(graph.data.size());  // argsort indices
  std::span const order_view(order);

  // clang-format off
  #pragma omp parallel for default(none) shared(graph, core_distances, data, indices, order_view)  // clang-format on
  for (int32_t row = 0; row < graph.size(); ++row) {
    int32_t const start = graph.indptr[row];
    int32_t const end = graph.indptr[row + 1];

    // apply core distances
    float const row_core = core_distances[row];
    for (int32_t idx = start; idx < end; ++idx)
      // Set infinite distance for negative indices (indicating missing edges)
      if (int32_t const col = graph.indices[idx]; col < 0)
        graph.data[idx] = std::numeric_limits<float>::infinity();
      else
        graph.data[idx] =
            std::max({graph.data[idx], row_core, core_distances[col]});

    // fill argsort indices
    auto row_order = order_view.subspan(start, end - start);
    std::iota(row_order.begin(), row_order.end(), start);

    // sort argsort indices (some OpenMPs do not support std::ranges::sort yet)
    std::sort(
        row_order.begin(), row_order.end(),
        [&data = graph.data](uint32_t const a, uint32_t const b) {
          return data[a] < data[b];
        }
    );

    // fill sorted data and indices
    for (int32_t idx = start; idx < end; ++idx) {
      data[idx] = graph.data[order_view[idx]];
      indices[idx] = graph.indices[order_view[idx]];
    }
  }
}

SparseGraph compute_mutual_reachability(
    SparseGraph graph, array_ref<float> const core_distances
) {
  array_ref<float> const data = new_array<float>(graph.data.size());
  array_ref<int32_t> const indices = new_array<int32_t>(graph.indices.size());
  auto [graph_view, graph_cap] = SparseGraph::allocate_copy(graph);
  apply_core_distances(
      graph_view, to_view(core_distances), to_view(data), to_view(indices)
  );
  return SparseGraph{
      array_ref<float const>{data}, array_ref<int32_t const>{indices},
      graph.indptr
  };
}

// --- Module definitions

NB_MODULE(_sparse_graph_ext, m) {
  m.doc() = "Module for representing sparse CSR graph.";

  nb::class_<SparseGraph>(m, "SparseGraph")
      .def(
          "__init__",
          [](SparseGraph *t, nb::handle data, nb::handle indices,
             nb::handle indptr) {
            // Support np.memmap and np.ndarray input types for sklearn
            // pickling. The output of np.asarray can cast to nanobind ndarrays.
            auto const asarray = nb::module_::import_("numpy").attr("asarray");
            new (t) SparseGraph(
                nb::cast<array_ref<float const>>(asarray(data), false),
                nb::cast<array_ref<int32_t const>>(asarray(indices), false),
                nb::cast<array_ref<int32_t const>>(asarray(indptr), false)
            );
          },
          nb::arg("data"), nb::arg("indices"), nb::arg("indptr"),
          nb::sig(
              "def __init__(self, data: np.ndarray[tuple[int], "
              "np.dtype[np.float32]], indices: np.ndarray[tuple[int], "
              "np.dtype[np.int32]], indptr: np.ndarray[tuple[int], "
              "np.dtype[np.int32]]) -> None"
          ),
          R"(
          Parameters
          ----------
          data
              An array of distances.
          indices
              An array of column indices.
          indptr
              The CSR indptr array.
          )"
      )
      .def_ro(
          "data", &SparseGraph::data, nb::rv_policy::reference,
          "A 1D array with data values."
      )
      .def_ro(
          "indices", &SparseGraph::indices, nb::rv_policy::reference,
          "A 1D array with indices values."
      )
      .def_ro(
          "indptr", &SparseGraph::indptr, nb::rv_policy::reference,
          "A 1D array with indptr values."
      )
      .def(
          "__iter__",
          [](SparseGraph const &self) {
            return nb::make_tuple(self.data, self.indices, self.indptr)
                .attr("__iter__")();
          }
      )
      .def(
          "__reduce__",
          [](SparseGraph const &self) {
            return nb::make_tuple(
                nb::type<SparseGraph>(),
                nb::make_tuple(self.data, self.indices, self.indptr)
            );
          }
      )
      .doc() = "SparseGraph contains a (square) distance matrix in CSR format.";

  m.def(
      "extract_core_distances", &extract_core_distances, nb::arg("graph"),
      nb::arg("min_samples") = 5, nb::arg("is_sorted") = false,
      R"(
          Extracts core distances from a sparse graph.

          Parameters
          ----------
          graph
                The sparse graph to extract core distances from.
          min_samples
                The number of nearest neighbors to consider for core distance.
          is_sorted
                Whether the rows of the graph are sorted by distance.

          Returns
          -------
          core_distances
                An array of core distances.
        )"
  );

  m.def(
      "compute_mutual_reachability", &compute_mutual_reachability,
      nb::arg("graph"), nb::arg("core_distances"),
      R"(
          Applies core distances to a sparse graph to compute mutual
          reachability.

          Parameters
          ----------
          graph
                The sparse graph to extract core distances from.
          core_distances
                An array of core distances, one for each point in the graph.

          Returns
          -------
          mutual_graph
                A new sparse graph with mutual reachability distances. Rows are
                sorted by mutual reachability distance.
        )"
  );
}