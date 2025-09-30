#include "sparse_graph.h"

#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

// --- Implementation details

namespace {

// Extract core distances

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

// Compute mutual reachability

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

}  // namespace

// --- Function API

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

// --- Class API

SparseGraph::SparseGraph(
    array_ref<float const> const &data, array_ref<int32_t const> const &indices,
    array_ref<int32_t const> const &indptr
)
    : data(data), indices(indices), indptr(indptr) {}

SparseGraph::SparseGraph(
    SparseGraphWriteView const view, SparseGraphCapsule cap
)
    : data(to_array(view.data, std::move(cap.data), view.data.size())),
      indices(
          to_array(view.indices, std::move(cap.indices), view.indices.size())
      ),
      indptr(  //
          to_array(view.indptr, std::move(cap.indptr), view.indptr.size())
      ) {}

std::pair<SparseGraphWriteView, SparseGraphCapsule> SparseGraph::allocate_knn(
    size_t const num_points, size_t const num_neighbors
) {
  auto [data, data_cap] = new_buffer<float>(num_points * num_neighbors);
  auto [indices, indices_cap] = new_buffer<int32_t>(num_points * num_neighbors);
  auto [indptr, indptr_cap] = new_buffer<int32_t>(num_points + 1);

  std::ranges::fill(data, std::numeric_limits<float>::infinity());
  std::ranges::fill(indices, -1);
  std::ranges::transform(
      std::views::iota(0ul, num_points + 1ul), indptr.begin(),
      [num_neighbors](int32_t const i) { return i * num_neighbors; }
  );

  return std::make_pair(
      SparseGraphWriteView{data, indices, indptr},
      SparseGraphCapsule{
          std::move(data_cap), std::move(indices_cap), std::move(indptr_cap)
      }
  );
}

std::pair<SparseGraphWriteView, SparseGraphCapsule> SparseGraph::allocate_copy(
    SparseGraph const &graph
) {
  auto [data, data_cap] = new_buffer<float>(graph.data.size());
  auto [indices, indices_cap] = new_buffer<int32_t>(graph.indices.size());
  auto [indptr, indptr_cap] = new_buffer<int32_t>(graph.indptr.size());

  auto [data_view, indices_view, indptr_view] = graph.view();
  std::ranges::copy(data_view, data.begin());
  std::ranges::copy(indices_view, indices.begin());
  std::ranges::copy(indptr_view, indptr.begin());

  return std::make_pair(
      SparseGraphWriteView{data, indices, indptr},
      SparseGraphCapsule{
          std::move(data_cap), std::move(indices_cap), std::move(indptr_cap)
      }
  );
}

SparseGraphView SparseGraph::view() const {
  return {to_view(data), to_view(indices), to_view(indptr)};
}

size_t SparseGraphWriteView::size() const {
  return indptr.size() - 1u;
}

size_t SparseGraphView::size() const {
  return indptr.size() - 1u;
}

size_t SparseGraph::size() const {
  return indptr.size() - 1u;  // num points in the matrix!
}
