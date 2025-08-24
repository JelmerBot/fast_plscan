#ifndef PLSCAN_API_SPARSE_GRAPH_
#define PLSCAN_API_SPARSE_GRAPH_

#include <algorithm>
#include <ranges>
#include <span>

#include "_array.h"

// Ownership for a SparseGraphWriteView.
struct SparseGraphCapsule {
  nb::capsule data;
  nb::capsule indices;
  nb::capsule indptr;
};

// Non-owning view of a csr graph
struct SparseGraphWriteView {
  std::span<float> const data;
  std::span<int32_t> const indices;
  std::span<int32_t> const indptr;

  [[nodiscard]] size_t size() const {
    return indptr.size() - 1u;
  }
};

// Non-owning view of a csr graph
struct SparseGraphView {
  std::span<float const> const data;
  std::span<int32_t const> const indices;
  std::span<int32_t const> const indptr;

  [[nodiscard]] size_t size() const {
    return indptr.size() - 1u;
  }
};

// Sparse (square) distance matrix in compressed sparse row (CSR) format.
struct SparseGraph {
  array_ref<float const> data;
  array_ref<int32_t const> indices;
  array_ref<int32_t const> indptr;

  SparseGraph() = default;
  SparseGraph(SparseGraph &&) = default;
  SparseGraph(SparseGraph const &) = default;
  SparseGraph &operator=(SparseGraph &&) = default;
  SparseGraph &operator=(SparseGraph const &) = default;

  // Python-side constructor.
  SparseGraph(
      array_ref<float const> const &data,
      array_ref<int32_t const> const &indices,
      array_ref<int32_t const> const &indptr
  )
      : data(data), indices(indices), indptr(indptr) {}

  // C++-side constructor.
  SparseGraph(SparseGraphWriteView const view, SparseGraphCapsule cap)
      : data(to_array(view.data, std::move(cap.data), view.data.size())),
        indices(
            to_array(view.indices, std::move(cap.indices), view.indices.size())
        ),
        indptr(  //
            to_array(view.indptr, std::move(cap.indptr), view.indptr.size())
        ) {}

  // Allocate a knn graph to fill later.
  static auto allocate_knn(
      size_t const num_points, size_t const num_neighbors
  ) {
    auto [data, data_cap] = new_buffer<float>(num_points * num_neighbors);
    auto [indices, indices_cap] =
        new_buffer<int32_t>(num_points * num_neighbors);
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

  // Allocate a mutable copy from an existing SparseGraph
  static auto allocate_copy(SparseGraph const &graph) {
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

  [[nodiscard]] SparseGraphView view() const {
    return {to_view(data), to_view(indices), to_view(indptr)};
  }

  [[nodiscard]] size_t size() const {
    return indptr.size() - 1u;  // num points in the matrix!
  }
};

#endif  // PLSCAN_API_SPARSE_GRAPH_
