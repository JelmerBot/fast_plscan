#ifndef PLSCAN_API_SPARSE_GRAPH_
#define PLSCAN_API_SPARSE_GRAPH_

#include <span>

#include "_array.h"

// Non-owning view of a csr graph
struct SparseGraphView {
  std::span<float> data;
  std::span<int32_t> indices;
  std::span<int32_t> indptr;

  [[nodiscard]] size_t size() const {
    return indptr.size() - 1u;
  }
};

// Sparse (square) distance matrix in compressed sparse row (CSR) format.
struct SparseGraph {
  array_ref<float> const data;
  array_ref<int32_t> const indices;
  array_ref<int32_t> const indptr;

  [[nodiscard]] SparseGraphView view() const {
    return {to_view(data), to_view(indices), to_view(indptr)};
  }

  [[nodiscard]] size_t size() const {
    return indptr.size() - 1u; // num points in the matrix!
  }
};

#endif  // PLSCAN_API_SPARSE_GRAPH_
