#ifndef PLSCAN_API_SPARSE_GRAPH_
#define PLSCAN_API_SPARSE_GRAPH_

#include "array.h"

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

  [[nodiscard]] size_t size() const;
};

// Non-owning view of a csr graph
struct SparseGraphView {
  std::span<float const> const data;
  std::span<int32_t const> const indices;
  std::span<int32_t const> const indptr;

  [[nodiscard]] size_t size() const;
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
  );

  // C++-side constructor.
  SparseGraph(SparseGraphWriteView view, SparseGraphCapsule cap);

  // Allocate a knn graph to fill later.
  static std::pair<SparseGraphWriteView, SparseGraphCapsule> allocate_knn(
      size_t num_points, size_t num_neighbors
  );

  // Allocate a mutable copy from an existing SparseGraph
  static std::pair<SparseGraphWriteView, SparseGraphCapsule> allocate_copy(
      SparseGraph const &graph
  );

  [[nodiscard]] SparseGraphView view() const;
  [[nodiscard]] size_t size() const;
};

// --- Function API

array_ref<float> extract_core_distances(
    SparseGraph graph, int32_t min_samples, bool is_sorted
);

SparseGraph compute_mutual_reachability(
    SparseGraph graph, array_ref<float> core_distances
);
#endif  // PLSCAN_API_SPARSE_GRAPH_
