#include "_spanning_tree.h"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <numeric>
#include <ranges>
#include <vector>

#include "_sparse_graph.h"

struct Edge {
  int32_t parent = -1;
  int32_t child = -1;
  float distance = std::numeric_limits<float>::infinity();
};

class SpanningState {
  std::vector<uint32_t> parent;
  std::vector<uint32_t> rank;
  std::vector<int32_t> remap;

 public:
  std::vector<uint32_t> component;
  std::vector<Edge> candidates;

  explicit SpanningState(size_t const num_points)
      : parent(num_points),
        rank(num_points, 0u),
        remap(num_points),
        component(num_points),
        candidates(num_points, Edge{}) {
    std::iota(parent.begin(), parent.end(), 0u);
    std::ranges::copy(parent, component.begin());
  }

  NB_INLINE void update(size_t const num_components) {
    // Prepare buffers for new iteration
    candidates.resize(num_components);
    std::ranges::fill(candidates, Edge{});
    std::ranges::fill(remap, -1);

    // List monotonic component labels per point
    int32_t counter = 0;
    for (uint32_t idx = 0; idx < component.size(); ++idx) {
      uint32_t const comp = find(idx);
      if (remap[comp] < 0)
        remap[comp] = counter++;
      component[idx] = static_cast<uint32_t>(remap[comp]);
    }
  }

  NB_INLINE uint32_t find(uint32_t x) {
    while (parent[x] != x) {
      x = parent[x];
      parent[x] = parent[parent[x]];
    }
    return x;
  }

  NB_INLINE auto link(uint32_t x, uint32_t y) {
    if (rank[x] < rank[y])
      std::swap(x, y);
    parent[y] = x;
    if (rank[x] == rank[y])
      ++rank[x];
  }
};

void combine_vectors(std::vector<Edge> &dest, std::vector<Edge> const &src) {
  for (size_t idx = 0; idx < src.size(); ++idx)
    if (src[idx].distance < dest[idx].distance)
      dest[idx] = src[idx];
}

#pragma omp declare reduction(                                             \
        merge_edges : std::vector<Edge> : combine_vectors(omp_out, omp_in) \
) initializer(omp_priv = omp_orig)

void fill_candidates(SpanningState &state, SparseGraphView const graph) {
  std::vector<Edge> &candidates = state.candidates;
  std::span const component(state.component);

  // clang-format off
  #pragma omp parallel for default(none) shared(graph, component) reduction(merge_edges : candidates)  // clang-format on
  for (int32_t row = 0; row < graph.size(); ++row) {
    uint32_t const comp = component[row];
    int32_t const start = graph.indptr[row];
    if (float const distance = graph.data[start];
        distance < candidates[comp].distance)
      candidates[comp] = Edge{row, graph.indices[start], distance};
  }
}

size_t apply_candidates(
    SpanningTreeView tree, SpanningState &state, size_t &num_edges
) {
  size_t const start_count = num_edges;
  for (auto [parent, child, distance] : state.candidates) {
    if (child < 0)
      continue;
    uint32_t const from = state.find(static_cast<uint32_t>(parent));
    uint32_t const to = state.find(static_cast<uint32_t>(child));
    if (from == to)
      continue;
    state.link(from, to);
    tree.parent[num_edges] = static_cast<uint32_t>(parent);
    tree.child[num_edges] = static_cast<uint32_t>(child);
    tree.distance[num_edges++] = distance;
  }
  return num_edges - start_count;
}

void update_graph(
    SparseGraphView const graph, std::span<uint32_t> const component
) {  // clang-format off
  #pragma omp parallel for default(none) shared(graph, component)  // clang-format on
  for (int32_t row = 0; row < graph.size(); ++row) {
    int32_t const start = graph.indptr[row];
    int32_t const end = graph.indptr[row + 1];
    size_t counter = start;
    for (int32_t idx = start; idx < end; ++idx) {
      int32_t const col = graph.indices[idx];
      // Skip if the column index is -1 (indicating no edge)
      if (col < 0)
        break;
      if (component[col] == component[row])
        continue;
      graph.indices[counter] = graph.indices[idx];
      graph.data[counter++] = graph.data[idx];
    }
    // Mark new end of the row with -1 values
    if (counter < end) {
      graph.indices[counter] = -1;
      graph.data[counter] = std::numeric_limits<float>::infinity();
    }
  }
}

size_t process_graph(SpanningTreeView tree, SparseGraphView const graph) {
  nb::gil_scoped_release guard{};

  size_t num_edges = 0u;
  size_t num_components = graph.size();
  SpanningState state(num_components);

  while (num_components > 1) {
    fill_candidates(state, graph);
    size_t const new_edges = apply_candidates(tree, state, num_edges);
    if (new_edges == 0)
      break;

    num_components -= new_edges;
    state.update(num_components);
    update_graph(graph, state.component);
  }
  return num_edges;
}

SpanningTree compute_spanning_forest(SparseGraph graph) {
  auto [tree_view, tree_cap] = SpanningTree::allocate(graph.size() - 1u);
  size_t num_edges = process_graph(tree_view, graph.view());
  return {tree_view, std::move(tree_cap), num_edges};
}

NB_MODULE(_spanning_tree, m) {
  m.doc() = "Module for spanning tree computation in PLSCAN.";

  nb::class_<SpanningTree>(m, "SpanningTree")
      .def(
          nb::init<array_ref<uint32_t>, array_ref<uint32_t>, array_ref<float>>(
          ),
          nb::arg("parent").noconvert(), nb::arg("child").noconvert(),
          nb::arg("distance").noconvert()
      )
      .def_ro("parent", &SpanningTree::parent, nb::rv_policy::reference)
      .def_ro("child", &SpanningTree::child, nb::rv_policy::reference)
      .def_ro("distance", &SpanningTree::distance, nb::rv_policy::reference)
      .def(
          "__iter__",
          [](SpanningTree const &self) {
            return nb::make_tuple(self.parent, self.child, self.distance)
                .attr("__iter__")();
          }
      )
      .doc() = R"(
        SpanningTree contains a sorted minimum spanning tree (MST).

        Parameters
        ----------
        parent : numpy.ndarray[tuple[int], np.dtype[np.uint64]]
            An array of parent node indices.
        child : numpy.ndarray[tuple[int], np.dtype[np.uint64]]
            An array of child node indices.
        distance : numpy.ndarray[tuple[int], np.dtype[np.float32]]
            An array of distances between the nodes.
      )";

  m.def(
      "compute_spanning_forest", &compute_spanning_forest, nb::arg("graph"),
      R"(
            Computes a minimum spanning forest from a sparse graph.

            Parameters
            ----------
            graph : plscan._sparse_graph.SparseGraph
                The input sparse graph.

            Returns
            -------
            plscan._spanning_tree.SpanningTree
                The computed spanning forest.
        )"
  );
}