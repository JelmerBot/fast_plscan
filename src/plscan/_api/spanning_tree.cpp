#include "spanning_tree.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <ranges>
#include <vector>

#include "distances.h"

// --- Implementation details

namespace {

// General spanning forest construction helpers

struct Edge {
  int32_t parent = -1;
  int32_t child = -1;
  float distance = std::numeric_limits<float>::infinity();
};

struct SpanningState {
 private:
  std::vector<uint32_t> parent;
  std::vector<uint32_t> rank;

 public:
  std::vector<int64_t> remap;      // needs -1 value
  std::vector<int64_t> component;  // needs -1 value
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
    std::ranges::fill(remap, -1ll);

    // List monotonic component labels per point
    int64_t counter = 0;
    for (size_t idx = 0; idx < component.size(); ++idx) {
      uint32_t const comp = find(idx);
      if (remap[comp] < 0)
        remap[comp] = counter++;
      component[idx] = remap[comp];
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

[[nodiscard]] size_t apply_candidates(
    SpanningTreeWriteView mst, SpanningState &state, size_t &num_edges
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
    mst.parent[num_edges] = static_cast<uint32_t>(parent);
    mst.child[num_edges] = static_cast<uint32_t>(child);
    mst.distance[num_edges++] = distance;
  }
  return num_edges - start_count;
}

void combine_vectors(std::vector<Edge> &dest, std::vector<Edge> const &src) {
  for (size_t idx = 0; idx < src.size(); ++idx)
    if (src[idx].distance < dest[idx].distance)
      dest[idx] = src[idx];
}

#pragma omp declare reduction(                                             \
        merge_edges : std::vector<Edge> : combine_vectors(omp_out, omp_in) \
) initializer(omp_priv = omp_orig)

// Extract spanning forest from sparse graph

void find_candidates(SpanningState &state, SparseGraphWriteView const graph) {
  std::vector<Edge> &candidates = state.candidates;
  std::span<int64_t const> const component = state.component;

  // clang-format off
  #pragma omp parallel for default(none) shared(graph, component) reduction(merge_edges : candidates)  // clang-format on
  for (int32_t row = 0; row < graph.size(); ++row) {
    int64_t const comp = component[row];
    int32_t const start = graph.indptr[row];
    if (float const distance = graph.data[start];
        distance < candidates[comp].distance)
      candidates[comp] = Edge{row, graph.indices[start], distance};
  }
}

void update_graph(
    SpanningState const &state, SparseGraphWriteView const graph
) {
  std::span const component = state.component;

  // clang-format off
  #pragma omp parallel for default(none) shared(graph, component)  // clang-format on
  for (int32_t row = 0; row < graph.size(); ++row) {
    int32_t const start = graph.indptr[row];
    int32_t const end = graph.indptr[row + 1];
    int32_t counter = start;
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

size_t process_graph(
    SpanningTreeWriteView mst, SparseGraphWriteView const graph
) {
  nb::gil_scoped_release guard{};

  size_t num_edges = 0u;
  size_t num_components = graph.size();
  SpanningState state(num_components);

  while (num_components > 1) {
    find_candidates(state, graph);
    size_t const new_edges = apply_candidates(mst, state, num_edges);
    if (new_edges == 0)
      break;

    num_components -= new_edges;
    state.update(num_components);
    update_graph(state, graph);
  }
  return num_edges;
}

// Spanning forest from general space tree helpers

struct TraversalState {
  std::vector<int64_t> node_component;
  std::vector<float> component_nn_dist;
  std::vector<float> candidate_dist;
  std::vector<int32_t> candidate_idx;

  explicit TraversalState(SpaceTreeView const tree)
      : node_component(tree.node_data.size(), -1ll),
        component_nn_dist(tree.data.shape(0)),
        candidate_dist(tree.data.shape(0)),
        candidate_idx(tree.data.shape(0)) {}

  NB_INLINE void update(
      SpanningState const &state, SpaceTreeView const tree,
      size_t const num_components
  ) {
    component_nn_dist.resize(num_components);
    std::ranges::fill(
        component_nn_dist, std::numeric_limits<float>::infinity()
    );
    std::ranges::fill(candidate_dist, std::numeric_limits<float>::infinity());
    std::ranges::fill(candidate_idx, -1);

    size_t const num_nodes = node_component.size();
    std::span const component = state.component;
    for (size_t _i = 1; _i <= num_nodes; ++_i) {
      size_t const idx = num_nodes - _i;
      if (auto const &[idx_start, idx_end, is_leaf, _] = tree.node_data[idx];
          is_leaf) {
        bool flag = true;
        int64_t const candidate_comp = component[tree.idx_array[idx_start]];
        for (int64_t _j = idx_start + 1; _j < idx_end; ++_j)
          if (component[tree.idx_array[_j]] != candidate_comp) {
            flag = false;
            break;
          }

        if (flag)
          node_component[idx] = candidate_comp;
      } else if (auto const left = idx * 2 + 1;
                 node_component[left] == node_component[left + 1])
        node_component[idx] = node_component[left];
    }
  }
};

template <typename rdist_fun_t, typename min_rdist_fun_t>
class RowQueryState {
  SpaceTreeView const tree;
  std::span<float const> const core_distances;
  std::span<int64_t const> const point_component;
  std::span<int64_t const> const node_component;

  std::span<float const> const point;
  float const current_core_dist;
  int64_t const current_component;
  float &component_nn_dist;  // per thread best distance for the component
  float &candidate_dist;     // per point best distance
  int32_t &candidate_idx;    // per point best index

  rdist_fun_t const rdist_fun;
  min_rdist_fun_t const min_rdist_fun;

 public:
  RowQueryState(
      SpanningState const &state, TraversalState &traversal_state,
      SpaceTreeView const tree, std::span<float> const core_distances,
      rdist_fun_t rdist_fun, min_rdist_fun_t min_rdist_fun,
      int64_t const point_idx
  )
      : tree(tree),
        core_distances(core_distances),
        point_component(state.component),
        node_component(traversal_state.node_component),
        point(&tree.data(point_idx, 0), tree.data.shape(1)),
        current_core_dist(core_distances[point_idx]),
        current_component(state.component[point_idx]),
        component_nn_dist(traversal_state.component_nn_dist[current_component]),
        candidate_dist(traversal_state.candidate_dist[point_idx]),
        candidate_idx(traversal_state.candidate_idx[point_idx]),
        rdist_fun(std::move(rdist_fun)),
        min_rdist_fun(std::move(min_rdist_fun)) {}

  void perform_query() {
    constexpr size_t node_idx = 0ull;
    float const lower_bound = min_rdist_fun(tree, point, node_idx);
    recursive_query(lower_bound, node_idx);
  }

 private:
  void recursive_query(float const lower_bound, size_t const node_idx) {
    // Cannot improve downstream this node (wrap in read-lock)
    if (lower_bound > std::min(candidate_dist, component_nn_dist) or
        current_core_dist > component_nn_dist or
        node_component[node_idx] == current_component)
      return;

    if (auto const &[idx_start, idx_end, is_leaf, _] = tree.node_data[node_idx];
        is_leaf)
      process_leaf(idx_start, idx_end);
    else
      traverse_node(node_idx);
  }

  void process_leaf(int64_t const idx_start, int64_t const idx_end) const {
    for (int64_t _i = idx_start; _i < idx_end; ++_i) {
      int64_t const idx = tree.idx_array[_i];

      // Skip if the point is already in the component or does not improve
      // on the best found so far (per-thread)
      if (point_component[idx] == current_component or
          core_distances[idx] >= component_nn_dist)
        continue;

      // Compute the mutual reachability distance
      float const dist = std::max(
          {rdist_fun(point, {&tree.data(idx, 0), tree.data.shape(1)}),
           current_core_dist, core_distances[idx]}
      );

      // Update the candidate distance and index
      if (dist < candidate_dist) {
        candidate_dist = dist;
        candidate_idx = static_cast<int32_t>(idx);
        // Update the component nearest neighbor distance. This can be a
        // data-race! Adding locks introduces more costs than working with
        // potentially non-optimal component_nn_dists. The data race does not
        // influence correctness, only the level of pruning in the traversal.
        if (dist < component_nn_dist)
          component_nn_dist = dist;
      }
    }
  }

  void traverse_node(size_t const node_idx) {
    size_t left = node_idx * 2 + 1;
    size_t right = left + 1;

    float left_lower_bound = min_rdist_fun(tree, point, left);
    float right_lower_bound = min_rdist_fun(tree, point, right);
    if (left_lower_bound > right_lower_bound) {
      std::swap(left, right);
      std::swap(left_lower_bound, right_lower_bound);
    }

    recursive_query(left_lower_bound, left);
    recursive_query(right_lower_bound, right);
  }
};

size_t initialize_mst_from_knn(
    SpanningTreeWriteView mst, SpanningState &state, SparseGraphView const knn,
    std::span<float> const core_distances, size_t &num_edges
) {
  std::vector<Edge> &candidates = state.candidates;
  // clang-format off
  #pragma omp parallel for default(none) shared(knn, core_distances, candidates)  // clang-format on
  for (int32_t row = 0; row < knn.size(); ++row)
    for (int32_t idx = knn.indptr[row]; idx < knn.indptr[row + 1]; ++idx)
      if (int32_t const col = knn.indices[idx];
          core_distances[row] >= core_distances[col])
        candidates[row] = Edge{row, col, core_distances[row]};

  return apply_candidates(mst, state, num_edges);
}

template <typename rdist_fun_t, typename min_rdist_fun_t>
void component_aware_query(
    SpanningState const &state, TraversalState &traversal_state,
    SpaceTreeView const tree, std::span<float> const core_distances,
    rdist_fun_t rdist_fun, min_rdist_fun_t min_rdist_fun
) {
  // clang-format off
  #pragma omp parallel for default(none) shared(state, traversal_state, core_distances, tree, rdist_fun, min_rdist_fun)  // clang-format on
  for (int64_t point_idx = 0; point_idx < tree.data.shape(0); ++point_idx) {
    RowQueryState query_state{state,          traversal_state, tree,
                              core_distances, rdist_fun,       min_rdist_fun,
                              point_idx};
    query_state.perform_query();
  }
}

void find_candidates(
    SpanningState &state, TraversalState const &traversal_state
) {
  std::vector<Edge> &candidates = state.candidates;
  std::span<int64_t const> const component = state.component;
  std::span const edge_dist = traversal_state.candidate_dist;
  std::span const edge_idx = traversal_state.candidate_idx;

  // clang-format off
  #pragma omp parallel for default(none) shared(edge_dist, edge_idx, component) reduction(merge_edges : candidates)  // clang-format on
  for (int32_t row = 0; row < edge_dist.size(); ++row) {
    int64_t const comp = component[row];
    if (float const distance = edge_dist[row];
        distance < candidates[comp].distance)
      candidates[comp] = Edge{row, edge_idx[row], distance};
  }
}

template <
    typename rdist_fun_t, typename min_rdist_fun_t, typename to_dist_fun_t>
size_t space_tree_boruvka(
    SpanningTreeWriteView mst, SpaceTreeView const tree,
    SparseGraphView const knn, std::span<float> const core_distances,
    rdist_fun_t &&rdist_fun, min_rdist_fun_t &&min_rdist_fun,
    to_dist_fun_t &&to_dist_fun
) {
  nb::gil_scoped_release guard{};

  size_t num_edges = 0u;
  size_t num_components = knn.size();
  SpanningState state(num_components);
  TraversalState traversal_state(tree);
  size_t new_edges = initialize_mst_from_knn(
      mst, state, knn, core_distances, num_edges
  );

  while (true) {
    num_components -= new_edges;
    if (num_components == 1)
      break;

    state.update(num_components);
    traversal_state.update(state, tree, num_components);

    component_aware_query<rdist_fun_t, min_rdist_fun_t>(
        state, traversal_state, tree, core_distances, rdist_fun, min_rdist_fun
    );
    find_candidates(state, traversal_state);
    new_edges = apply_candidates(mst, state, num_edges);
    if (new_edges == 0)
      break;
  }

  // Convert the distances to the final form, the compiler optimizes this
  // away entirely if the to_dist_fun is a no-op!
  for (float &dist_val : core_distances)
    dist_val = to_dist_fun(dist_val);
  for (size_t idx = 0; idx < num_edges; ++idx)
    mst.distance[idx] = to_dist_fun(mst.distance[idx]);

  return num_edges;
}

using space_tree_boruvka_fun_t = size_t (*)(
    SpanningTreeWriteView, SpaceTreeView, SparseGraphView, std::span<float>,
    nb::dict
);

// Spanning forest from kdtree

template <Metric metric>
size_t run_kdtree_boruvka(
    SpanningTreeWriteView mst, SpaceTreeView const tree,
    SparseGraphView const knn, std::span<float> const core_distances,
    nb::dict const metric_kws
) {
  return space_tree_boruvka(
      mst, tree, knn, core_distances, get_rdist<metric>(metric_kws),
      get_kdtree_min_rdist<metric>(metric_kws),
      get_rdist_to_dist<metric>(metric_kws)
  );
}

space_tree_boruvka_fun_t get_kdtree_executor(char const *const metric) {
  // Must match Metric enumeration order!
  constexpr static std::array lookup = {
      run_kdtree_boruvka<Metric::Euclidean>,
      run_kdtree_boruvka<Metric::Cityblock>,
      run_kdtree_boruvka<Metric::Chebyshev>,
      run_kdtree_boruvka<Metric::Minkowski>,
  };

  auto const idx = parse_metric(metric);
  if (idx >= lookup.size())
    throw nb::value_error(
        nb::str("Missing KDTree boruvka for '{}'").format(metric).c_str()
    );

  return lookup[idx];
}

// Spanning forest from balltree

template <Metric metric>
size_t run_balltree_boruvka(
    SpanningTreeWriteView mst, SpaceTreeView const tree,
    SparseGraphView const knn, std::span<float> const core_distances,
    nb::dict const metric_kws
) {
  return space_tree_boruvka(
      mst, tree, knn, core_distances, get_rdist<metric>(metric_kws),
      get_balltree_min_rdist<metric>(metric_kws),
      get_rdist_to_dist<metric>(metric_kws)
  );
}

space_tree_boruvka_fun_t get_balltree_executor(char const *const metric) {
  // Must match Metric enumeration order!
  constexpr static std::array lookup = {
      run_balltree_boruvka<Metric::Euclidean>,
      run_balltree_boruvka<Metric::Cityblock>,
      run_balltree_boruvka<Metric::Chebyshev>,
      run_balltree_boruvka<Metric::Minkowski>,
      run_balltree_boruvka<Metric::Hamming>,
      run_balltree_boruvka<Metric::Braycurtis>,
      run_balltree_boruvka<Metric::Canberra>,
      run_balltree_boruvka<Metric::Haversine>,
      run_balltree_boruvka<Metric::SEuclidean>,
      run_balltree_boruvka<Metric::Mahalanobis>,
      run_balltree_boruvka<Metric::Dice>,
      run_balltree_boruvka<Metric::Jaccard>,
      run_balltree_boruvka<Metric::Russellrao>,
      run_balltree_boruvka<Metric::Rogerstanimoto>,
      run_balltree_boruvka<Metric::Sokalsneath>,
  };

  auto const idx = parse_metric(metric);
  if (idx >= lookup.size())
    throw nb::value_error(  //
        nb::str("Missing BallTree boruvka for '{}'").format(metric).c_str()
    );

  return lookup[idx];
}

}  // namespace

// --- Function API

SpanningTree extract_spanning_forest(SparseGraph graph) {
  // Build the spanning tree structure
  auto [mst_view, mst_cap] = SpanningTree::allocate(graph.size() - 1u);
  auto [graph_view, graph_cap] = SparseGraph::allocate_copy(graph);
  size_t num_edges = process_graph(mst_view, graph_view);
  return {mst_view, std::move(mst_cap), num_edges};
}

SpanningTree compute_spanning_tree_kdtree(
    SpaceTree const tree, SparseGraph const knn,
    array_ref<float> const core_distances, char const *const metric,
    nb::dict const metric_kws
) {
  // Build the spanning tree structure
  auto [mst_view, mst_cap] = SpanningTree::allocate(knn.size() - 1u);
  size_t num_edges = get_kdtree_executor(metric)(
      mst_view, tree.view(), knn.view(), to_view(core_distances), metric_kws
  );
  return {mst_view, std::move(mst_cap), num_edges};
}

SpanningTree compute_spanning_tree_balltree(
    SpaceTree const tree, SparseGraph const knn,
    array_ref<float> const core_distances, char const *const metric,
    nb::dict const metric_kws
) {
  // Build the spanning tree structure
  auto [mst_view, mst_cap] = SpanningTree::allocate(knn.size() - 1u);
  size_t num_edges = get_balltree_executor(metric)(
      mst_view, tree.view(), knn.view(), to_view(core_distances), metric_kws
  );
  return {mst_view, std::move(mst_cap), num_edges};
}

// --- Class API

size_t SpanningTreeWriteView::size() const {
  return parent.size();
}

size_t SpanningTreeView::size() const {
  return parent.size();
}

// SpanningTree constructors and member functions

SpanningTree::SpanningTree(
    array_ref<uint32_t const> const parent,
    array_ref<uint32_t const> const child, array_ref<float const> const distance
)
    : parent(parent), child(child), distance(distance) {}

SpanningTree::SpanningTree(
    SpanningTreeWriteView const view, SpanningTreeCapsule cap,
    size_t const num_edges
)
    : parent(to_array(view.parent, std::move(cap.parent), num_edges)),
      child(to_array(view.child, std::move(cap.child), num_edges)),
      distance(to_array(view.distance, std::move(cap.distance), num_edges)) {}

std::pair<SpanningTreeWriteView, SpanningTreeCapsule> SpanningTree::allocate(
    size_t const num_edges
) {
  auto [parents, parent_cap] = new_buffer<uint32_t>(num_edges);
  auto [children, child_cap] = new_buffer<uint32_t>(num_edges);
  auto [distances, distance_cap] = new_buffer<float>(num_edges);
  return {
      {parents, children, distances},
      {std::move(parent_cap), std::move(child_cap), std::move(distance_cap)}
  };
}

SpanningTreeView SpanningTree::view() const {
  return {to_view(parent), to_view(child), to_view(distance)};
}

size_t SpanningTree::size() const {
  return parent.size();
}
