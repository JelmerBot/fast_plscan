#include "linkage_tree.h"

#include <algorithm>
#include <vector>

// --- Implementation details

namespace {

class LinkageState {
  std::vector<uint32_t> parent;
  std::vector<uint32_t> child_count;
  std::vector<float> child_size;

 public:
  LinkageState(
      size_t const num_points, size_t const num_edges,
      std::optional<array_ref<float>> const sample_weights
  )
      : parent(num_points + num_edges, 0u),
        child_count(num_points + num_edges),
        child_size(num_points + num_edges) {
    // Initialize the working arrays.
    std::fill_n(child_count.begin(), num_points, 1);
    if (!sample_weights)
      std::fill_n(child_size.begin(), num_points, 1.0f);
    else
      std::copy_n(
          to_view(*sample_weights).begin(), num_points, child_size.begin()
      );
  }

  NB_INLINE uint32_t find(uint32_t node) {
    uint32_t relabel = node;
    while (parent[node] != 0u && parent[node] != node)
      node = parent[node];

    parent[node] = node;
    while (parent[relabel] != node) {
      uint32_t const next_relabel = parent[relabel];
      parent[relabel] = node;
      relabel = next_relabel;
    }
    return node;
  }

  NB_INLINE auto link(
      uint32_t const next, uint32_t const left, uint32_t const right
  ) {
    parent[left] = next;
    parent[right] = next;
    child_count[next] = child_count[left] + child_count[right];
    child_size[next] = child_size[left] + child_size[right];
    return std::make_pair(child_count[next], child_size[next]);
  }
};

size_t process_spanning_tree(
    LinkageTreeWriteView tree, SpanningTreeView const mst,
    size_t const num_points,
    std::optional<array_ref<float>> const sample_weights
) {
  nb::gil_scoped_release guard{};
  LinkageState state{num_points, mst.size(), sample_weights};

  size_t idx = 0;
  for (; idx < mst.size(); ++idx) {
    size_t const next = num_points + idx;
    uint32_t const left = state.find(mst.parent[idx]);
    uint32_t const right = state.find(mst.child[idx]);
    std::tie(tree.child[idx], tree.parent[idx]) = std::minmax(left, right);
    std::tie(tree.child_count[idx], tree.child_size[idx]) = state.link(
        next, left, right
    );
  }

  return idx;
}

}  // namespace

// --- Function API

LinkageTree compute_linkage_tree(
    SpanningTree const mst, size_t const num_points,
    std::optional<array_ref<float>> const sample_weights
) {
  auto [tree_view, tree_cap] = LinkageTree::allocate(num_points - 1);
  size_t const num_edges = process_spanning_tree(
      tree_view, mst.view(), num_points, sample_weights
  );
  return {tree_view, std::move(tree_cap), num_edges};
}

// --- Class API

LinkageTree::LinkageTree(
    array_ref<uint32_t const> const parent,
    array_ref<uint32_t const> const child,
    array_ref<uint32_t const> const child_count,
    array_ref<float const> const child_size
)
    : parent(parent),
      child(child),
      child_count(child_count),
      child_size(child_size) {}

LinkageTree::LinkageTree(
    LinkageTreeWriteView const view, LinkageTreeCapsule cap,
    size_t const num_edges
)
    : parent(to_array(view.parent, std::move(cap.parent), num_edges)),
      child(to_array(view.child, std::move(cap.child), num_edges)),
      child_count(
          to_array(view.child_count, std::move(cap.child_count), num_edges)
      ),
      child_size(to_array(view.child_size, std::move(cap.child_size), num_edges)
      ) {}

std::pair<LinkageTreeWriteView, LinkageTreeCapsule> LinkageTree::allocate(
    size_t const num_edges
) {
  size_t const buffer_size = 2 * num_edges;
  auto [parent, parent_cap] = new_buffer<uint32_t>(buffer_size);
  auto [child, child_cap] = new_buffer<uint32_t>(buffer_size);
  auto [count, count_cap] = new_buffer<uint32_t>(buffer_size);
  auto [size, size_cap] = new_buffer<float>(buffer_size);
  return {
      {parent, child, count, size}, {parent_cap, child_cap, count_cap, size_cap}
  };
}

LinkageTreeView LinkageTree::view() const {
  return {
      to_view(parent),
      to_view(child),
      to_view(child_count),
      to_view(child_size),
  };
}

size_t LinkageTreeWriteView::size() const {
  return parent.size();
}

size_t LinkageTreeView::size() const {
  return parent.size();
}

size_t LinkageTree::size() const {
  return parent.size();
}
