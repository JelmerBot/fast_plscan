#include "leaf_tree.h"

#include <algorithm>

// --- Implementation details

namespace {

void fill_min_dist(
    LeafTreeWriteView leaf_tree, CondensedTreeView const condensed_tree,
    size_t const num_points
) {
  // last occurrence in the condensed tree!
  for (size_t idx = 0; idx < condensed_tree.size(); ++idx) {
    size_t const parent_idx = condensed_tree.parent[idx] - num_points;
    leaf_tree.min_distance[parent_idx] = condensed_tree.distance[idx];
  }
}

void fill_parent_and_max_dist(
    LeafTreeWriteView leaf_tree, CondensedTreeView const condensed_tree,
    size_t const num_points
) {
  // fill default values to match the phantom root cluster.
  size_t const num_leaves = leaf_tree.size();
  std::fill_n(leaf_tree.parent.begin(), num_leaves, 0u);
  std::fill_n(
      leaf_tree.max_distance.begin(), num_leaves, condensed_tree.distance[0]
  );

  // fill in actual values for non-root clusters.
  for (size_t const idx : condensed_tree.cluster_rows) {
    size_t const child_idx = condensed_tree.child[idx] - num_points;
    leaf_tree.parent[child_idx] = condensed_tree.parent[idx] - num_points;
    leaf_tree.max_distance[child_idx] = condensed_tree.distance[idx];
  }
}

void fill_sizes(
    LeafTreeWriteView leaf_tree, CondensedTreeView const condensed_tree,
    size_t const num_points, float const min_size
) {
  // fill in default min size values
  std::fill_n(leaf_tree.min_size.begin(), leaf_tree.size(), min_size);

  // reverse cluster row pairs.
  size_t const num_clusters = condensed_tree.cluster_rows.size();
  for (size_t _i = 1; _i <= num_clusters; _i += 2) {
    size_t const _row_idx = num_clusters - _i;
    size_t const left_idx = condensed_tree.cluster_rows[_row_idx];
    size_t const right_idx = condensed_tree.cluster_rows[_row_idx - 1u];

    float const size = std::min(
        condensed_tree.child_size[left_idx],
        condensed_tree.child_size[right_idx]
    );
    uint32_t const out_idx = condensed_tree.child[left_idx] - num_points;
    uint32_t const parent_idx = condensed_tree.parent[left_idx] - num_points;
    leaf_tree.max_size[out_idx] = size;
    leaf_tree.max_size[out_idx - 1u] = size;
    leaf_tree.min_size[parent_idx] = std::max(
        {size, leaf_tree.min_size[out_idx - 1u], leaf_tree.min_size[out_idx]}
    );
    // Update the phantom root min-size for root-parents.
    if (leaf_tree.parent[parent_idx] == 0)
      leaf_tree.min_size[0] = std::max(
          leaf_tree.min_size[0], leaf_tree.min_size[parent_idx]
      );
  }

  // set the root sizes to largest observed min size to provide an upper
  // observed size limit (for plotting). Can't know their exact size here...
  leaf_tree.max_size[0] = static_cast<float>(num_points);
  for (size_t idx = 1; idx < leaf_tree.size(); ++idx)
    if (leaf_tree.parent[idx] == 0u)
      leaf_tree.max_size[idx] = leaf_tree.min_size[0];
}

void process_clusters(
    LeafTreeWriteView leaf_tree, CondensedTreeView const condensed_tree,
    size_t const num_points, float const min_size
) {
  nb::gil_scoped_release guard{};
  fill_min_dist(leaf_tree, condensed_tree, num_points);
  fill_parent_and_max_dist(leaf_tree, condensed_tree, num_points);
  fill_sizes(leaf_tree, condensed_tree, num_points, min_size);
}

}  // namespace

// --- Function API

LeafTree compute_leaf_tree(
    CondensedTree const condensed_tree, size_t const num_points,
    float const min_size
) {
  CondensedTreeView const condensed_view = condensed_tree.view();
  size_t const last_cluster_row = condensed_view.cluster_rows.back();
  size_t const max_label = condensed_view.child[last_cluster_row] - num_points;
  auto [tree_view, tree_cap] = LeafTree::allocate(max_label + 1u);
  process_clusters(tree_view, condensed_view, num_points, min_size);
  return {tree_view, std::move(tree_cap)};
};

array_ref<uint32_t const> apply_size_cut(
    LeafTree const &leaf_tree, float const cut_size
) {
  size_t num_selected = 0;
  auto [out_view, out_cap] = new_buffer<uint32_t>(leaf_tree.size());
  {
    nb::gil_scoped_release guard{};
    LeafTreeView const leaf_tree_view = leaf_tree.view();
    for (uint32_t idx = 0; idx < leaf_tree_view.size(); ++idx)
      if (leaf_tree_view.min_size[idx] <= cut_size &&
          leaf_tree_view.max_size[idx] > cut_size)
        out_view[num_selected++] = idx;
  }
  return to_array(out_view, std::move(out_cap), num_selected);
}

array_ref<uint32_t const> apply_distance_cut(
    LeafTree const &leaf_tree, float const cut_distance
) {
  size_t num_selected = 0;
  auto [out_view, out_cap] = new_buffer<uint32_t>(leaf_tree.size());
  {
    nb::gil_scoped_release guard{};
    LeafTreeView const leaf_tree_view = leaf_tree.view();
    for (uint32_t idx = 0; idx < leaf_tree_view.size(); ++idx)
      if (leaf_tree_view.min_distance[idx] <= cut_distance &&
          leaf_tree_view.max_distance[idx] > cut_distance)
        out_view[num_selected++] = idx;
  }
  return to_array(out_view, std::move(out_cap), num_selected);
}

// --- Class API

LeafTree::LeafTree(
    array_ref<uint32_t const> const parent,
    array_ref<float const> const min_distance,
    array_ref<float const> const max_distance,
    array_ref<float const> const min_size, array_ref<float const> const max_size
)
    : parent(parent),
      min_distance(min_distance),
      max_distance(max_distance),
      min_size(min_size),
      max_size(max_size) {}

LeafTree::LeafTree(LeafTreeWriteView const view, LeafTreeCapsule cap)
    : parent(  //
          to_array(view.parent, std::move(cap.parent), view.parent.size())
      ),
      min_distance(to_array(
          view.min_distance, std::move(cap.min_distance),
          view.min_distance.size()
      )),
      max_distance(to_array(
          view.max_distance, std::move(cap.max_distance),
          view.max_distance.size()
      )),
      min_size(
          to_array(view.min_size, std::move(cap.min_size), view.min_size.size())
      ),
      max_size(
          to_array(view.max_size, std::move(cap.max_size), view.max_size.size())
      ) {}

std::pair<LeafTreeWriteView, LeafTreeCapsule> LeafTree::allocate(
    size_t const num_clusters
) {
  auto [parent, parent_cap] = new_buffer<uint32_t>(num_clusters);
  auto [min_distance, min_distance_cap] = new_buffer<float>(num_clusters);
  auto [max_distance, max_distance_cap] = new_buffer<float>(num_clusters);
  auto [min_size, min_size_cap] = new_buffer<float>(num_clusters);
  auto [max_size, max_size_cap] = new_buffer<float>(num_clusters);
  return {
      {parent, min_distance, max_distance, min_size, max_size},
      {std::move(parent_cap), std::move(min_distance_cap),
       std::move(max_distance_cap), std::move(min_size_cap),
       std::move(max_size_cap)}
  };
}

LeafTreeView LeafTree::view() const {
  return {
      to_view(parent),   to_view(min_distance), to_view(max_distance),
      to_view(min_size), to_view(max_size),
  };
}

size_t LeafTreeWriteView::size() const {
  return parent.size();
}

size_t LeafTreeView::size() const {
  return parent.size();
}

size_t LeafTree::size() const {
  return parent.size();
}
