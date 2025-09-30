#include "condensed_tree.h"

#include <vector>

// --- Implementation details

namespace {

struct RowInfo {
  uint32_t const parent;
  float distance;
  float const size;
  uint32_t const left;
  uint32_t const left_count;
  float const left_size;
  uint32_t const right;
  uint32_t const right_count;
  float const right_size;
};

struct CondenseState {
  CondensedTreeWriteView condensed_tree;
  LinkageTreeView const linkage_tree;
  SpanningTreeView const spanning_tree;
  std::vector<uint32_t> parent_of;
  std::vector<size_t> pending_idx;
  std::vector<float> pending_distance;

  explicit CondenseState(
      CondensedTreeWriteView condensed_tree, LinkageTreeView const linkage_tree,
      SpanningTreeView const spanning_tree, size_t const num_points
  )
      : condensed_tree(std::move(condensed_tree)),
        linkage_tree(linkage_tree),
        spanning_tree(spanning_tree),
        parent_of(num_points - 1u, num_points),
        pending_idx(num_points - 1u),
        pending_distance(num_points - 1u) {}

  template <typename function_t>
  auto process_rows(
      size_t const num_points, float const min_size, function_t get_row
  ) {
    size_t const num_edges = linkage_tree.size();
    auto next_label = static_cast<uint32_t>(num_points);
    size_t cluster_count = 0u;
    size_t idx = 0u;

    // Iterate over the rows in reverse order.
    for (size_t _i = 1ull; _i <= num_edges; ++_i) {
      size_t const node_idx = num_edges - _i;
      RowInfo row = get_row(node_idx, num_points);

      // Append or write points to reserved spots.
      size_t out_idx = update_output_index(row, idx, node_idx, min_size);
      store_or_delay(row, out_idx, num_points, min_size);

      // Write rows for cluster merges.
      if (row.left_size >= min_size && row.right_size >= min_size)
        write_merge(row, idx, cluster_count, next_label, num_points);
    }
    return std::make_pair(idx, cluster_count);
  }

  [[nodiscard]] RowInfo get_row(  //
      size_t const node_idx, size_t const num_points
  ) const {
    uint32_t const left = linkage_tree.parent[node_idx];
    uint32_t const right = linkage_tree.child[node_idx];
    return {
        parent_of[node_idx],
        spanning_tree.distance[node_idx],
        linkage_tree.child_size[node_idx],
        left,
        left < num_points ? 1u : linkage_tree.child_count[left - num_points],
        left < num_points ? 1.0f : linkage_tree.child_size[left - num_points],
        right,
        right < num_points ? 1u : linkage_tree.child_count[right - num_points],
        right < num_points ? 1.0f : linkage_tree.child_size[right - num_points]
    };
  }

  [[nodiscard]] RowInfo get_row(
      size_t const node_idx, size_t const num_points,
      std::span<float> const weights
  ) const {
    uint32_t const left = linkage_tree.parent[node_idx];
    uint32_t const right = linkage_tree.child[node_idx];
    return {
        parent_of[node_idx],
        spanning_tree.distance[node_idx],
        linkage_tree.child_size[node_idx],
        left,
        left < num_points ? 1u : linkage_tree.child_count[left - num_points],
        left < num_points ? weights[left]
                          : linkage_tree.child_size[left - num_points],
        right,
        right < num_points ? 1u : linkage_tree.child_count[right - num_points],
        right < num_points ? weights[right]
                           : linkage_tree.child_size[right - num_points]
    };
  }

 private:
  size_t update_output_index(
      RowInfo &row, size_t &idx, size_t const node_idx, float const min_size
  ) const {
    size_t out_idx;
    if (row.size < min_size) {
      // Points in pruned branches go to a reserved spot
      out_idx = pending_idx[node_idx];
      row.distance = pending_distance[node_idx];
    } else {
      // Points in accepted branches are appended to the end at `idx`
      out_idx = idx;
      // Reserve spots for potential pruned descendants.
      idx += (row.left_size < min_size) * row.left_count +
             (row.right_size < min_size) * row.right_count;
    }
    return out_idx;
  }

  void store_or_delay(
      RowInfo const &row, size_t &out_idx, size_t const num_points,
      float const min_cluster_size
  ) {
    // Sides that represent a single points are written to the output index.
    // Non-point sides propagate their parent and reserved spots.
    if (row.left < num_points)
      write_row(out_idx, row.parent, row.distance, row.left, row.left_size);
    else
      delay_row(
          out_idx, row.parent, row.distance, row.left, row.left_count,
          row.left_size, num_points, min_cluster_size
      );

    if (row.right < num_points)
      write_row(out_idx, row.parent, row.distance, row.right, row.right_size);
    else
      delay_row(
          out_idx, row.parent, row.distance, row.right, row.right_count,
          row.right_size, num_points, min_cluster_size
      );
  }

  void write_row(
      size_t &out_idx, uint32_t const parent, float const distance,
      uint32_t const child, float const child_size
  ) const {
    condensed_tree.parent[out_idx] = parent;
    condensed_tree.child[out_idx] = child;
    condensed_tree.distance[out_idx] = distance;
    condensed_tree.child_size[out_idx] = child_size;
    ++out_idx;
  }

  void delay_row(
      size_t &out_idx, uint32_t const parent, float const distance,
      uint32_t const child, uint32_t const child_count, float const child_size,
      size_t const num_points, float const min_cluster_size
  ) {
    uint32_t const child_idx = child - num_points;
    // Propagate the parent
    parent_of[child_idx] = parent;
    if (child_size < min_cluster_size) {
      // Propagate the reserved output index and pruned distance.
      pending_idx[child_idx] = out_idx;
      pending_distance[child_idx] = distance;
      out_idx += child_count;
    }
  }

  void write_merge(
      RowInfo const &row, size_t &idx, size_t &cluster_count,
      uint32_t &next_label, size_t const num_points
  ) {
    // Adjust numbering for phantom root and real roots
    uint32_t const parent = row.parent == num_points ? ++next_label
                                                     : row.parent;
    // Introduces new parent labels and appends rows for the merge
    parent_of[row.left - num_points] = ++next_label;
    condensed_tree.cluster_rows[cluster_count++] = idx;
    write_row(idx, parent, row.distance, next_label, row.left_size);
    parent_of[row.right - num_points] = ++next_label;
    condensed_tree.cluster_rows[cluster_count++] = idx;
    write_row(idx, parent, row.distance, next_label, row.right_size);
  }
};

std::pair<size_t, size_t> process_hierarchy(
    CondensedTreeWriteView tree, LinkageTreeView const linkage,
    SpanningTreeView const mst, size_t const num_points, float const min_size,
    std::optional<array_ref<float>> const sample_weights
) {
  nb::gil_scoped_release guard{};
  CondenseState state{tree, linkage, mst, num_points};
  if (sample_weights) {
    return state.process_rows(
        num_points, min_size,
        [&state,
         weights = std::span(sample_weights->data(), sample_weights->size())](
            size_t const node_idx, size_t const num_points
        ) { return state.get_row(node_idx, num_points, weights); }
    );
  }
  return state.process_rows(
      num_points, min_size,
      [&state](size_t const node_idx, size_t const num_points) {
        return state.get_row(node_idx, num_points);
      }
  );
}

}  // namespace

// --- Function API

CondensedTree compute_condensed_tree(
    LinkageTree const linkage, SpanningTree const mst, size_t const num_points,
    float const min_size, std::optional<array_ref<float>> const sample_weights
) {
  auto [tree_view, tree_cap] = CondensedTree::allocate(linkage.size());
  auto [filled_edges, cluster_count] = process_hierarchy(
      tree_view, linkage.view(), mst.view(), num_points, min_size,
      sample_weights
  );
  return {tree_view, std::move(tree_cap), filled_edges, cluster_count};
}

// --- Class API

CondensedTree::CondensedTree(
    array_ref<uint32_t const> const parent,
    array_ref<uint32_t const> const child,
    array_ref<float const> const distance,
    array_ref<float const> const child_size,
    array_ref<uint32_t const> const cluster_rows
)
    : parent(parent),
      child(child),
      distance(distance),
      child_size(child_size),
      cluster_rows(cluster_rows) {}

CondensedTree::CondensedTree(
    CondensedTreeWriteView const view, CondensedTreeCapsule cap,
    size_t const num_edges, size_t const num_clusters
)
    : parent(to_array(view.parent, std::move(cap.parent), num_edges)),
      child(to_array(view.child, std::move(cap.child), num_edges)),
      distance(to_array(view.distance, std::move(cap.distance), num_edges)),
      child_size(to_array(view.child_size, std::move(cap.child_size), num_edges)
      ),
      cluster_rows(
          to_array(view.cluster_rows, std::move(cap.cluster_rows), num_clusters)
      ) {}

std::pair<CondensedTreeWriteView, CondensedTreeCapsule> CondensedTree::allocate(
    size_t const num_edges
) {
  size_t const buffer_size = 2 * num_edges;
  auto [parent, parent_cap] = new_buffer<uint32_t>(buffer_size);
  auto [child, child_cap] = new_buffer<uint32_t>(buffer_size);
  auto [dist, dist_cap] = new_buffer<float>(buffer_size);
  auto [size, size_cap] = new_buffer<float>(buffer_size);
  auto [rows, rows_cap] = new_buffer<uint32_t>(num_edges);
  return {
      {parent, child, dist, size, rows},
      {std::move(parent_cap), std::move(child_cap), std::move(dist_cap),
       std::move(size_cap), std::move(rows_cap)}
  };
}

CondensedTreeView CondensedTree::view() const {
  return {
      to_view(parent),     to_view(child),        to_view(distance),
      to_view(child_size), to_view(cluster_rows),
  };
}

size_t CondensedTreeWriteView::size() const {
  return parent.size();
}

size_t CondensedTreeView::size() const {
  return parent.size();
}

size_t CondensedTree::size() const {
  return parent.size();
}
