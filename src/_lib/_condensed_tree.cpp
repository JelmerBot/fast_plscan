#include "_condensed_tree.h"

#include <nanobind/stl/optional.h>

#include <optional>
#include <vector>

#include "_linkage_tree.h"
#include "_spanning_tree.h"

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

NB_MODULE(_condensed_tree_ext, m) {
  m.doc() = "Module for condensed tree computation in PLSCAN.";

  nb::class_<CondensedTree>(m, "CondensedTree")
      .def(
          "__init__",
          [](CondensedTree *t, nb::handle parent, nb::handle child,
             nb::handle distance, nb::handle child_size,
             nb::handle cluster_rows) {
            // Support np.memmap and np.ndarray input types for sklearn
            // pickling. The output of np.asarray can cast to nanobind ndarrays.
            auto const asarray = nb::module_::import_("numpy").attr("asarray");
            new (t) CondensedTree(
                nb::cast<array_ref<uint32_t const>>(asarray(parent), false),
                nb::cast<array_ref<uint32_t const>>(asarray(child), false),
                nb::cast<array_ref<float const>>(asarray(distance), false),
                nb::cast<array_ref<float const>>(asarray(child_size), false),
                nb::cast<array_ref<uint32_t const>>(
                    asarray(cluster_rows), false
                )
            );
          },
          nb::arg("parent"), nb::arg("child"), nb::arg("distance"),
          nb::arg("child_size"), nb::arg("cluster_rows"),
          nb::sig(
              "def __init__(self, parent: np.ndarray[tuple[int], "
              "np.dtype[np.uint32]], child: np.ndarray[tuple[int], "
              "np.dtype[np.uint32]], distance: np.ndarray[tuple[int], "
              "np.dtype[np.float32]], child_size: np.ndarray[tuple[int], "
              "np.dtype[np.float32]], cluster_rows: np.ndarray[tuple[int], "
              "np.dtype[np.uint32]]) -> None"
          ),
          R"(
            Parameters
            ----------
            parent
                An array of parent cluster indices. Clusters are labelled
                with indices starting from the number of points.
            child
                An array of child node and cluster indices. Clusters are labelled
                with indices starting from the number of points.
            distance
                The distance at which the child side connects to the parent side.
            child_size
                The (weighted) size in the child side of the link.
            cluster_rows
                The row indices with a cluster as child.
          )"
      )
      .def_ro(
          "parent", &CondensedTree::parent, nb::rv_policy::reference,
          "A 1D array of parent cluster indices."
      )
      .def_ro(
          "child", &CondensedTree::child, nb::rv_policy::reference,
          "A 1D array of child cluster indices."
      )
      .def_ro(
          "distance", &CondensedTree::distance, nb::rv_policy::reference,
          "A 1D array of distances."
      )
      .def_ro(
          "child_size", &CondensedTree::child_size, nb::rv_policy::reference,
          "A 1D array of child sizes."
      )
      .def_ro(
          "cluster_rows", &CondensedTree::cluster_rows,
          nb::rv_policy::reference, "A 1D array of cluster row indices."
      )
      .def(
          "__iter__",
          [](CondensedTree const &self) {
            return nb::make_tuple(
                       self.parent, self.child, self.distance, self.child_size,
                       self.cluster_rows
            )
                .attr("__iter__")();
          }
      )
      .def(
          "__reduce__",
          [](CondensedTree &self) {
            return nb::make_tuple(
                nb::type<CondensedTree>(),
                nb::make_tuple(
                    self.parent, self.child, self.distance, self.child_size,
                    self.cluster_rows
                )
            );
          }
      )
      .doc() = "CondensedTree contains a pruned dendrogram.";

  m.def(
      "compute_condensed_tree", &compute_condensed_tree,
      nb::arg("linkage_tree"), nb::arg("minimum_spanning_tree"),
      nb::arg("num_points"), nb::arg("min_cluster_size") = 5.0f,
      nb::arg("sample_weights") = nb::none(),
      R"(
        Prunes a linkage tree to create a condensed tree.

        Parameters
        ----------
        linkage_tree
            The input linkage tree. Must originate from and have the same size
            as the spanning tree.
        spanning_tree
            The input minimum spanning tree (sorted).
        min_cluster_size
            The minimum size of clusters to be included in the condensed tree.
            Default is 5.0.
        sample_weights
            The data point sample weights. If not provided, all
            points get an equal weight. Must have a value for each data point!

        Returns
        -------
        condensed_tree
            A CondensedTree with parent, child, distance, child_size,
            and cluster_rows arrays. The child_size array contains the
            (weighted) size of the child cluster, which is the sum of the
            sample weights for all points in the child cluster. The cluster_rows
            array contains the row indices with clusters as child.
        )"
  );
}