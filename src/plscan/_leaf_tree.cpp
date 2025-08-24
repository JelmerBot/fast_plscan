#include "_leaf_tree.h"

#include <algorithm>

#include "_condensed_tree.h"

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

NB_MODULE(_leaf_tree_ext, m) {
  m.doc() = "Module for leaf tree computation in PLSCAN.";

  nb::class_<LeafTree>(m, "LeafTree")
      .def(
          "__init__",
          [](LeafTree *t, nb::handle parent, nb::handle min_distance,
             nb::handle max_distance, nb::handle min_size,
             nb::handle max_size) {
            // Support np.memmap and np.ndarray input types for sklearn
            // pickling. The output of np.asarray can cast to nanobind
            // ndarrays.
            auto const asarray = nb::module_::import_("numpy").attr("asarray");
            new (t) LeafTree(
                nb::cast<array_ref<uint32_t const>>(asarray(parent), false),
                nb::cast<array_ref<float const>>(asarray(min_distance), false),
                nb::cast<array_ref<float const>>(asarray(max_distance), false),
                nb::cast<array_ref<float const>>(asarray(min_size), false),
                nb::cast<array_ref<float const>>(asarray(max_size), false)
            );
          },
          nb::arg("parent"), nb::arg("min_distance"), nb::arg("max_distance"),
          nb::arg("min_size"), nb::arg("max_size"),
          nb::sig(
              "def __init__(self, parent: np.ndarray[tuple[int], "
              "np.dtype[np.uint32]], min_distance: np.ndarray[tuple[int], "
              "np.dtype[np.float32]], max_distance: np.ndarray[tuple[int], "
              "np.dtype[np.float32]], min_size: np.ndarray[tuple[int], "
              "np.dtype[np.float32]], max_size: np.ndarray[tuple[int], "
              "np.dtype[np.float32]]) -> None"
          ),
          R"(
            Parameters
            ----------
            parent
                The parent cluster IDs.
            min_distance
                The minimum distance at which the cluster exists.
            max_distance
                The distance at which the cluster connects to its parent.
            min_size
                The min_cluster_size at which the cluster becomes a leaf.
            max_size
                The min_cluster_size at which the cluster stops being a leaf.
          )"
      )
      .def_ro(
          "parent", &LeafTree::parent, nb::rv_policy::reference,
          "A 1D array with parent cluster IDs."
      )
      .def_ro(
          "min_distance", &LeafTree::min_distance, nb::rv_policy::reference,
          "A 1D array with minimum leaf cluster distances."
      )
      .def_ro(
          "max_distance", &LeafTree::max_distance, nb::rv_policy::reference,
          "A 1D array with maximum leaf cluster distances."
      )
      .def_ro(
          "min_size", &LeafTree::min_size, nb::rv_policy::reference,
          "A 1D array with minimum leaf cluster sizes."
      )
      .def_ro(
          "max_size", &LeafTree::max_size, nb::rv_policy::reference,
          "A 1D array with maximum leaf cluster sizes."
      )
      .def(
          "__iter__",
          [](LeafTree const &self) {
            return nb::make_tuple(
                       self.parent, self.min_distance, self.max_distance,
                       self.min_size, self.max_size
            )
                .attr("__iter__")();
          }
      )
      .def(
          "__reduce__",
          [](LeafTree const &self) {
            return nb::make_tuple(
                nb::type<LeafTree>(),
                nb::make_tuple(
                    self.parent, self.min_distance, self.max_distance,
                    self.min_size, self.max_size
                )
            );
          }
      )
      .doc() = R"(
        LeafTree lists information for the clusters in a condensed tree.

        Indexing with [cluster_id - num_points] gives information for the
        cluster with cluster_id.        
      )";

  m.def(
      "compute_leaf_tree", &compute_leaf_tree, nb::arg("condensed_tree"),
      nb::arg("num_points"), nb::arg("min_cluster_size") = 5.0f,
      R"(
        Computes a leaf tree from a condensed tree.

        Parameters
        ----------
        condensed_tree
            The input condensed tree.
        num_points
            The number of points in the dataset.
        min_cluster_size
            The minimum size of clusters to be included in the leaf tree.

        Returns
        -------
        leaf_tree
            A LeafTree containing parent, min_distance, max_distance, min_size,
            and max_size arrays. The min/max distance arrays contain each
            cluster's distance range. The min_size and max_size arrays contain
            the minimum and maximum min_cluster_size thresholds for which the
            clusters are leaves, respectively. Some clusters never become
            leaves, indicated by a min_size larger than the max_size.
      )"
  );

  m.def(
      "apply_size_cut", &apply_size_cut, nb::arg("leaf_tree"),
      nb::arg("cut_size"),
      R"(
        Finds the cluster IDs for leaf-clusters that exist at the 
        given cut_size threshold. The threshold is interpreted as a
        birth value in a left-open (birth, death] size interval.

        Parameters
        ----------
        leaf_tree
            The input leaf tree.
        size_cut
            The size threshold for selecting clusters. The threshold is 
            interpreted as a birth value in a left-open (birth, death] size 
            interval.

        Returns
        -------
        selected_clusters
            The cluster IDs for leaf-clusters that exist at the 
            given cut_size threshold. 
      )"
  );

  m.def(
      "apply_distance_cut", &apply_distance_cut, nb::arg("leaf_tree"),
      nb::arg("cut_distance"),
      R"(
        Finds the cluster IDs for clusters that exist at the given cut distance 
        threshold.

        Parameters
        ----------
        leaf_tree
            The input leaf tree.
        distance_cut
            The distance threshold for selecting clusters.

        Returns
        -------
        selected_clusters
            The cluster IDs for clusters that exist at the given distance 
            threshold. 
      )"
  );
}