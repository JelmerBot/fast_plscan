#include "_labelling.h"

#include <vector>

#include "_condensed_tree.h"
#include "_leaf_tree.h"

[[nodiscard]] std::vector<int64_t> compute_segment_labels(
    LeafTreeView const leaf_tree, std::span<uint32_t const> const selected
) {
  size_t const num_segments = leaf_tree.size();
  size_t const num_clusters = selected.size();

  // phantom root always gets the noise label
  std::vector<int64_t> segment_labels(num_segments);
  segment_labels[0] = -1;

  int64_t label = 0;
  for (int64_t segment_idx = 1; segment_idx < num_segments; ++segment_idx)
    if (label < num_clusters && selected[label] == segment_idx)
      // bump label if we found the next selected cluster
      segment_labels[segment_idx] = label++;
    else
      // otherwise, inherit the label from the parent segment
      segment_labels[segment_idx] =
          segment_labels[leaf_tree.parent[segment_idx]];
  return segment_labels;
}

[[nodiscard]] std::vector<float> compute_leaf_persistence(
    LeafTreeView const leaf_tree, std::span<uint32_t const> const selected
) {
  size_t const num_clusters = selected.size();
  std::vector<float> leaf_persistence(num_clusters);
  for (size_t label = 0; label < num_clusters; ++label) {
    uint32_t const segment_idx = selected[label];
    leaf_persistence[label] = leaf_tree.max_distance[segment_idx] -
                              leaf_tree.min_distance[segment_idx];
  }
  return leaf_persistence;
}

void fill_labels(
    LabellingWriteView result, LeafTreeView const leaf_tree,
    CondensedTreeView const condensed_tree,
    std::span<uint32_t const> const selected_clusters,
    std::vector<int64_t> const &segment_labels,
    std::vector<float> const &leaf_persistence, size_t const num_points
) {
  // fill in default values to support points without any edges
  std::fill_n(result.label.begin(), num_points, -1u);
  std::fill_n(result.probability.begin(), num_points, 0.0f);

  // iterate over the cluster tree!
  for (size_t idx = 0; idx < condensed_tree.size(); ++idx) {
    size_t const child = condensed_tree.child[idx];
    // skip cluster rows
    if (child >= num_points)
      continue;

    // child is a point, so we can label it
    size_t const parent_idx = condensed_tree.parent[idx] - num_points;
    int64_t const label = segment_labels[parent_idx];
    result.label[child] = label;
    if (label >= 0) {
      float const max_dist = leaf_tree.max_distance[selected_clusters[label]];
      float const point_persistence = max_dist - condensed_tree.distance[idx];
      float const probability = point_persistence / leaf_persistence[label];
      result.probability[child] = std::min(1.0f, probability);
    }
  }
}

void compute_labels(
    LabellingWriteView result, LeafTreeView const leaf_tree,
    CondensedTreeView const condensed_tree,
    std::span<uint32_t const> const selected_clusters, size_t const num_points
) {
  nb::gil_scoped_release guard{};
  auto const segment_labels = compute_segment_labels(
      leaf_tree, selected_clusters
  );
  auto const leaf_persistence = compute_leaf_persistence(
      leaf_tree, selected_clusters
  );
  fill_labels(
      result, leaf_tree, condensed_tree, selected_clusters, segment_labels,
      leaf_persistence, num_points
  );
}

Labelling compute_cluster_labels(
    LeafTree const leaf_tree, CondensedTree const condensed_tree,
    array_ref<uint32_t const> const selected_clusters, size_t const num_points
) {
  auto [label_view, label_cap] = Labelling::allocate(num_points);
  compute_labels(
      label_view, leaf_tree.view(), condensed_tree.view(),
      to_view(selected_clusters), num_points
  );
  return {label_view, std::move(label_cap)};
}

NB_MODULE(_labelling, m) {
  m.doc() = "Module for cluster labelling in PLSCAN.";

  nb::class_<Labelling>(m, "Labelling")
      .def(
          "__init__",
          [](Labelling *t, nb::handle label, nb::handle probability) {
            // Support np.memmap and np.ndarray input types for sklearn
            // pickling. The output of np.asarray can cast to nanobind ndarrays.
            auto const asarray = nb::module_::import_("numpy").attr("asarray");
            new (t) Labelling(
                nb::cast<array_ref<int32_t const>>(asarray(label), false),
                nb::cast<array_ref<float const>>(asarray(probability), false)
            );
          },
          nb::arg("label"), nb::arg("probability"),
          R"(
            Parameters
            ----------
            label
                The data point cluster labels.
            persistence
                The data point cluster membership probabilities.
          )"
      )
      .def_ro(
          "label", &Labelling::label, nb::rv_policy::reference,
          "A 1D array with cluster labels."
      )
      .def_ro(
          "probability", &Labelling::probability, nb::rv_policy::reference,
          "A 1D array with cluster membership probabilities."
      )
      .def(
          "__iter__",
          [](Labelling const &self) {
            return nb::make_tuple(self.label, self.probability)
                .attr("__iter__")();
          }
      )
      .def(
          "__reduce__",
          [](Labelling const &self) {
            return nb::make_tuple(
                nb::type<Labelling>(),
                nb::make_tuple(self.label, self.probability)
            );
          }
      )
      .doc() = "Labelling contains the cluster labels and probabilities.";

  m.def(
      "compute_cluster_labels", &compute_cluster_labels, nb::arg("leaf_tree"),
      nb::arg("condensed_tree"), nb::arg("selected_clusters").noconvert(),
      nb::arg("num_points"),
      R"(
        Computes cluster labels and membership probabilities for the points.

        Parameters
        ----------
        leaf_tree
            The input leaf tree.
        condensed_tree
            The input condensed tree.
        selected_clusters
            The condensed_tree parent IDs of the selected clusters.
        num_points
            The number of points in the condensed tree.

        Returns
        -------
        labelling
            The Labelling containing arrays for the cluster labels and
            membership probabilities. Labels -1 indicate points classified as
            noise.
      )"
  );
}