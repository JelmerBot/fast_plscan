#include "labelling.h"

#include <vector>

// --- Implementation details

namespace {

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

}  // namespace

// --- Function API

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

// --- Class API

Labelling::Labelling(
    array_ref<int32_t const> const label,
    array_ref<float const> const probability
)
    : label(label), probability(probability) {}

Labelling::Labelling(LabellingWriteView const view, LabellingCapsule cap)
    : label(to_array(view.label, std::move(cap.label), view.label.size())),
      probability(to_array(
          view.probability, std::move(cap.probability), view.probability.size()
      )) {}

std::pair<LabellingWriteView, LabellingCapsule> Labelling::allocate(
    size_t const num_points
) {
  auto [label, label_cap] = new_buffer<int64_t>(num_points);
  auto [prob, prob_cap] = new_buffer<float>(num_points);
  return {{label, prob}, {std::move(label_cap), std::move(prob_cap)}};
}

LabellingView Labelling::view() const {
  return {to_view(label), to_view(probability)};
}

size_t LabellingWriteView::size() const {
  return label.size();
}

size_t LabellingView::size() const {
  return label.size();
}

size_t Labelling::size() const {
  return label.size();
}
