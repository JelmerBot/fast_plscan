#include "_linkage_tree.h"

#include <nanobind/stl/optional.h>

#include <algorithm>
#include <optional>
#include <vector>

#include "_spanning_tree.h"

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

  size_t idx;
  for (idx = 0; idx < mst.size(); ++idx) {
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

NB_MODULE(_linkage_tree, m) {
  m.doc() = "Module for single-linkage computation in PLSCAN.";

  nb::class_<LinkageTree>(m, "LinkageTree")
      .def(
          nb::init<
              array_ref<uint32_t const>, array_ref<uint32_t const>,
              array_ref<uint32_t const>, array_ref<float const>>(),
          nb::arg("parent").noconvert(), nb::arg("child").noconvert(),
          nb::arg("child_count").noconvert(), nb::arg("child_size").noconvert(),
          R"(
            Parameters
            ----------
            parent
                An array of parent node and cluster indices. Clusters are
                labelled with indices starting from the number of points.
            child
                An array of child node and cluster indices. Clusters are labelled
                with indices starting from the number of points.
            child_count
                The number of points contained in the child side of the link.
            child_size
                The (weighted) size in the child side of the link.
          )"
      )
      .def_ro(
          "parent", &LinkageTree::parent, nb::rv_policy::reference,
          "A 1D array with parent values. "
      )
      .def_ro(
          "child", &LinkageTree::child, nb::rv_policy::reference,
          "A 1D array with child values. "
      )
      .def_ro(
          "child_count", &LinkageTree::child_count, nb::rv_policy::reference,
          "A 1D array with child_count values. "
      )
      .def_ro(
          "child_size", &LinkageTree::child_size, nb::rv_policy::reference,
          "A 1D array with child_size values. "
      )
      .def(
          "__iter__",
          [](LinkageTree const &self) {
            return nb::make_tuple(
                       self.parent, self.child, self.child_count,
                       self.child_size
            )
                .attr("__iter__")();
          }
      )
      .def(
          "__reduce__",
          [](LinkageTree const &self) {
            return nb::make_tuple(
                nb::type<LinkageTree>(),
                nb::make_tuple(
                    self.parent, self.child, self.child_count, self.child_size
                )
            );
          }
      )
      .doc() = "LinkageTree contains a single-linkage dendrogram.";

  m.def(
      "compute_linkage_tree", &compute_linkage_tree,
      nb::arg("minimum_spanning_tree"), nb::arg("num_points"),
      nb::arg("sample_weights") = nb::none(),
      R"(
        Constructs a LinkageTree containing a single-linkage
        dendrogram.

        Parameters
        ----------
        minimum_spanning_tree
            The SpanningTree containing the (sorted/partial) minimum spanning
            tree.
        num_points
            The number of data points in the data set.
        sample_weights
            The data point sample weights. If not provided, all points
            get an equal weight.

        Returns
        -------
        tree
            A LinkageTree containing the parent, child, child_count,
            and child_size arrays of the single-linkage dendrogram.
            Count refers to the number of data points in the child
            cluster. Size refers to the (weighted) size of the child
            cluster, which is the sum of the sample weights for all
            points in the child cluster.
      )"
  );
}