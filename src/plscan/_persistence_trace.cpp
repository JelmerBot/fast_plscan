#include "_persistence_trace.h"

#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <vector>

#include "_condensed_tree.h"
#include "_leaf_tree.h"

// -- Compute persistence trace

size_t initialize_trace(
    PersistenceTraceWriteView result, LeafTreeView const leaf_tree
) {
  // Copy the size thresholds into the trace_min_size buffer.
  size_t const num_leaves = leaf_tree.size();
  for (size_t idx = 1; idx < num_leaves; ++idx) {
    result.min_size[2 * (idx - 1)] = leaf_tree.min_size[idx];
    result.min_size[2 * (idx - 1) + 1u] = leaf_tree.max_size[idx];
  }

  // Find the (number of) unique sizes
  auto const begin_it = result.min_size.begin();
  auto const end_it = result.min_size.end();
  std::sort(begin_it, end_it);
  size_t const trace_size = std::distance(
      begin_it, std::unique(begin_it, end_it)
  );

  // Initialize the persistence trace
  std::fill_n(result.persistence.data(), trace_size, 0.0f);
  return trace_size;
}

[[nodiscard]] auto find_fill_range(
    std::span<float> min_sizes, size_t const trace_size, float const birth,
    float const death
) {
  auto const begin_it = min_sizes.begin();
  auto const end_it = std::next(
      begin_it, static_cast<std::ptrdiff_t>(trace_size)
  );
  auto const start_it = std::lower_bound(begin_it, end_it, birth);
  auto const stop_it = std::lower_bound(start_it, end_it, death);
  return std::make_pair(
      std::distance(begin_it, start_it), std::distance(begin_it, stop_it)
  );
}

template <typename function_t>
void fill_persistences(
    PersistenceTraceWriteView result, LeafTreeView const leaf_tree,
    size_t const trace_size, function_t &&get_persistences
) {
  for (size_t idx = 1; idx < leaf_tree.size(); ++idx) {
    float const birth = leaf_tree.min_size[idx];
    float const death = leaf_tree.max_size[idx];
    if (death <= birth)
      continue;

    float const persistence = get_persistences(idx);
    auto [fill_idx, end_idx] = find_fill_range(
        result.min_size, trace_size, birth, death
    );
    while (fill_idx < end_idx)
      result.persistence[fill_idx++] += persistence;
  }
}

[[nodiscard]] std::vector<float> compute_bi_persistences(
    LeafTreeView const leaf_tree, CondensedTreeView const condensed_tree,
    size_t const num_points
) {
  // Working variables.
  size_t const num_rows = condensed_tree.size();
  size_t const num_leaves = leaf_tree.size();
  std::vector collected(num_leaves, 0.0f);
  std::vector bi_persistences(num_leaves, 0.0f);

  // Iterate over the rows in reverse order.
  for (size_t _i = 1; _i <= num_rows; ++_i) {
    // skip cluster rows
    size_t const idx = num_rows - _i;
    if (condensed_tree.child[idx] >= num_points)
      continue;

    // aggregate point distance-persistence
    float const distance = condensed_tree.distance[idx];
    uint64_t parent_idx = condensed_tree.parent[idx] - num_points;
    // skip roots (i.e. direct children of the phantom root)
    while (leaf_tree.parent[parent_idx] > 0) {
      collected[parent_idx] += condensed_tree.child_size[idx];
      // use greater equals here to match the min_size trace values that use
      // birth in (birth, death] intervals!
      if (collected[parent_idx] >= leaf_tree.min_size[parent_idx] &&
          collected[parent_idx] < leaf_tree.max_size[parent_idx])
        bi_persistences[parent_idx] += leaf_tree.max_distance[parent_idx] -
                                       distance;
      parent_idx = leaf_tree.parent[parent_idx];
    }
  }

  return bi_persistences;
}

size_t fill_size_persistence(
    PersistenceTraceWriteView result, LeafTreeView const leaf_tree
) {
  nb::gil_scoped_release guard{};
  size_t const trace_size = initialize_trace(result, leaf_tree);
  fill_persistences(
      result, leaf_tree, trace_size,
      [leaf_tree](size_t const idx) {
        // skip roots (i.e. direct children of the phantom root)
        return static_cast<float>(leaf_tree.parent[idx] > 0u) *
               (leaf_tree.max_size[idx] - leaf_tree.min_size[idx]);
      }
  );
  return trace_size;
}

size_t fill_bi_persistence(
    PersistenceTraceWriteView result, LeafTreeView const leaf_tree,
    CondensedTreeView const condensed_tree, size_t const num_points
) {
  nb::gil_scoped_release guard{};
  size_t const trace_size = initialize_trace(result, leaf_tree);
  std::vector<float> bi_persistences = compute_bi_persistences(
      leaf_tree, condensed_tree, num_points
  );
  fill_persistences(
      result, leaf_tree, trace_size,
      [&bi_persistences](size_t const idx) { return bi_persistences[idx]; }
  );
  return trace_size;
}

PersistenceTrace compute_size_persistence(LeafTree const leaf_tree) {
  size_t const buffer_size = 2 * (leaf_tree.size() - 1u);
  auto [trace_view, trace_cap] = PersistenceTrace::allocate(buffer_size);
  size_t const trace_size = fill_size_persistence(trace_view, leaf_tree.view());
  return PersistenceTrace{trace_view, std::move(trace_cap), trace_size};
}

PersistenceTrace compute_bi_persistence(
    LeafTree const leaf_tree, CondensedTree const condensed_tree,
    size_t const num_points
) {
  size_t const buffer_size = 2 * (leaf_tree.size() - 1u);
  auto [trace_view, trace_cap] = PersistenceTrace::allocate(buffer_size);
  size_t trace_size = fill_bi_persistence(
      trace_view, leaf_tree.view(), condensed_tree.view(), num_points
  );
  return PersistenceTrace{trace_view, std::move(trace_cap), trace_size};
}

// -- Compute leaf tree icicles

[[nodiscard]] auto collect_traces(
    LeafTreeView const leaf_tree, CondensedTreeView const condensed_tree,
    size_t const num_points
) {
  nb::gil_scoped_release guard{};

  size_t const num_rows = condensed_tree.size();
  size_t const num_leaves = leaf_tree.size();

  // We use re-sizeable vectors here because we don't know the number of points
  // in each leaf cluster in advance. That would require also tracking the child
  // counts in the condensed and leaf trees.
  using trace_t = std::unique_ptr<std::vector<float>>;
  std::vector<trace_t> sizes{num_leaves};
  std::vector<trace_t> stabilities{num_leaves};
  for (size_t idx = 0; idx < num_leaves; ++idx) {
    sizes[idx] = std::make_unique<std::vector<float>>();
    stabilities[idx] = std::make_unique<std::vector<float>>();
  }

  // collect the child points (reverse order).
  std::vector collected(num_leaves, 0.0f);
  for (size_t _i = 1; _i <= num_rows; ++_i) {
    // skip cluster rows
    size_t const idx = num_rows - _i;
    if (condensed_tree.child[idx] >= num_points)
      continue;

    float const distance = condensed_tree.distance[idx];
    uint64_t parent_idx = condensed_tree.parent[idx] - num_points;
    // skip the roots (i.e. direct children of the phantom root)
    while (leaf_tree.parent[parent_idx] > 0) {
      collected[parent_idx] += condensed_tree.child_size[idx];
      // Use greater than (not greater equals) here so the icicles reflect the
      // min_size_threshold rather than birth in (birth, death] intervals!
      // This is possible because we also collect the size value here.
      if (collected[parent_idx] > leaf_tree.min_size[parent_idx]) {
        sizes[parent_idx]->push_back(collected[parent_idx]);
        stabilities[parent_idx]->push_back(
            leaf_tree.max_distance[parent_idx] - distance
        );
      }
      parent_idx = leaf_tree.parent[parent_idx];
    }
  }

  return std::make_pair(std::move(sizes), std::move(stabilities));
}

[[nodiscard]] std::pair<
    std::vector<array_ref<float>>, std::vector<array_ref<float>>>
vectors_to_arrays(
    std::vector<std::unique_ptr<std::vector<float>>> &&sizes,
    std::vector<std::unique_ptr<std::vector<float>>> &&stabilities
) {
  size_t const num_leaves = sizes.size();
  std::vector<array_ref<float>> size_arrays(num_leaves);
  std::vector<array_ref<float>> stability_arrays(num_leaves);

  auto deleter = [](void *ptr) noexcept {
    delete static_cast<std::vector<float> *>(ptr);
  };

  for (size_t idx = 0; idx < num_leaves; ++idx) {
    auto *size_ptr = sizes[idx].release();
    size_arrays[idx] = array_ref<float>(
        size_ptr->data(), {size_ptr->size()}, nb::capsule{size_ptr, deleter}
    );
    auto *stability_ptr = stabilities[idx].release();
    stability_arrays[idx] = array_ref<float>(
        stability_ptr->data(), {stability_ptr->size()},
        nb::capsule{stability_ptr, deleter}
    );
  }

  return std::make_pair(size_arrays, stability_arrays);
}

auto compute_stability_icicles(
    LeafTree const leaf_tree, CondensedTree const condensed_tree,
    size_t const num_points
) {
  auto [sizes, stabilities] = collect_traces(
      leaf_tree.view(), condensed_tree.view(), num_points
  );
  return vectors_to_arrays(std::move(sizes), std::move(stabilities));
}

// --- Module definitions

NB_MODULE(_persistence_trace, m) {
  m.doc() = "Module for persistence trace computation in PLSCAN.";
  nb::class_<PersistenceTrace>(m, "PersistenceTrace")
      .def(
          "__init__",
          [](PersistenceTrace *t, nb::handle min_size, nb::handle persistence) {
            // Support np.memmap and np.ndarray input types for sklearn
            // pickling. The output of np.asarray can cast to nanobind ndarrays.
            auto const asarray = nb::module_::import_("numpy").attr("asarray");
            new (t) PersistenceTrace(
                nb::cast<array_ref<float const>>(asarray(min_size), false),
                nb::cast<array_ref<float const>>(asarray(persistence), false)
            );
          },
          nb::arg("min_size"), nb::arg("persistence"),
          R"(
            Parameters
            ----------
            min_size
                The minimum cluster sizes at which leaf-clusters start to exist.
            persistence
                The persistence sum for the leaf-clusters that exist at the
                minimum cluster sizes.
          )"
      )
      .def_ro(
          "min_size", &PersistenceTrace::min_size, nb::rv_policy::reference,
          "A 1D array with minimum cluster sizes."
      )
      .def_ro(
          "persistence", &PersistenceTrace::persistence,
          nb::rv_policy::reference, "A 1D array with total persistence values."
      )
      .def(
          "__iter__",
          [](PersistenceTrace const &self) {
            return nb::make_tuple(self.min_size, self.persistence)
                .attr("__iter__")();
          }
      )
      .def(
          "__reduce__",
          [](PersistenceTrace const &self) {
            return nb::make_tuple(
                nb::type<PersistenceTrace>(),
                nb::make_tuple(self.min_size, self.persistence)
            );
          }
      )
      .doc() = "PersistenceTrace lists the persistences per min_cluster_size.";

  m.def(
      "compute_size_persistence", &compute_size_persistence,
      nb::arg("leaf_tree"),
      R"(
        Computes the total min_cluster_size persistence trace.

        Parameters
        ----------
        leaf_tree
            The input leaf tree.

        Returns
        -------
        persistence_trace
            A PersistenceTrace containing arrays for the minimum cluster size
            and total persistence values. The min_size array contains all unique
            min_cluster_sizes at which clusters become leaves. The persistence
            array contains the total persistence of leaf clusters at those
            minimum size thresholds.
      )"
  );

  m.def(
      "compute_bi_persistence", &compute_bi_persistence, nb::arg("leaf_tree"),
      nb::arg("condensed_tree"), nb::arg("num_points"),
      R"(
        Computes a leaf tree from a condensed tree.

        Parameters
        ----------
        leaf_tree
            The input leaf tree.
        condensed_tree
            The input condensed tree.
        num_points
            The number of points in the condensed tree.

        Returns
        -------
        persistence_trace
            A PersistenceTrace containing arrays for the minimum cluster size
            and total persistence values. The min_size array contains all unique
            min_cluster_sizes at which clusters become leaves. The persistence
            array contains the total persistence of leaf clusters at those
            minimum size thresholds.
      )"
  );

  m.def(
      "compute_stability_icicles", &compute_stability_icicles,
      nb::arg("leaf_tree"), nb::arg("condensed_tree"), nb::arg("num_points"),
      R"(
        Computes the icicle size--stability trace for the LeafTree plot.

        Parameters
        ----------
        leaf_tree
            The input leaf tree.
        condensed_tree
            The input condensed tree.
        num_points
            The number of points in the condensed tree.
        Returns
        -------
        sizes
            The icicle min cluster sizes (births in (birth, death])).
        stabilities
            The icicle stabilities.
      )"
  );
}