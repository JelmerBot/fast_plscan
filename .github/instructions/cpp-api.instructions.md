---
description: "Use when writing, editing, or reviewing C++ code in src/fast_plscan/_api/. Covers nanobind binding patterns, numpy array-like input handling for sklearn pickling support, GIL release/acquire conventions, OpenMP parallelism, and array type aliases."
applyTo: "src/fast_plscan/_api/**"
---

# C++ Extension (`_api`) Guidelines

For documentation and docstring updates, also follow `documentation.instructions.md`.

General function-structure conventions are defined in `../copilot-instructions.md`.

## Internal API and ABI Policy

- The C++ bindings, C++ ABI, and `_api` internal C++ interfaces are implementation details under project control.
- They may be changed, refactored, split, merged, renamed, or reordered when those changes are approved.
- Backward compatibility is not required at the internal C++ API/ABI layer, changes are allowed as long as the calling Python code is updated accordingly.

## Array Type Aliases

All array arguments use the type aliases defined in `array.h`. Use these consistently:

```cpp
// 1D numpy C-contiguous array (most common)
template <typename scalar_t>
using array_ref = nb::ndarray<scalar_t, nb::ndim<1>, nb::numpy, nb::c_contig>;

// 2D numpy C-contiguous array
template <typename scalar_t, int N>
using ndarray_ref = nb::ndarray<scalar_t, nb::ndim<N>, nb::numpy, nb::c_contig>;
```

Convert to `std::span` inside C++ functions via `to_view()`, and use `row_view()` for rows of a 2D array. Never work with raw `.data()` pointers directly when a `span` is available.

Allocate output arrays with the `new_array<T>(size)` and `new_buffer<T>(size)` helpers in `array.h` — these manage ownership via `nb::capsule` and avoid manual `new`/`delete`.

## GIL Release/Acquire Pattern

All computationally intensive functions must release the GIL for the duration of their work using an RAII guard declared as the **first statement** in the function body, after all Python object arguments have been resolved:

```cpp
SpanningTree compute_spanning_forest(SpaceTree tree, ...) {
  nb::gil_scoped_release guard{};
  // All C++ computation here — no Python API calls allowed
  #pragma omp parallel for ...
  for (...) { ... }
  return result;
}
```

The GIL is **never explicitly re-acquired** in this codebase. Python callback objects (e.g., custom distance functions obtained via `get_dist`) exist only so users can test that the C++ metric implementations produce the same results as `sklearn.metrics.pairwise_distances`. They are **never passed back into C++ compute functions** that release the GIL. All parallelised kernels use pure C++ distance functors resolved at compile time. A future extension may retrieve the underlying C++ function pointer and pass it back into a compute function, but that is not yet supported.

## Memory Ownership: Owning Class / WriteView / View

Every logical module follows the same three-tier memory ownership model. Each module defines:

| Type | Role |
|---|---|
| `FooWriteView` | Mutable `std::span` views into pre-allocated buffers. Passed into compute functions for writing. |
| `FooView` | Const `std::span` views. Used when a module reads another module's data without taking ownership. |
| `Foo` (owning) | Holds `array_ref<T const>` fields backed by `nb::capsule`s. Owns memory and lives on the Python side. |
| `FooCapsule` | Groups the capsules required to construct an owning `Foo` from a `FooWriteView`. |

**SpaceTree** is the exception: it is always constructed by sklearn/numpy on the Python side and has no `WriteView`. It exposes only a `SpaceTreeView` for C++ consumption.

### Lifecycle

1. **Allocate** — call `Foo::allocate(size)` (a static factory) which returns a `(FooWriteView, FooCapsule)` pair via `new_buffer`.
2. **Fill** — pass the `FooWriteView` into the compute function (with the GIL released). The function writes into the spans.
3. **Convert** — construct the owning `Foo` from `(FooWriteView, FooCapsule, actual_size)`. The constructor calls `to_array()` to produce the `array_ref` fields.
4. **Return** — the owning `Foo` is returned by value to Python, which becomes the sole owner.

```cpp
// Example: spanning_tree.cpp
SpanningTree extract_spanning_forest(SparseGraph graph) {
  auto [mst_view, mst_cap] = SpanningTree::allocate(graph.size() - 1u);
  size_t num_edges = process_graph(mst_view, ...);
  return {mst_view, std::move(mst_cap), num_edges};  // Owning type
}
```

### Consumer functions use View, not owning type

When an owning object is passed as *input* to another compute function, the function takes a `FooView` (const spans), not the owning `Foo`. Call `.view()` on the owning object at the call site:

```cpp
// Correct: intermediate function receives a read-only view
LeafTree compute_leaf_tree(
    CondensedTree const condensed_tree, ...
) {
  CondensedTreeView const condensed_view = condensed_tree.view();
  process_clusters(tree_write_view, condensed_view, ...);  // view only
  ...
}
```

Never pass an owning type by reference into a compute function that will outlive the Python object. Ownership always remains on the Python side.

## Numpy Array-Like Inputs for Sklearn Pickling

Sklearn's `pickle`/`__reduce__` round-trip can store fitted attributes as `np.memmap` objects rather than plain `np.ndarray`. Nanobind cannot cast `np.memmap` directly to `array_ref<T>`. The fix: **call `numpy.asarray()` on every array argument in `__init__`** before casting, so that any array-like (memmap, subclass, etc.) is normalised to a plain `ndarray`:

```cpp
.def(
    "__init__",
    [](MyObject *t, nb::handle arr) {
        // Support np.memmap and np.ndarray for sklearn pickling.
        // np.asarray produces a plain ndarray that nanobind can cast.
        auto const asarray = nb::module_::import_("numpy").attr("asarray");
        new (t) MyObject(
            nb::cast<array_ref<float const>>(asarray(arr), false)
        );
    },
    nb::arg("arr")
)
```

The second argument `false` to `nb::cast` disables an implicit copy — data is borrowed, so the object must not outlive the source array.

Every class that is a fitted sklearn attribute **must** implement both `__iter__` (for tuple unpacking) and `__reduce__` (for pickling):

```cpp
.def("__iter__", [](MyObject const &self) {
    return nb::make_tuple(self.field_a, self.field_b).attr("__iter__")();
})
.def("__reduce__", [](MyObject const &self) {
    return nb::make_tuple(
        nb::type<MyObject>(),
        nb::make_tuple(self.field_a, self.field_b)
    );
})
```

Read-only array fields exposed to Python use `nb::rv_policy::reference` to avoid copies:

```cpp
.def_ro("field_a", &MyObject::field_a, nb::rv_policy::reference, "…")
```

## OpenMP Parallelism

- Use `#pragma omp parallel for default(none) shared(...) [reduction(...)]` for all parallel loops. List every variable explicitly — never rely on implicit sharing.
- Custom OpenMP reductions (e.g., merging edge vectors) are declared with `#pragma omp declare reduction` near the top of the `.cpp` file.
- `clang-format` is suppressed on pragma lines using the `// clang-format off … // clang-format on` pair on the same line:

```cpp
// clang-format off
#pragma omp parallel for default(none) shared(data, result)  // clang-format on
for (int32_t i = 0; i < n; ++i) { ... }
```

- `NB_INLINE` (nanobind's force-inline hint) is used on hot inner methods called from parallel loops.

## Binding Organisation

Each logical module gets its own `add_*_bindings(nb::module_ &m)` function in `bindings.cpp`. The `NB_MODULE` entry point calls each in order.

Docstrings use raw string literals with NumPy docstring style (`Parameters`, `Returns`):

```cpp
m.def("my_func", &my_func, nb::arg("x"), nb::arg("y") = 0,
    R"(
      Short summary.

      Parameters
      ----------
      x
          Description of x.
      y
          Description of y. Default is 0.

      Returns
      -------
      result
          Description of result.
    )"
);
```

## Metric Dispatch

Treat metric dispatch as a single, explicit contract shared by bindings, tree query, and distance kernels.

- Keep one canonical mapping from Python metric names to internal metric identifiers.
- If dispatch logic depends on enum order or lookup-table position, document that dependency at the declaration site and update all dependent tables together.
- Constrain template specializations with concepts or equivalent compile-time checks so KDTree/BallTree compatibility is enforced at compile time.
- When adding a metric, update all dispatch layers in one cohesive change: Python exposure, internal identifier mapping, compatibility guards, and tests.
