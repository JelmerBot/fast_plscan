---
description: "Use when writing, editing, or reviewing Python code in src/fast_plscan/. Covers the public vs internal API boundary, sklearn estimator conventions, parameter and input validation patterns, input preprocessing pipeline, and C++ assumptions about inputs."
applyTo: "src/fast_plscan/*.py"
---

# Python API Guidelines

For documentation and docstring updates, also follow `documentation.instructions.md`.

## Public vs Internal API

The public API is defined exclusively in `__init__.py` via `__all__`:

| Symbol | Source |
|---|---|
| `PLSCAN` | `sklearn.py` — sklearn-compatible estimator |
| `compute_mutual_spanning_tree` | `api.py` — functional, feature-vector input |
| `extract_mutual_spanning_forest` | `api.py` — functional, CSR distance matrix input |
| `clusters_from_spanning_forest` | `api.py` — functional, spanning tree input |
| `get_distance_callback` | `api.py` — testing utility (see below) |

Everything else is internal:

- `_helpers.py` — CSR/kNN converters, persistence helpers. Import with a leading `_`. Never re-export.
- `plots.py` — returned by fitted `PLSCAN` properties, not imported directly by users.
- `_api` (C++ extension) — never imported from user code; accessed only through the Python API layers.

`get_distance_callback` is public for testing purposes only: it lets callers verify that the C++ metric implementations produce the same results as `sklearn.metrics.pairwise_distances`. It is not wired into any compute path.

## Module Responsibilities

| Module | Purpose |
|---|---|
| `sklearn.py` | `PLSCAN` class. All user-facing parameter validation, input dispatch, thread management, and property accessors. |
| `api.py` | Three functional pipeline steps. Each accepts already-validated inputs and delegates directly to `_api`. |
| `_helpers.py` | Internal converters: `sort_spanning_tree`, `most_persistent_clusters`, `knn_to_csr`, `distance_matrix_to_csr`, `remove_self_loops`. |
| `plots.py` | Thin wrappers around `_api` tree objects. Returned by `PLSCAN` properties. |
| `_api/` | C++ extension. No Python logic. |

## Sklearn Estimator Conventions

### Parameter validation

Declare all constructor parameters in `_parameter_constraints` using sklearn validators. Use `Interval`, `StrOptions`, and `None` (for optional parameters):

```python
_parameter_constraints = dict(
    min_samples=[Interval(Integral, 2, None, closed="left")],
    space_tree=[StrOptions({"auto", "kd_tree", "ball_tree"})],
    metric=[StrOptions({*VALID_BALLTREE_METRICS, "precomputed"})],
    num_threads=[None, Interval(Integral, 1, None, closed="left")],
)
```

Begin `fit()` with `self._validate_params()`. Additional cross-parameter checks (e.g., `min_cluster_size >= min_samples`, `metric_kws` only for Minkowski) are done manually afterwards and raise `InvalidParameterError`.

### Fitted attributes

Public fitted attributes use a trailing `_` suffix. They are set only in `fit()` and are `None` class-level defaults before fitting. Internal fitted state uses a leading `_` prefix:

```python
# Public (documented, numpy type-annotated)
labels_: np.ndarray[tuple[int], np.dtype[np.int64]] = None
probabilities_: np.ndarray[tuple[int], np.dtype[np.float32]] = None
core_distances_: np.ndarray[tuple[int], np.dtype[np.float32]] = None
selected_clusters_: np.ndarray[tuple[int], np.dtype[np.intp]] = None

# Internal (not documented as public API)
self._minimum_spanning_tree  # SpanningTree C++ object
self._condensed_tree         # CondensedTree C++ object
self._leaf_tree              # LeafTree C++ object
self._linkage_tree           # LinkageTree C++ object
self._persistence_trace      # PersistenceTrace C++ object
self._mutual_graph           # SparseGraph or None
self._neighbors              # kNN indices ndarray or None
self._num_points             # int
```

Guard every property and post-fit method with `check_is_fitted(self, "_attribute_name")`.

### Properties wrapping internal state

`persistence_trace_`, `leaf_tree_`, and `condensed_tree_` are `@property` accessors that wrap internal C++ objects in `plots.*` classes. They do not return the raw C++ objects directly.

`single_linkage_tree_` and `minimum_spanning_tree_` return `np.column_stack(...)` numpy arrays in scipy linkage format.

## Input Validation and Preprocessing Pipeline

### Feature vector input (`metric != "precomputed"`)

1. Resolve `space_tree`: if `"auto"`, pick `"kd_tree"` when `metric in KDTree.valid_metrics`, else `"ball_tree"`.
2. Validate with `validate_data(self, X, dtype=np.float32, ensure_min_samples=min_samples+1)`. This coerces to `float32` and checks finiteness.
3. Call `compute_mutual_spanning_tree(X, ...)` in `api.py`.

Inside `api.py`:
- `V` (variance) for `seuclidean` and `VI` (inverse covariance) for `mahalanobis` are computed from `data` only when the user has not already provided them in `metric_kws`. User-supplied values always take precedence.
- Build a sklearn `KDTree32`/`BallTree32` from the float32 data.
- Call `.get_arrays()` to extract raw numpy arrays and pass them to `SpaceTree(...)`.
- Query kNN with `min_samples + 1` neighbors (the +1 accounts for the explicit self-loop that sklearn's trees always return as the first neighbor on each row).
- Call `extract_core_distances(knn, min_samples, is_sorted=True)` — kNN rows are already distance-sorted.
- Compute the spanning tree, then call `sort_spanning_tree` to sort edges by distance ascending.

### Precomputed distance input (`metric == "precomputed"`)

Dispatch in `_check_input`:

| Input type | Format | is_sorted | is_mst |
|---|---|---|---|
| `tuple(edges_2d, num_points)` | (parent, child, distance) edge array, shape (n_edges, 3) | True | True |
| `tuple(distances_2d, indices_2d)` | kNN graph, self-loop in column 0 | True | False |
| `np.ndarray` (1D or 2D square) | Condensed or full distance matrix | False | False |
| `csr_array` | Sparse distance matrix | False | False |

- **MST tuple**: cast columns to `uint32`/`uint32`/`float32`, wrap in `SpanningTree` directly. `core_distances_` and `_mutual_graph` are `None`.
- **kNN tuple**: validated with `check_array`, then converted via `knn_to_csr`. The full array (including the self-loop column) is placed into the CSR matrix; the zero-distance self-loop entries are then silently removed by `eliminate_zeros()`.
- **Dense matrix**: `squareform` on 1D input; diagonal filled with 0; converted to CSR via `distance_matrix_to_csr`, zeros eliminated.
- **Sparse CSR**: self-loops removed via `remove_self_loops`; data cast to `float32`, indices to `int32`.

After precomputed dispatch, call `extract_mutual_spanning_forest(X, min_samples=..., is_sorted=...)`.

Inside `api.py`'s `extract_mutual_spanning_forest`:
- Note: `min_samples` passed to `extract_core_distances` is `min_samples - 1` because precomputed CSR graphs have **no** explicit self-loop.
- Compute mutual reachability weights, then extract the spanning forest, then sort.

### Sample weights

Validated with `_check_sample_weight(..., dtype=np.float32, ensure_non_negative=True)`. Additionally checked: no weight may exceed `min_cluster_size` (would create a single-point cluster larger than the minimum size threshold).

### Thread count

Set via `set_num_threads(self.num_threads)` before any C++ call when `num_threads is not None`. Reset to `get_max_threads()` at the end of `fit()`.

## C++ Assumptions About Inputs

All C++ functions receive already-validated and converted data. The Python layer is solely responsible for ensuring these preconditions:

| Assumption | Ensured by |
|---|---|
| Feature data: `float32`, C-contiguous 2D array | `validate_data(..., dtype=np.float32)` |
| CSR arrays: `data` is `float32`, `indices`/`indptr` are `int32` | `remove_self_loops`, `distance_matrix_to_csr`, explicit casts |
| kNN: first column is self-loop (index equals row, distance 0) | sklearn tree `.get_arrays()` contract |
| kNN rows sorted by distance ascending | sklearn tree query always returns sorted neighbors |
| Precomputed CSR: **no** explicit self-loops | `remove_self_loops` / `eliminate_zeros` |
| Spanning tree edges sorted by distance ascending | `sort_spanning_tree` in `api.py` |
| `min_samples` passed to `extract_core_distances`: +1 for kNN (self-loop present), -1 for CSR (no self-loop) | Adjustment in `api.py` |
| `SpaceTree node_data`: viewed as `float64` | `.view(np.float64)` applied to the raw array from `tree.get_arrays()` |
| All distances are non-negative and finite (except kNN/MST bi-persistence) | `check_array(ensure_non_negative=True)` / `validate_data` |

## Type Annotation Style

Use explicit numpy shape/dtype annotations for all array parameters and return types:

```python
data: np.ndarray[tuple[int, int], np.dtype[np.float32]]
labels_: np.ndarray[tuple[int], np.dtype[np.int64]]
sample_weights: np.ndarray[tuple[int], np.dtype[np.float32]] | None
```

Generic parameters use standard `Any` from `typing`. Sparse matrices are typed as `csr_array` (scipy), not `csr_matrix`.
