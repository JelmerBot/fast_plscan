---
description: "Use when writing, editing, or reviewing Python code in src/fast_plscan/. Covers the public vs internal API boundary, sklearn estimator conventions, prediction helpers, parameter and input validation patterns, input preprocessing pipeline, and C++ assumptions about inputs."
applyTo: "src/fast_plscan/*.py"
---

# Python API Guidelines

For documentation and docstring updates, also follow `documentation.instructions.md`.

General function-structure conventions are defined in `../copilot-instructions.md`.

## Public vs Internal API

Public API includes both package-level exports from `__init__.py` and public modules under `src/fast_plscan/`. Package-level public symbols are defined in `__init__.py` via `__all__`. Private
modules use a leading underscore (e.g., `_helpers.py`) and are not re-exported in `__init__.py`. The C++ extension module (`_api`) is entirely internal and never imported directly by user code; it is accessed only through the Python API layers.

`get_distance_callback` is public for testing purposes only: it lets callers verify that the C++ metric implementations produce the same results as `sklearn.metrics.pairwise_distances`. It is not wired into any compute path.

## Module Responsibilities

| Module | Purpose |
|---|---|
| `sklearn.py` | `PLSCAN` class. All user-facing parameter validation, input dispatch, thread management, and property accessors. |
| `api.py` | Free functional implementing the core algorithms. Each accepts already-validated inputs and delegates to helpers in `_api`. |
| `prediction.py` | Public functions for computing cross-cluster memberships and  labels of unseen points. |
| `plots.py` | Thin wrappers around `_api` tree objects implementing visualization logic. Returned by `PLSCAN` properties. |
| `_helpers.py` | Internal converters and re-used helpers. |
| `_api/` | C++ extension implementing the algorithm. No Python logic. |

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

Begin `fit()` with `self._validate_params()`. Additional cross-parameter checks (e.g., `min_cluster_size >= min_samples`, `metric_kws`) are done manually afterwards and raise `InvalidParameterError`.

### Fitted attributes

Public fitted attributes use a trailing `_` suffix. They are set only in `fit()` and are `None` class-level defaults before fitting. Internal fitted state uses a leading `_` prefix.

Guard every property and post-fit method with `check_is_fitted(self, "_attribute_name")`.

### Properties wrapping internal state

`persistence_trace_`, `leaf_tree_`, and `condensed_tree_` are `@property` accessors that wrap internal C++ objects in `plots.*` classes. They do not return the raw C++ objects directly.

`single_linkage_tree_` and `minimum_spanning_tree_` return `np.column_stack(...)` numpy arrays in scipy linkage format.

## C++ Assumptions About Inputs

All C++ functions receive already-validated and converted data. The Python layer is solely responsible for ensuring these preconditions:

`kNN`s are used in **two different paths** with different preconditions:

1. Internal kNN (computed by the implementation from feature input; `metric != "precomputed"`)
2. User-provided graph input (`metric == "precomputed"`, e.g. `(distances, indices)` kNN tuple, dense matrix, or CSR)

### Path A: Internal kNN (feature input)

Preconditions expected by C++ from Python:

- kNN includes explicit self-loop in first column
- kNN rows are sorted by ascending distance
- Distances are non-negative
- Indices are non-negative and less than `n`

### Path B: User-provided precomputed graph input

Preconditions expected by C++ from Python:

- CSR graph has no explicit self-loops
- CSR graph has no explicit zero entries
- CSR distances are positive
- CSR indices are non-negative and less than `n`
- CSR rows are sorted by ascending distance

MST tuple input (`(edges, num_points)`) is a separate precomputed path and bypasses kNN/CSR preprocessing; only edge-shape/value checks are applied before constructing `SpanningTree`.

### Preprocessing conventions by path

1. Feature input (`metric != "precomputed"`): validate feature matrix, build/query space tree, compute core distances from internal kNN, compute mutual-reachability spanning tree, then sort edges by distance.
2. Precomputed graph input (`metric == "precomputed"`): validate shape/content, normalize to canonical CSR dtypes, remove self-loops/explicit zeros, compute core distances with the no-self-loop rank convention, compute mutual-reachability graph and spanning forest, then sort edges by distance.
3. Precomputed MST tuple input: validate tuple/edge invariants and use directly as a sorted spanning tree input (without graph normalization).

## Type Annotation Style

Use explicit numpy shape/dtype annotations for all array parameters and return types:

```python
data: np.ndarray[tuple[int, int], np.dtype[np.float32]]
labels_: np.ndarray[tuple[int], np.dtype[np.int64]]
sample_weights: np.ndarray[tuple[int], np.dtype[np.float32]] | None
```

Sparse matrices are typed as `csr_array` (scipy), not `csr_matrix`.
