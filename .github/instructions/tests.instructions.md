---
description: "Use when writing, editing, or reviewing test code in tests/. Covers pytest conventions, fixture scoping, shared validation helpers in checks.py, metric parametrization sets, conftest structure, and plot regression testing."
applyTo: "tests/**"
---

# Testing Guidelines

## File Organisation

Tests are split by module, class, or topic — one test file per concern:

| File | Covers |
|---|---|
| `test_api.py` | Functional API: `compute_mutual_spanning_tree`, `extract_mutual_spanning_forest`, `clusters_from_spanning_forest` |
| `test_sklearn.py` | `PLSCAN` estimator: all input modes, all parameters, sklearn compatibility checks |
| `test_distances.py` | `get_distance_callback` parity against `sklearn.metrics.pairwise_distances` |
| `test_space_tree.py` | `kdtree_query` / `balltree_query` and `check_node_data` |
| `test_plots.py` | Matplotlib image regression tests for all plot types |
| `conftest.py` | Session-scoped fixtures and metric classification sets |
| `checks.py` | Shared `valid_*` assertion helpers |

## `conftest.py` — Fixtures and Thread Control

### Thread control hooks

Threading is set to single-threaded for the entire session via pytest hooks, not per-test:

```python
def pytest_sessionstart(session):
    set_num_threads(1)

def pytest_sessionfinish(session, exitstatus):
    set_num_threads(get_max_threads())
```

Never call `set_num_threads` inside individual tests.

### Fixtures

All data fixtures are `scope="session"` — they are created once and reused across the entire test run. Do not change this to function scope without a strong reason, as many fixtures are expensive to compute.

The fixture dependency chain is:

```
X (float32, 200×2, StandardScaler, make_blobs)
├── con_dists  (pdist condensed, float32)
│   └── dists  (squareform full matrix)
│       └── g_dists (CSR, no self-loops, eliminate_zeros)
│           └── mst (scipy MST → sorted (parent, child, dist) array, float64)
├── knn        (NearestNeighbors 8 neighbours, WITH self-loop, 5 rows have inf/-1 missing edges)
│   └── g_knn  (knn_to_csr)
└── knn_no_loops (NearestNeighbors 8 neighbours, WITHOUT self-loop, same missing edges)

X_bool (float32, 200×100, Bernoulli p=0.25, PCG64 seed 10)

kdtree  (KDTree32 on X)
balltree (BallTree32 on X)
```

Always use the existing fixtures rather than constructing data inline. Never create ad-hoc `make_blobs` or `np.random` calls inside tests — the fixtures already use fixed seeds (`random_state=10/7`, `PCG64(10)`) to guarantee reproducibility across OSes.

### Metric classification sets

Three sets are exported from `conftest.py` and imported into test files:

```python
boolean_metrics          # metrics that require boolean (float32 0/1) input
numerical_balltree_metrics    # all VALID_BALLTREE_METRICS minus boolean_metrics
duplicate_metrics        # aliases ("p", "infinity", "manhattan", "l1", "l2") — excluded
                         # from parametrize to avoid running the same test twice
```

Use `numerical_balltree_metrics - duplicate_metrics` or `set(PLSCAN.VALID_KDTREE_METRICS) - duplicate_metrics` as the parametrize source when covering metrics. Use `X` for numerical metrics and `X_bool` for boolean metrics.

## `checks.py` — Shared Validation Helpers

`checks.py` defines `valid_*` functions that centralise all dtype, shape, and invariant assertions for the types returned by the library. Import them with `from .checks import *`.

**Never** duplicate their assertions inline in a test. Call the appropriate helper instead:

| Helper | Validates |
|---|---|
| `valid_spanning_forest(msf, X)` | `SpanningTree`, sorted distances, non-negative indices, `≤ n-1` edges |
| `valid_neighbor_indices(indices, X, min_samples)` | shape `(n, min_samples+1)`, `int32`, all in `[0, n)` |
| `valid_mutual_graph(mut_graph, X, *, missing=False)` | `SparseGraph`, correct `indptr`, sorted rows; `missing=True` allows `-1` indices |
| `valid_core_distances(cd, X)` | `ndarray`, finite, shape `(n,)` |
| `valid_labels(labels, X)` | `int64`, shape `(n,)`, all `≥ -1` |
| `valid_probabilities(probs, X)` | `float32`, shape `(n,)`, finite, all `≥ 0` |
| `valid_selected_clusters(sel, labels)` | `uint32`, count matches `labels.max() + 1` unless all-noise |
| `valid_persistence_trace(trace)` | `PersistenceTrace`, `min_size ≥ 2.0`, `persistence ≥ 0.0` |
| `valid_leaf(leaf_tree)` | `LeafTree`, correct dtypes, `min_distance ≤ max_distance` |
| `valid_linkage(linkage_tree, X)` | `LinkageTree`, correct dtypes, parent ≥ child |
| `valid_condensed(condensed_tree, X)` | `CondensedTree`, correct dtypes, parent != child, `parent ≥ n` |

When an entire pipeline is exercised, call all relevant helpers in sequence. Do add narrow result-specific assertions after the helpers (e.g., `assert labels.max() == 2`, `assert np.any(labels == -1)`).

## Pytest Conventions

### Parametrize

Use `@pytest.mark.parametrize` for metric and space-tree variants. Stack decorators for a cartesian product:

```python
@pytest.mark.parametrize("metric", PLSCAN.VALID_KDTREE_METRICS)
@pytest.mark.parametrize("space_tree", ["kd_tree", "ball_tree"])
def test_kdtree_metrics(X, metric, space_tree):
    ...
```

When only one axis is needed, use a single decorator:

```python
@pytest.mark.parametrize("space_tree", ["auto", "kd_tree", "ball_tree"])
def test_one_component(X, space_tree):
    ...
```

### Input mutation guard

Tests that pass mutable inputs to the library must assert the input is unchanged afterwards:

```python
_in = mst.copy()
c = PLSCAN(metric="precomputed").fit((mst, X.shape[0]))
assert np.allclose(mst, _in)
```

Apply this pattern for all precomputed inputs (MST arrays, distance matrices, kNN arrays, CSR graphs).

### Error path tests

Group invalid-input tests in dedicated `test_bad_*` functions. Each invalid case is a separate `pytest.raises` call — never use a loop:

```python
def test_bad_min_samples(X, knn):
    with pytest.raises(InvalidParameterError):
        PLSCAN(min_samples=-1).fit(knn)
    with pytest.raises(InvalidParameterError):
        PLSCAN(min_samples=0).fit(knn)
```

Use `InvalidParameterError` (from `sklearn.utils._param_validation`) for parameter constraint failures, `ValueError` for data shape/content failures, and `NotFittedError` for calling post-fit methods before fitting.

### Sklearn compatibility

Run `check_estimator(PLSCAN(...))` in `test_sklearn.py` to verify the estimator passes sklearn's standard checks. This is a single test, not parametrized.

## Plot Regression Tests (`test_plots.py`)

Use `@image_comparison` from `matplotlib.testing.decorators`. Baseline images live in `tests/baseline_images/test_plots/`. Always set a non-zero `tol` with a comment explaining why:

```python
@image_comparison(
    baseline_images=["condensed_tree_dist"],
    extensions=["png"],
    style="mpl20",
    tol=12.71,  # branches can switch places without changing meaning
)
def test_condensed_tree_dist(knn):
    plt.figure()
    PLSCAN(...).fit(knn).condensed_tree_.plot(...)
```

Plot tests do not call any `valid_*` helpers — visual output is validated by the image comparison alone.

## C++ Extension Access in Tests

Never import from `fast_plscan._api` directly in `test_api.py` or `test_sklearn.py`. Use the public Python API (`fast_plscan.*` and `fast_plscan.api.*`). Direct `_api` access is limited to:

- `test_space_tree.py` — testing the low-level `kdtree_query`/`balltree_query`/`check_node_data` functions.
- `checks.py` — importing C++ types for `isinstance` checks.
- `conftest.py` — `set_num_threads` / `get_max_threads` for thread control.
