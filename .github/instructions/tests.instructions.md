---
description: "Use when writing, editing, or reviewing test code in tests/. Covers pytest conventions, fixture scoping, shared validation helpers in checks.py, metric parametrization sets, conftest structure, and plot regression testing."
applyTo: "tests/**"
---

# Testing Guidelines

## File Organisation

Tests are split by module, class, or topic — one test file per concern. In addition:

| File | Covers |
|---|---|
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

Use shared fixtures from `conftest.py` as the default test inputs.

- Keep expensive datasets and derived artifacts (`distances`, graph forms, tree forms) session-scoped unless a change requires finer isolation.
- Reuse canonical numerical and boolean datasets from fixtures instead of generating ad-hoc data inside tests.
- Keep fixture-generated data deterministic (fixed seeds and stable transforms) so results are reproducible across operating systems and CI environments.
- Model each input family with a canonical fixture set: feature arrays, precomputed distances, sparse graphs, kNN-style inputs, and tree-based inputs.

### Metric classification sets

Use metric classification sets from `conftest.py` to parametrize tests.

- Separate boolean-only metrics from numerical metrics.
- Exclude alias metrics when they are semantically duplicates, unless a test explicitly checks alias handling.
- Match dataset type to metric family (boolean dataset for boolean metrics, numerical dataset otherwise).

## `checks.py` — Shared Validation Helpers

`checks.py` defines `valid_*` functions that centralise all dtype, shape, and invariant assertions for the types returned by the library. Import them with `from .checks import *`. **Never** duplicate their assertions inline in a test. Call the appropriate helper instead.

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

Apply this pattern for all precomputed inputs (MST arrays, distance matrices, kNN arrays, CSR graphs). Don't repeat this check in tests focussed on properties or helper functions of the PLSCAN estimator.

### Error path tests

Use `InvalidParameterError` (from `sklearn.utils._param_validation`) for parameter constraint failures, `ValueError` for data shape/content failures, and `NotFittedError` for calling post-fit methods before fitting.

### Sklearn compatibility

Run `check_estimator(PLSCAN(...))` in `test_sklearn.py` to verify the estimator passes sklearn's standard checks. This is a single test, not parametrized.

## Plot Regression Tests (`test_plots.py`)

Use `@image_comparison` from `matplotlib.testing.decorators`. Baseline images live in `tests/test_plots/baseline_images/`. Plot tests do not call any `valid_*` helpers — visual output is validated by the image comparison alone.

## C++ Extension Access in Tests

Never import from `fast_plscan._api` directly in `test_api.py` or `test_sklearn.py`. Use the public Python API. Direct `_api` access is limited to:

- `test_internals.py` — testing the low-level `kdtree_query`/`balltree_query`/`check_node_data` functions.
- `checks.py` — importing C++ types for `isinstance` checks.
- `conftest.py` — `set_num_threads` / `get_max_threads` for thread control.
