---
description: "Add one or more tests to the fast-plscan test suite. Follows conftest fixtures, checks.py helpers, and project pytest conventions."
name: "Add Test"
argument-hint: "Describe what to test (e.g. 'PLSCAN with sample_weights', 'clusters_from_spanning_forest empty MST error')"
agent: "agent"
---

Add tests to the fast-plscan test suite for: $input

Follow the [test conventions](../instructions/tests.instructions.md) precisely.

## Steps

1. **Identify the correct test file** based on what is being tested:
   - `test_api.py` — functional API (`compute_mutual_spanning_tree`, `extract_mutual_spanning_forest`, `clusters_from_spanning_forest`)
   - `test_sklearn.py` — `PLSCAN` estimator (inputs, parameters, sklearn compatibility)
   - `test_distances.py` — `get_distance_callback` metric parity
   - `test_space_tree.py` — `kdtree_query` / `balltree_query` internals
   - `test_plots.py` — matplotlib plot output (image regression)

2. **Reuse existing fixtures** from `conftest.py` — never create inline data. Available session fixtures:
   - `X` — float32, 200×2, StandardScaler-normalised blobs
   - `X_bool` — float32, 200×100, boolean-valued
   - `dists`, `con_dists` — full and condensed distance matrices
   - `g_dists`, `g_knn` — CSR sparse graphs
   - `knn`, `knn_no_loops` — kNN tuples with and without self-loop column
   - `mst` — precomputed MST edge array (parent, child, distance)
   - `kdtree`, `balltree` — `KDTree32` / `BallTree32` on `X`

3. **Validate all outputs with `valid_*` helpers** from `checks.py` (imported via `from .checks import *`). Add narrow result-specific assertions (e.g. `assert labels.max() == 2`) after the helpers, not instead of them.

4. **Apply `@pytest.mark.parametrize`** for metric or space-tree variants. Use the metric sets from `conftest`:
   - `numerical_balltree_metrics - duplicate_metrics` for numerical metrics
   - `boolean_metrics` paired with `X_bool`

5. **For error-path tests**, name the function `test_bad_<parameter>` and use one `pytest.raises` per invalid case (no loops). Use `InvalidParameterError` for parameter constraints, `ValueError` for data problems.

6. **For input mutation tests**, copy the input before calling the library and assert it is unchanged afterwards.

7. **Do not call `_api` directly** in `test_api.py` or `test_sklearn.py`. Use the public Python API only.

## Output

Write the complete test function(s) and insert them into the appropriate existing test file at a logical location (grouped by theme). Do not create new test files unless the topic has no existing home.
