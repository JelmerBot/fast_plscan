# fast-plscan — Project Guidelines

## Architecture

The library is split into a Python layer and a C++ core:

- `src/fast_plscan/_api/` — C++ extension compiled with **nanobind** and OpenMP (C++23).
  Key modules: `space_tree`, `spanning_tree`, `distances`, `labelling`, `condensed_tree`, `leaf_tree`, `persistence_trace`.
- `src/fast_plscan/api.py` — Functional Python API (`compute_mutual_spanning_tree`, `extract_mutual_spanning_forest`, `clusters_from_spanning_forest`).
- `src/fast_plscan/sklearn.py` — Scikit-learn-compatible `PLSCAN` estimator (`BaseEstimator`, `ClusterMixin`).
- `src/fast_plscan/plots.py` — Visualization classes (`LeafTree`, `CondensedTree`, `PersistenceTrace`).
- `src/fast_plscan/_helpers.py` — Internal utilities (CSR conversions, persistence helpers).
- `src/fast_plscan/__init__.py` — Public API; all exported symbols listed in `__all__`.

Public entry-point: `from fast_plscan import PLSCAN` (sklearn interface) or the functional API functions.

## Build and Test

**Install for development** (requires CMake ≥ 3.18, a C++23 compiler, and the clang-cl toolchain on Windows):
```sh
pip install nanobind scikit-build-core[pyproject]
pip install --no-build-isolation -ve .
```
Add `--config-settings editable.rebuild=true` to auto-rebuild C++ on import.

**Run tests:**
```sh
pytest .
```
Test dependencies: `pytest`, `networkx`, `pandas`.

**Windows notes:** The build uses `-T ClangCL`; wheels are repaired with `delvewheel`.

## Conventions

Detailed conventions are maintained in per-topic instruction files that load automatically when relevant files are in context:

- **C++ extension** (`src/fast_plscan/_api/`): see [instructions/cpp-api.instructions.md](instructions/cpp-api.instructions.md) — nanobind binding patterns, GIL release, memory ownership (owning class / WriteView / View), OpenMP, sklearn pickling support.
- **Python API** (`src/fast_plscan/*.py`): see [instructions/python-api.instructions.md](instructions/python-api.instructions.md) — public vs internal API boundary, sklearn estimator conventions, input validation, preprocessing pipeline, C++ input assumptions.
- **Documentation** (`docs/`, docstrings, API reference): see [instructions/documentation.instructions.md](instructions/documentation.instructions.md) — Sphinx structure, autosummary boundaries, docstring conventions, and notebook documentation patterns.
- **Tests** (`tests/`): see [instructions/tests.instructions.md](instructions/tests.instructions.md) — fixture scoping, `valid_*` helpers, metric parametrization sets, error-path conventions, plot regression tests.

### Quick reference
- **Type hints** use explicit numpy shapes/dtypes: `np.ndarray[tuple[int, int], np.dtype[np.float32]]`.
- Fitted sklearn attributes carry a trailing `_` (e.g., `labels_`, `leaf_tree_`).
- All tests run single-threaded via `pytest_sessionstart`; never call `set_num_threads` inside individual tests.
- Validate test outputs with `valid_*()` helpers from `checks.py` — never duplicate inline dtype/shape assertions.

## Key Dependencies

| Package | Version |
|---------|---------|
| numpy | ≥2, <3 |
| scipy | ≥1, <2 |
| scikit-learn | ≥1.6, <2 |
| matplotlib | ≥3, <4 |

Python 3.10–3.14 supported (ABI3 stable wheel tagged `cp312-abi3`).
