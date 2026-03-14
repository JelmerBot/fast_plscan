---
description: "Add a new API feature to fast-plscan end-to-end: design the API surface, agree on tests, then implement. Follows TDD and the C++ vs Python placement rules."
name: "Add Feature"
argument-hint: "Describe the feature to add (e.g. 'weighted core distances', 'outlier score attribute')"
agent: "agent"
---

Add a new feature to fast-plscan: $input

Follow the [Python API conventions](../instructions/python-api.instructions.md), [C++ extension conventions](../instructions/cpp-api.instructions.md), [test conventions](../instructions/tests.instructions.md), and [documentation conventions](../instructions/documentation.instructions.md).

---

## Phase 1 — Design (do not write implementation code yet)

Keep planning output concise. Prefer short bullet lists over long prose.
Limit the design response to the minimum needed for decision-making.

### 1. API surface

Propose the public API changes:
- New parameters on `PLSCAN.__init__` or `fit()`, or new public functions in `api.py` or `__init__.py`
- New fitted attributes (`name_` suffix) or properties on `PLSCAN`
- New functional API functions if the feature is useful standalone
- Any new types returned to the user

### 2. Placement decision

For each new computation, decide where it lives:

**Put in C++ (`src/fast_plscan/_api/`)** when:
- The algorithm benefits from parallelism (OpenMP loops over points, edges, or clusters)
- It requires tight iteration over large arrays where Python overhead would dominate (e.g. tree traversals, graph algorithms)
- It is a natural extension of an existing C++ module (spanning tree, condensed tree, labelling, etc.)

**Keep in Python** when:
- It can be expressed efficiently with numpy/scipy operations
- It requires integration with other Python packages (sklearn, scipy.signal, matplotlib, etc.)
- It is primarily a data format conversion or reshaping step
- The data size makes Python overhead negligible

---

**Stop here. Present the API surface concisely. Ask for corrections and explicit approval before proceeding.**

---

## Phase 2 — Tests

Once the design is approved, create a full set of tests needed to verify the feature:
1. Re-use existing fixtures where possible; add new ones to `tests/fixtures.py` if needed.
2. Re-use existing `valid_*` helpers where possible; add new ones to `tests/checks.py` if needed.
3. Write the test functions and put them in the most appropriate comment-denoted topic-group within an existing test file. Only add module-level test files if the feature is large and self-contained enough to warrant it.
4. Do not run tests that are guaranteed to fail before implementation. Run pre-implementation tests only when needed to verify assumptions or clarify unexpected behavior.

---

**Stop here. Let the user review test changes. Continue on approval, or refine when requested. Before continuing, update the API plan to incorporate manual changes made to the test files where possible.**

--- 

## Phase 3 — Implementation

Implement the feature to make the tests pass:

### If adding C++ code

1. Define new structs (owning class / WriteView / View / Capsule) in the appropriate `.h` file following the memory ownership pattern.
2. Implement the algorithm in the `.cpp` file:
   - Release the GIL at the top of every compute function: `nb::gil_scoped_release guard{};`
   - Parallelise with `#pragma omp parallel for default(none) shared(...)` where applicable
3. Add bindings in the matching `add_*_bindings` section of `bindings.cpp`:
   - Use `numpy.asarray()` in `__init__` for array arguments to support sklearn pickling
   - Expose read-only fields with `nb::rv_policy::reference`
   - Implement `__iter__` and `__reduce__` if the type is a fitted attribute
4. Expose the new symbol through the Python layer (import in `api.py` or `sklearn.py` as appropriate)

### If adding Python code

1. Add internal helpers to `_helpers.py` if reusable across modules.
2. Add public functional API to `api.py` and re-export from `__init__.py` / `__all__` if applicable.
3. Add parameters to `PLSCAN` with `_parameter_constraints` entries; call `self._validate_params()` first in `fit()`.
4. Ensure all C++ inputs satisfy the preconditions documented in the Python API instructions.

## Phase 4 — Documentation

Document every new public feature (parameters, methods, attributes, or functional API):
1. Update or add relevant docs page(s) in `docs/`.
2. For each example, explain:
   - what the feature does,
   - when and how to use it.
3. Refer to related HDBSCAN functionality or terminology if available.
4. Keep examples executable and consistent with current public API names.
5. If a notebook is not the best fit, add a clear code example in the most relevant `.rst` page.
6. Check docstrings are included in the autosummary path.

---

**Stop here. Let the user review documentation updates. Continue on approval, or refine when requested.**

### Final check

- Run `pytest .` and confirm all new and existing tests pass.
- Verify the public API changes are reflected in `__init__.py` and `__all__` if applicable.
- Verify docs/notebook references and examples are updated for each new feature.
- At completion, give a ready-to-copy commit command with a message summarising the complete feature addition (e.g. "Add weighted core distances feature with API, tests, and docs").
