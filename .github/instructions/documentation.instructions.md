---
description: "Use when writing, editing, or reviewing documentation, docstrings, or API reference content. Covers Sphinx structure in docs/, NumPy-style docstrings, autosummary pages, and tutorial notebooks."
applyTo:
  - "docs/**"
  - "src/fast_plscan/*.py"
  - "src/fast_plscan/_api/**"
---

# Documentation Guidelines

## Documentation Architecture

This project documents the library with three coordinated layers:

1. Source docstrings in Python and C++ bindings.
2. Sphinx pages in `docs/*.rst` for static narrative content.
3. Jupyter notebooks in `docs/*.ipynb` for executable tutorials and demos.

Agents should keep these layers consistent when API behavior changes.

## Sphinx Project Structure (`docs/`)

- `docs/conf.py` is the source of truth for doc build behavior.
- The docs use `autodoc`, `autosummary`, `napoleon`, `sphinx_autodoc_typehints`, and `nbsphinx`.
- Notebook execution is disabled during docs builds (`nbsphinx_execute = "never"`), so notebooks must be committed in a renderable state and not depend on runtime execution during `sphinx-build`.
- `docs/index.rst` defines the top-level information architecture via toctrees:
  - Features (usage/tutorial notebooks)
  - Demonstrations (experiment/demo notebooks)
  - API reference (autosummary output)
  - Development (static contributor docs)

When adding a new page or notebook, include it in the correct toctree in `docs/index.rst`.

## API Reference Pattern

- `docs/reference_plscan.rst` defines autosummary directives for API reference generation.
- Files in `docs/_autosummary/` are generated artifacts. Do not hand-edit generated files unless the workflow explicitly requires it.
- Autosummary only covers the python API, so changes to C++-docstrings do not have to be validated against generated pages, but should still be kept up to date.
- Prefer updating source docstrings and autosummary inputs (`reference_plscan.rst`, package exports) over patching generated pages.

## Docstring Conventions

Use NumPy-style docstrings for user-facing symbols:

- Short summary line first.
- `Parameters` and `Returns` sections for public callables.
- Clear shape/dtype constraints where relevant.
- Keep parameter names and semantics consistent with actual function signatures.
- When changing behavior, update the docstring in the same change.

Python API docstrings live primarily in `src/fast_plscan/api.py` and `src/fast_plscan/sklearn.py`.
Docstrings for C++-bound functions/classes are defined in nanobind bindings under `src/fast_plscan/_api/`.

## Notebook Documentation Pattern

Notebooks under `docs/` are first-class docs pages.

- Start with concise markdown context that explains intent.
- Use small, sequential code cells that demonstrate one idea at a time.
- Keep examples reproducible with stable local paths/data references under `docs/data/`.
- Restrict examples to the public API (`from fast_plscan import PLSCAN` and functional APIs) rather than internal modules.
- Prefer pedagogical progression: setup, fit/compute, inspect outputs, visualize.

Keep notebook text and code aligned with current API names, defaults, and return semantics.

## Static RST Page Pattern

- Use reStructuredText directives consistently (`.. toctree::`, `.. code-block::`, `.. figure::`).
- Place long-form setup/build instructions in static `.rst` pages (for example `docs/local_development.rst`).
- Keep section headings and terminology consistent with the rest of the docs.

## Change Coordination Rules

When changing API surface, defaults, parameter behavior, or output schema:

1. Update source docstrings.
2. Update relevant docs pages/notebooks in `docs/`.
3. Ensure `docs/index.rst` navigation still reflects available content.
4. Ensure autosummary inputs still point to the correct modules/symbols.

Do not leave documentation updates for a follow-up when behavior changes are introduced.

## Build and Validation

Use the Sphinx make wrapper in `docs/Makefile`:

- `make html` from `docs/` (or equivalent `sphinx-build -M html . _build`).
- Use `make help` to inspect available targets.

Any doc-focused change should be validated by a local docs build when feasible.