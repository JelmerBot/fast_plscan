---
description: "Use when writing, editing, or reviewing documentation, docstrings, or API reference content. Covers Sphinx structure in docs/, NumPy-style docstrings, autosummary pages, and tutorial notebooks."
applyTo:
  - "docs/**"
  - "src/fast_plscan/*.py"
  - "src/fast_plscan/_api/**"
---

# Documentation Guidelines

General cross-cutting conventions are defined in `../copilot-instructions.md`.

## Documentation Architecture

This project documents the library with three coordinated layers:

1. Source docstrings in Python and C++ bindings.
2. Sphinx pages in `docs/*.rst` for static content without code examples.
3. Jupyter notebooks in `docs/*.ipynb` for tutorials and demos demonstrating API usage.

Agents should keep these layers consistent when API behavior changes.

## Sphinx Project Structure (`docs/`)

- `docs/conf.py` is the source of truth for doc build behavior.
- The docs use `autodoc`, `autosummary`, `napoleon`, `sphinx_autodoc_typehints`, and `nbsphinx`.
- Notebook execution is disabled during docs builds (`nbsphinx_execute = "never"`), so notebooks must be committed in a renderable state and not depend on runtime execution during `sphinx-build`.
- `docs/index.rst` defines the top-level information architecture via toctrees.
- The API reference is generated via `autosummary` directives and need not be manually maintained. Only ensure that public features are included in the autosummary path.

When adding a new page or notebook, include it in the correct toctree in `docs/index.rst`.

## Docstring Conventions

Use NumPy-style docstrings for user-facing symbols:

- Short summary line first.
- `Parameters` and `Returns` sections for public callables.
- Clear shape/dtype constraints where relevant.
- Keep parameter names and semantics consistent with actual function signatures.
- When changing behavior, update the docstring in the same change.

Docstrings for C++-bound functions/classes are defined in nanobind bindings under `src/fast_plscan/_api/`.

## Notebook Documentation Pattern

Notebooks under `docs/` are first-class docs pages.

- Start with concise markdown context that explains intent.
- Use small, sequential code cells that demonstrate one idea at a time.
- Keep examples reproducible with stable local paths/data references under `docs/data/`.
- Restrict examples to the public API (`from fast_plscan import PLSCAN` and functional APIs) rather than internal modules.
- Prefer pedagogical progression: setup, fit/compute, inspect outputs, visualize.

Keep notebook text and code aligned with current API names, defaults, and return semantics.

## Change Coordination Rules

When changing API surface, defaults, parameter behavior, or output schema:

1. Update source docstrings.
2. Update examples in docs pages/notebooks in `docs/`.
3. Ensure `docs/index.rst` navigation still reflects available content.
4. Ensure autosummary inputs still point to the correct modules/symbols.

Do not leave documentation updates for a follow-up when behavior changes are introduced.

## Build and Validation

Use the Sphinx make wrapper in `docs/Makefile`:

- `make html` from `docs/`.
- When removing public features, also run `make clean` to clear out old autosummary pages.

Any doc-focused change should be validated by a local docs build when feasible.