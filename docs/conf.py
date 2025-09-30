# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import builtins
from importlib.metadata import version as get_version

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))
setattr(builtins, "--BUILDING-DOCS--", True)

# -- Project information -----------------------------------------------------

project = "plscan"
copyright = "2025, Jelmer Bot"
author = "Jelmer Bot"

# -- General configuration ---------------------------------------------------

release = get_version("plscan")
version = ".".join(release.split(".")[:2])
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_*.ipynb"]
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "nbsphinx",
]

autoclass_content = "both"
autodoc_default_flags = ["members"]
napoleon_use_rtype = False
autosummary_generate = True
autosummary_ignore_module_all = False
typehints_defaults = "comma"
always_use_bars_union = True
nbsphinx_assume_equations = False
nbsphinx_execute = "never"
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
htmlhelp_basename = "plscan_doc"
