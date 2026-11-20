"""Sphinx configuration for DimSpin."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from dimspin import __version__ as release  # noqa: E402

project = "DimSpin"
author = "Jia-Xin Chen"
copyright = "2026, Jia-Xin Chen"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = False

templates_path = []
html_static_path = ["_static"]
exclude_patterns = ["_build"]
html_theme = "alabaster"

# MathJax 3 is bundled under docs/_static/mathjax (Apache-2.0), so the built
# documentation renders formulas without network access.
mathjax_path = "mathjax/es5/tex-mml-chtml.js"
