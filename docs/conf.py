import sys
from pathlib import Path

# Make the src-layout package importable when building from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TranscriptML'
copyright = '2026, Isaac Vock'
author = 'Isaac Vock'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

# Code highlighting
pygments_style = "friendly"
highlight_language = "python"

# Useful autodoc behavior
autosummary_generate = True
autodoc_mock_imports = ["torch", "typing_extensions"]
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
