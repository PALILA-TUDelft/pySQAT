# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'SQUAT4PY'
copyright = '2025, Gerard Mendoza Ferrandis'
author = 'Gerard Mendoza Ferrandis'
release = '0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': '__init__',
    'inherited-members': True,
    'show-inheritance': True,
}

# -- TODO configuration ------------------------------------------------------
todo_include_todos = True

# -- Add paths for modules ---------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../../SQAT4PY'))