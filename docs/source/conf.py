# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# -- Add directories to python PATH ------------------------------------------
CURRENT_DIR = os.path.dirname(__file__)
# add root directory to path
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT_DIR)

# add tut6 package directory to path
TUT6_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../QUAT4PY"))
sys.path.insert(0, TUT6_DIR)

project = 'SQAT4PY'
copyright = '2025, Gerard Mendoza Ferrandis'
author = 'Gerard Mendoza Ferrandis'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
 'sphinx.ext.napoleon',  # for parsing Google style docstrings https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
 'sphinx.ext.autodoc',  # for automatically generating code documentation for all packages/modules
 'sphinx_rtd_theme',
 'sphinx.ext.autosectionlabel'  # for cross-referencing using headings
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

html_logo = "images/logo.png"
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}