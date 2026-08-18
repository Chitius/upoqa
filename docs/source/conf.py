# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'UPOQA'
copyright = '2025, Yichuan Liu and Yingzhou Li'
author = 'Yichuan Liu and Yingzhou Li'
release = 'v1.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_static_path = ['_static']

autodoc_typehints = "description"
numpydoc_class_members_toctree = False

import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode', 'numpydoc',
    'sphinx.ext.mathjax', 
    'sphinx.ext.intersphinx',
]

numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = False

html_theme = 'sphinx_rtd_theme'

# autodoc_default_options = {
#     'exclude-members': ','.join([
#         'get_reorganized_inputs_or_exit_info',
#     ])
# }
