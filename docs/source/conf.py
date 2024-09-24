# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

sys.path.insert(0, os.path.abspath('../..'))

print(sys.path)

project = 'PumaGuard'
copyright = '2024, Pajarito Environmental Education Center Youth Group'
author = 'Pajarito Environmental Education Center Youth Group'
version = '0.1.1'
release = '2024'

# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-numfig
numfig = True

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_new_tab_link',
    'sphinxcontrib.mermaid',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
]

mermaid_version = "11.2.1"

napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_ivar = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'cloud'
html_static_path = ['_static']
