# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os
import subprocess

sys.path.insert(0, os.path.abspath('../..'))


def get_git_version():
    git = subprocess.Popen(
        ['git', 'describe', '--tags'], stdout=subprocess.PIPE)
    return str(git.stdout.readlines()[0])


project = 'PumaGuard'
copyright = '2024, Pajarito Environmental Education Center Nature Youth Group'
author = 'Pajarito Environmental Education Center Nature Youth Group'
version = get_git_version()
release = '2024'

# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-numfig
numfig = True

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinxcontrib.mermaid',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_new_tab_link',
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
