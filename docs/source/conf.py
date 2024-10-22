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
    'sphinx_new_tab_link',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinxcontrib.mermaid',
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

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True
