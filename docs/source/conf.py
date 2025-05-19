# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Traffic-Forecasting-System'
copyright = '2025, Stefano de Saraca'
author = 'Stefano de Saraca'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Core library for html generation from docstrings
    'sphinx.ext.autosummary'   # To create neat summary tables
    'sphinx.ext.viewcode',     # To add links to highlighted source code
    'sphinx.ext.intersphinx',  # To add link to other projects’ documentation
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
