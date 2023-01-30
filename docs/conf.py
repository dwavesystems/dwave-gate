# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dwave-gate'
copyright = '2022, D-Wave Systems Inc.'
author = 'D-Wave Systems Inc.'
release = '2022'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'reno.sphinxext',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # must be loaded before 'sphinx_autodoc_typehints'
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_member_order = 'bysource'
typehints_use_rtype = False  # avoids duplicate return types
napoleon_use_rtype = False
typehints_defaults = 'comma'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
