# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = 'dwave-gate'
copyright = '2022, D-Wave Systems Inc.'
author = 'D-Wave Systems Inc.'
release = '2022'

# -- General configuration ---------------------------------------------------

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

# -- Options for HTML output ----------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "collapse_navigation": True,
    "show_prev_next": False,
}
html_sidebars = {"**": ["search-field", "sidebar-nav-bs"]}  # remove ads