# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = 'dwave-gate'
copyright = '2026, D-Wave'
author = 'D-Wave'
release = '2026'

# -- General configuration ---------------------------------------------------

extensions = [
    #'reno.sphinxext',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # must be loaded before 'sphinx_autodoc_typehints'
    #'sphinx_autodoc_typehints',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'README.rst']

autodoc_member_order = 'bysource'
typehints_use_rtype = False  # avoids duplicate return types
napoleon_use_rtype = False
typehints_defaults = 'comma'

# -- Options for HTML output ----------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "collapse_navigation": True,
    "show_prev_next": False,
}
html_sidebars = {"**": ["search-field", "sidebar-nav-bs"]}  # remove ads

intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'pydantic': ('https://pydantic.dev/docs/validation/latest', None),
                       'dwave': ('https://docs.dwavequantum.com/en/latest/', None),
                       }

todo_include_todos = True       # Set to False when merging to dwave-gate