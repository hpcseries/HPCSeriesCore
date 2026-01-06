# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the Python package to the path so Sphinx can import it
sys.path.insert(0, os.path.abspath('../../python'))

# Mock imports for Cython extension modules
# This prevents import errors on Read the Docs where native libraries can't be built
autodoc_mock_imports = [
    'hpcs._core',
    'hpcs._simd',
]

# Check if we're building on Read the Docs
on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    # On RTD, inject documentation stubs for v0.8.0 functions defined in Cython
    # This allows Sphinx autodoc to extract proper signatures and docstrings
    # even when the compiled extensions aren't available
    try:
        import hpcs._docstubs as _stubs
        import hpcs

        # Monkey-patch hpcs module with stub functions for documentation
        for name in dir(_stubs):
            if not name.startswith('_'):
                setattr(hpcs, name, getattr(_stubs, name))
    except ImportError as e:
        print(f"Warning: Could not import documentation stubs: {e}")
        pass  # Continue anyway, autodoc_mock_imports will handle it

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HPCSeries Core'
copyright = '2025, HPCSeries Core Team'
author = 'HPCSeries Core Team'
version = '0.8.0'
release = '0.8.0'

# Project URLs
html_context = {
    'display_github': True,
    'github_user': 'hpcseries',
    'github_repo': 'HPCSeriesCore',
    'github_version': 'main',
    'conf_py_path': '/docs/source/',
}

# Custom sidebar
html_sidebars = {
    '**': [
        'globaltoc.html',
        'relations.html',
        'sourcelink.html',
        'searchbox.html',
    ]
}

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',           # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',          # Support for NumPy/Google style docstrings
    'sphinx.ext.viewcode',          # Add links to highlighted source code
    'sphinx.ext.intersphinx',       # Link to other project's documentation
    'sphinx.ext.mathjax',           # Render math equations
    'nbsphinx',                     # Jupyter notebook integration
    'sphinx.ext.autosummary',       # Generate summary tables
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings (for NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping (links to NumPy/Pandas docs)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# nbsphinx settings
nbsphinx_execute = 'never'  # Don't execute notebooks during build (they're pre-executed)
nbsphinx_allow_errors = False
nbsphinx_kernel_name = 'python3'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = '_static/logo.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = '_static/favicon.ico'

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto, manual, or own class]).
latex_documents = [
    ('index', 'HPCSeriesCore.tex', 'HPCSeries Core Documentation',
     'HPCSeries Core Team', 'manual'),
]
