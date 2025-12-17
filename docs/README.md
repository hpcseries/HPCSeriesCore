# HPCSeries Core Documentation

This directory contains the Sphinx documentation for HPCSeries Core.

## Prerequisites

Install documentation dependencies:

```bash
pip install -e ".[docs]"
```

This installs:
- `sphinx>=5.0` - Documentation generator
- `sphinx-rtd-theme` - Read the Docs theme
- `nbsphinx` - Jupyter notebook integration

## Building Documentation

### HTML Documentation

```bash
cd docs
make html
```

View the documentation by opening `build/html/index.html` in your browser.

### PDF Documentation

```bash
cd docs
make latexpdf
```

Requires LaTeX installation. Output will be in `build/latex/HPCSeriesCore.pdf`.

### Other Formats

```bash
make help          # Show all available targets
make clean         # Clean build directory
make dirhtml       # HTML with one page per directory
make singlehtml    # Single HTML page
make epub          # EPUB format
```

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main documentation page
│   ├── installation.rst     # Installation guide
│   ├── quickstart.rst       # Quick start guide
│   ├── api/                 # API reference
│   │   └── index.rst
│   ├── notebooks/           # Jupyter notebook tutorials
│   │   └── index.rst
│   └── user_guide/          # User guides
│       └── index.rst
├── Makefile                 # Build commands
└── build/                   # Generated documentation (gitignored)
```

## Writing Documentation

- Write content in reStructuredText (.rst) files in `source/`
- Python docstrings are automatically extracted via `autodoc`
- Jupyter notebooks in `../notebooks/` are included via `nbsphinx`
- Use NumPy-style docstrings for consistency

## Live Preview

For live documentation preview during development:

```bash
pip install sphinx-autobuild
cd docs
sphinx-autobuild source build/html
```

Then open http://127.0.0.1:8000 in your browser.
