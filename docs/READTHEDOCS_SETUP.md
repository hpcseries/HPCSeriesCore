# Read the Docs Setup Guide

This guide walks you through publishing HPCSeries Core documentation to Read the Docs.

## Prerequisites

- GitHub repository: `hpcseries/HPCSeriesCore`
- PyPI package: `hpcs` (already published at https://pypi.org/project/hpcs/)
- Read the Docs account (sign up at https://readthedocs.org/)

## Step 1: Import Project to Read the Docs

1. Go to https://readthedocs.org/dashboard/
2. Click **"Import a Project"**
3. Connect your GitHub account if not already connected
4. Find and select `hpcseries/HPCSeriesCore` from the list
5. Click **"Import"**

## Step 2: Configure Project Settings

After importing, configure the project:

### Basic Settings
- **Name**: `HPCSeries Core` (or `hpcs`)
- **Repository URL**: `https://github.com/hpcseries/HPCSeriesCore`
- **Default branch**: `main`
- **Default version**: `latest`

### Advanced Settings

Navigate to **Admin → Advanced Settings**:

- **Default version**: `latest`
- **Privacy level**: `Public`
- **Programming language**: `Python`
- **Documentation type**: `Sphinx`
- **Requirements file**: `docs/requirements.txt`
- **Python interpreter**: `CPython 3.12`
- **Install your project**: ✓ (checked)
- **Use system packages**: ☐ (unchecked)

Click **"Save"**

## Step 3: Trigger First Build

1. Go to **Builds** tab
2. Click **"Build version: latest"**
3. Monitor the build progress

The build should:
- Install dependencies from `docs/requirements.txt`
- Mock Cython extensions (since native library can't be built on RTD)
- Generate API documentation from docstrings
- Build HTML, PDF, and EPUB formats

## Step 4: Verify Documentation

Once the build completes:

1. Click **"View Docs"** to see your published documentation
2. Verify all sections are rendering correctly:
   - Installation guide
   - Quick start
   - API reference
   - User guide
   - Examples & notebooks

Your documentation will be available at:
**https://hpcseries-core.readthedocs.io/** (or similar, depending on project slug)

## Step 5: Configure Webhooks (Automatic)

Read the Docs automatically creates a webhook in your GitHub repository to trigger builds on every push.

Verify webhook setup:
1. Go to GitHub: `https://github.com/hpcseries/HPCSeriesCore/settings/hooks`
2. You should see a webhook pointing to `readthedocs.org/api/v2/webhook/...`
3. Webhook should be active with green checkmark

## Step 6: Set Up Version Management

Configure which versions to build:

1. Go to **Versions** tab in Read the Docs
2. **Activate** versions you want to build:
   - `latest` (tracks main branch)
   - `stable` (tracks latest release tag)
   - Specific version tags like `v0.7.0`, `v0.7.1`, etc.

### Version Configuration
- **latest**: Always builds from `main` branch (development docs)
- **stable**: Builds from latest Git tag (stable release docs)
- **v0.7.0**: Specific release version (archived docs)

## Step 7: Configure Redirects (Optional)

To redirect from old documentation URLs:

1. Go to **Admin → Redirects**
2. Add redirects as needed

## Step 8: Update PyPI Documentation URL

Once documentation is live, update PyPI to link to Read the Docs:

1. Verify `pyproject.toml` has correct documentation URL:
   ```toml
   [project.urls]
   Documentation = "https://hpcseries-core.readthedocs.io/"
   ```

2. Release a new version (0.7.1) or update metadata on PyPI

## Step 9: Add Documentation Badge to README

Add Read the Docs badge to `README.md`:

```markdown
[![Documentation Status](https://readthedocs.org/projects/hpcseries-core/badge/?version=latest)](https://hpcseries-core.readthedocs.io/en/latest/?badge=latest)
```

## Troubleshooting

### Build Fails with "Module not found"

**Issue**: Cython extensions can't be imported

**Solution**: The `.readthedocs.yml` and `conf.py` are configured to mock Cython extensions. Ensure:
- `READTHEDOCS=True` environment variable is set (automatic on RTD)
- Mock imports are configured in `docs/source/conf.py`

### Build Fails with "Missing dependencies"

**Issue**: Required packages not installed

**Solution**: Update `docs/requirements.txt` with missing dependencies

### Documentation looks wrong

**Issue**: Incorrect Sphinx configuration

**Solution**: Test build locally:
```bash
cd docs
pip install -r requirements.txt
READTHEDOCS=True sphinx-build -b html source build/html
```

### API documentation not generated

**Issue**: autodoc can't import modules

**Solution**: Verify mock imports in `conf.py` include all Cython extension modules

## Configuration Files

The following files configure Read the Docs:

### `.readthedocs.yml`
Main configuration file that specifies:
- Build OS and tools (Ubuntu 22.04, Python 3.12)
- Documentation format (Sphinx)
- Python dependencies
- Output formats (HTML, PDF, EPUB)

### `docs/requirements.txt`
Python dependencies needed to build documentation:
- Sphinx and extensions
- Theme (sphinx_rtd_theme)
- nbsphinx for Jupyter notebooks
- numpy, Cython for API documentation

### `docs/source/conf.py`
Sphinx configuration:
- Project metadata
- Extensions and their settings
- Theme configuration
- Mock imports for Cython extensions

## Continuous Documentation

Read the Docs automatically rebuilds documentation when:
- You push to `main` branch (updates `latest` version)
- You create a new Git tag (creates new version)
- You manually trigger a build from RTD dashboard

This ensures documentation always stays in sync with your code.

## Next Steps

After documentation is live:
1. Share the documentation URL with users
2. Add link to PyPI package page
3. Include documentation badge in README
4. Set up custom domain (optional): `docs.hpcseries.io`
5. Monitor build status and fix any warnings

## Support

- Read the Docs Documentation: https://docs.readthedocs.io/
- Sphinx Documentation: https://www.sphinx-doc.org/
- Sphinx RTD Theme: https://sphinx-rtd-theme.readthedocs.io/
