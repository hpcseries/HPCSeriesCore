# Guide: Re-Releasing v0.7.0 After Updates

This guide walks you through deleting the previous v0.7.0 release and creating a new one with the latest commits.

---

## Step 1: Delete Previous Release and Tag

### Delete GitHub Release

1. Go to: **https://github.com/hpcseries/HPCSeriesCore/releases**
2. Find the **v0.7.0** release
3. Click **"Edit"** (pencil icon)
4. Scroll down and click **"Delete this release"**
5. Confirm deletion

### Delete Local and Remote Tag

```bash
# Delete local tag
git tag -d v0.7.0

# Delete remote tag
git push origin :refs/tags/v0.7.0
# Or alternatively:
git push --delete origin v0.7.0
```

**Verify deletion:**
```bash
git tag | grep v0.7.0  # Should return nothing
```

---

## Step 2: Ensure Latest Changes are Committed

Check what's ready to commit:

```bash
git status
```

You should see these new/modified files ready:
- `.github/workflows/publish-to-pypi.yml` (NEW)
- `.github/workflows/publish-to-test-pypi.yml` (NEW)
- `.github/TRUSTED_PUBLISHING_SETUP.md` (NEW)
- `.github/RE-RELEASE_GUIDE.md` (NEW - this file)
- `.github/RELEASE_TEMPLATE.md` (UPDATED)
- `.github/PYPI_ACTIVATION_GUIDE.md` (NEW)
- `.github/RELEASE_READINESS_REPORT.md` (NEW)
- `pyproject.toml` (UPDATED - fixed license deprecation)
- `MANIFEST.in` (UPDATED - includes native library)

### Commit Changes

```bash
# Add all release-related files
git add .github/ pyproject.toml MANIFEST.in

# Create commit
git commit -m "$(cat <<'EOF'
Add PyPI Trusted Publishing and finalize v0.7.0 release

- Add GitHub Actions workflows for PyPI publication with OIDC
- Fix pyproject.toml license deprecation warnings
- Update MANIFEST.in to include pre-built native library
- Add comprehensive Trusted Publishing setup guide
- Update release template with PyPI installation instructions
- Add release readiness report and guides

Changes enable secure, automated PyPI publication on release.
EOF
)"

# Push to GitHub
git push origin main
```

---

## Step 3: Configure PyPI Trusted Publisher (One-Time)

**IMPORTANT:** Do this before creating the release!

### For Production PyPI

1. Go to: **https://pypi.org/manage/account/publishing/**
2. Click **"Add a new pending publisher"**
3. Fill in:
   ```
   PyPI Project Name: hpcs
   Owner: hpcseries
   Repository name: HPCSeriesCore
   Workflow name: publish-to-pypi.yml
   Environment name: (leave empty)
   ```
4. Click **"Add"**

### For TestPyPI (Optional - for testing first)

1. Go to: **https://test.pypi.org/manage/account/publishing/**
2. Click **"Add a new pending publisher"**
3. Fill in same details but use workflow: `publish-to-test-pypi.yml`
4. Click **"Add"**

---

## Step 4: Create New v0.7.0 Tag

```bash
# Create annotated tag with comprehensive message
git tag -a v0.7.0 -m "$(cat <<'EOF'
HPCSeries Core v0.7.0 - Architecture-Aware Compilation & PyPI Publication

Major features:
- Architecture-aware compilation (x86 Intel/AMD, ARM Graviton)
- PyPI package publication with Trusted Publishing
- AWS deployment infrastructure
- Comprehensive performance validation
- Thread scaling optimization (OMP_NUM_THREADS=2 universal)

Full release notes: https://github.com/hpcseries/HPCSeriesCore/releases/tag/v0.7.0
EOF
)"

# Push tag to trigger workflows
git push origin v0.7.0
```

**Verify tag:**
```bash
git show v0.7.0
```

---

## Step 5: Create GitHub Release

### Option A: Via GitHub Web UI (Recommended)

1. Go to: **https://github.com/hpcseries/HPCSeriesCore/releases/new**
2. **Choose a tag:** Select `v0.7.0` from dropdown
3. **Release title:** `HPCSeries Core v0.7.0`
4. **Description:** Copy content from `.github/RELEASE_TEMPLATE.md`
5. **Optional:** Check "Set as the latest release"
6. Click **"Publish release"**

### Option B: Via GitHub CLI

```bash
gh release create v0.7.0 \
  --title "HPCSeries Core v0.7.0" \
  --notes-file .github/RELEASE_TEMPLATE.md \
  --latest
```

---

## Step 6: Monitor GitHub Actions Publication

After creating the release, GitHub Actions will automatically:

1. **Build native library** - Compiles `libhpcs_core.a` with CMake
2. **Build Python packages** - Creates wheel and source distribution
3. **Verify packages** - Runs `twine check`
4. **Publish to PyPI** - Uploads using Trusted Publishing

**Watch progress:**
1. Go to: **https://github.com/hpcseries/HPCSeriesCore/actions**
2. Find the **"Publish to PyPI"** workflow run
3. Click to see live logs

**Expected duration:** 3-5 minutes

---

## Step 7: Verify Publication

### Check PyPI Package Page

**URL:** https://pypi.org/project/hpcs/0.7.0/

Should show:
- ✅ Version 0.7.0
- ✅ Upload date: 2025-12-17
- ✅ Description from README.md
- ✅ Project links (GitHub, Docs, Issues)
- ✅ Download files (wheel + sdist)

### Test Installation

```bash
# Create test environment
python3 -m venv test-install
source test-install/bin/activate

# Install from PyPI
pip install hpcs==0.7.0

# Verify installation
python3 -c "import hpcs; print(hpcs.__version__)"
# Should print: 0.7.0

# Test basic functionality
python3 -c "
import hpcs
import numpy as np
x = np.random.randn(1000000)
print('Mean:', hpcs.mean(x))
print('Std:', hpcs.std(x))
print('SIMD Info:', hpcs.simd_info())
"

# Cleanup
deactivate
rm -rf test-install
```

---

## Troubleshooting

### Issue: "Tag v0.7.0 already exists"

**Solution:** You didn't delete the old tag. Run:
```bash
git tag -d v0.7.0
git push --delete origin v0.7.0
```

### Issue: GitHub Actions fails with "403 Forbidden"

**Solution:** Trusted publisher not configured. See Step 3.

### Issue: "400 Bad Request: File already exists"

**Cause:** Version 0.7.0 already on PyPI (cannot be replaced)

**Solutions:**
- **If you own the package:** Delete old version from PyPI web UI first
- **If testing:** Use TestPyPI or bump to v0.7.1

### Issue: Workflow doesn't start

**Solution:** Check that:
1. Tag was pushed: `git ls-remote --tags origin | grep v0.7.0`
2. Release was published (not saved as draft)
3. Workflow file is on main branch

---

## Quick Command Summary

```bash
# Delete old release (manual: via GitHub web UI)
# Delete old tags
git tag -d v0.7.0
git push --delete origin v0.7.0

# Commit latest changes
git add .github/ pyproject.toml MANIFEST.in
git commit -m "Add PyPI Trusted Publishing and finalize v0.7.0 release"
git push origin main

# Configure PyPI trusted publisher (manual: via PyPI web UI)

# Create new tag
git tag -a v0.7.0 -m "HPCSeries Core v0.7.0"
git push origin v0.7.0

# Create release (manual: via GitHub web UI)
# Monitor workflow (manual: GitHub Actions page)

# Test installation
pip install hpcs==0.7.0
python3 -c "import hpcs; print(hpcs.__version__)"
```

---

## Post-Release Checklist

- [ ] GitHub release created and published
- [ ] GitHub Actions workflow completed successfully
- [ ] Package visible on PyPI: https://pypi.org/project/hpcs/0.7.0/
- [ ] Installation tested: `pip install hpcs==0.7.0`
- [ ] Version verified: `python3 -c "import hpcs; print(hpcs.__version__)"`
- [ ] Basic functionality tested
- [ ] Release notes accurate
- [ ] Documentation links working

---

**You're ready to re-release v0.7.0!** 🚀

Follow these steps in order, and the automated workflows will handle PyPI publication securely via Trusted Publishing.
