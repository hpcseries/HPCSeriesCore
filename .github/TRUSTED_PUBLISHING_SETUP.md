# PyPI Trusted Publishing Setup Guide

**Trusted Publishing** is PyPI's modern authentication system using OpenID Connect (OIDC). It's more secure than API tokens and is now the **recommended** publishing method.

## Why Use Trusted Publishing?

✅ **More Secure** - No long-lived API tokens in secrets
✅ **Automatic** - Publish on GitHub release or tag push
✅ **Zero Configuration** - No secrets to manage
✅ **Auditable** - Clear publication trail in GitHub Actions

---

## Setup Instructions

### Step 1: Configure PyPI Trusted Publisher

#### For Production PyPI

1. Go to: **https://pypi.org/manage/account/publishing/**
2. Click **"Add a new pending publisher"**
3. Fill in the form:
   ```
   PyPI Project Name: hpcs
   Owner: hpcseries
   Repository name: HPCSeriesCore
   Workflow name: publish-to-pypi.yml
   Environment name: (leave empty)
   ```
4. Click **"Add"**

#### For TestPyPI (Optional - for testing)

1. Go to: **https://test.pypi.org/manage/account/publishing/**
2. Click **"Add a new pending publisher"**
3. Fill in the form:
   ```
   PyPI Project Name: hpcs
   Owner: hpcseries
   Repository name: HPCSeriesCore
   Workflow name: publish-to-test-pypi.yml
   Environment name: (leave empty)
   ```
4. Click **"Add"**

---

## How to Publish

### Method 1: Create a GitHub Release (Production)

This is the **recommended** method for stable releases:

```bash
# 1. Tag the release
git tag -a v0.7.0 -m "HPCSeries Core v0.7.0"
git push origin v0.7.0

# 2. Go to GitHub and create a release
# https://github.com/hpcseries/HPCSeriesCore/releases/new
# - Select tag: v0.7.0
# - Release title: HPCSeries Core v0.7.0
# - Description: Use content from .github/RELEASE_TEMPLATE.md
# - Click "Publish release"

# 3. GitHub Actions will automatically:
#    - Build the native library
#    - Build Python wheels and sdist
#    - Publish to PyPI
```

**The workflow runs automatically on release publication!**

### Method 2: Manual Trigger (Production or Test)

1. Go to: **https://github.com/hpcseries/HPCSeriesCore/actions**
2. Select **"Publish to PyPI"** or **"Publish to TestPyPI"**
3. Click **"Run workflow"**
4. Click **"Run workflow"** again to confirm

### Method 3: Test with Tag (TestPyPI only)

```bash
# Create a test tag
git tag v0.7.0-test1
git push origin v0.7.0-test1

# Workflow automatically publishes to TestPyPI
```

---

## Workflow Files

Two GitHub Actions workflows are configured:

### 1. Production PyPI
**File:** `.github/workflows/publish-to-pypi.yml`

**Triggers:**
- When you create a GitHub release
- Manual trigger via GitHub Actions UI

**Publishes to:** https://pypi.org/project/hpcs/

### 2. TestPyPI (Optional)
**File:** `.github/workflows/publish-to-test-pypi.yml`

**Triggers:**
- When you push tags like `v0.7.0-test1`
- Manual trigger via GitHub Actions UI

**Publishes to:** https://test.pypi.org/project/hpcs/

---

## Security: How It Works

1. **GitHub Actions generates a unique OIDC token** when the workflow runs
2. **PyPI verifies the token** matches the configured publisher (repository, workflow, owner)
3. **PyPI grants temporary upload permissions** for that specific workflow run
4. **Token expires immediately** after the workflow completes

**No API tokens are stored anywhere!**

---

## Verifying Publication

After the workflow completes:

```bash
# Check PyPI package page
# https://pypi.org/project/hpcs/0.7.0/

# Test installation
pip install hpcs==0.7.0

# Verify version
python3 -c "import hpcs; print(hpcs.__version__)"
```

---

## Troubleshooting

### Error: "403 Forbidden: Invalid or non-existent authentication information"

**Cause:** Trusted publisher not configured on PyPI

**Fix:**
1. Go to https://pypi.org/manage/account/publishing/
2. Verify the publisher is added with correct details
3. Make sure the workflow name matches exactly: `publish-to-pypi.yml`

### Error: "400 Bad Request: File already exists"

**Cause:** Version 0.7.0 already published

**Fix:**
- **For TestPyPI:** Delete the old version or use a different version
- **For Production PyPI:** You cannot replace or delete. Bump version to 0.7.1

### Error: "The workflow must have id-token: write permissions"

**Cause:** Missing OIDC permission in workflow

**Fix:** Already configured in workflows - permissions are set correctly

### Workflow doesn't trigger on release

**Cause:** Release was created from a branch instead of a tag

**Fix:**
1. Create and push tag first: `git tag v0.7.0 && git push origin v0.7.0`
2. Then create release from that tag

---

## Comparison: Trusted Publishing vs API Tokens

| Feature | Trusted Publishing | API Tokens |
|---------|-------------------|------------|
| Security | ✅ Temporary credentials | ⚠️ Long-lived secrets |
| Setup | ✅ One-time PyPI config | ⚠️ Store token in GitHub Secrets |
| Rotation | ✅ Automatic | ⚠️ Manual |
| Compromise Risk | ✅ Low - expires immediately | ⚠️ High - permanent until revoked |
| Audit Trail | ✅ Clear GitHub Actions logs | ⚠️ Less visibility |
| PyPI Recommendation | ✅ **Recommended** | ⚠️ Legacy method |

---

## Quick Reference

### First Time Setup
1. Configure trusted publisher on PyPI (5 minutes)
2. Push tag or create release
3. Done! Workflow publishes automatically

### Every Release After
1. Create GitHub release
2. That's it! Automation handles the rest

---

## Additional Resources

- **PyPI Trusted Publishing Docs:** https://docs.pypi.org/trusted-publishers/
- **GitHub OIDC Docs:** https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect
- **pypa/gh-action-pypi-publish:** https://github.com/pypa/gh-action-pypi-publish

---

**Ready to publish v0.7.0 securely with Trusted Publishing!** 🚀
