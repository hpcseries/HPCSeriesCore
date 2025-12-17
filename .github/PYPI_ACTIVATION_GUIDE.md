# Virtual Environment Activation Guide

## Quick Start

### Linux/WSL/macOS
```bash
# Activate the virtual environment
source test-env/bin/activate

# Verify activation (you should see (test-env) in your prompt)
which python3

# Install build tools if needed
pip3 install --upgrade build twine

# Now proceed with PyPI publication
./.github/PYPI_QUICK_START.sh test
```

### Windows (PowerShell)
```powershell
# Activate the virtual environment
.\test-env\Scripts\Activate.ps1

# Install build tools if needed
pip3 install --upgrade build twine

# Now proceed with PyPI publication
.\.github\PYPI_QUICK_START.sh test
```

### Windows (Command Prompt)
```cmd
# Activate the virtual environment
test-env\Scripts\activate.bat

# Install build tools if needed
pip3 install --upgrade build twine

# Now proceed with PyPI publication
.github\PYPI_QUICK_START.sh test
```

---

## Full Publication Workflow

### Step-by-Step

```bash
# 1. Activate virtual environment
source test-env/bin/activate

# 2. Verify you're in the correct environment
python3 --version
which python3  # Should show: .../test-env/bin/python3

# 3. Install build dependencies
pip3 install --upgrade build twine

# 4. Verify native library exists
ls -lh build/libhpcs_core.a

# 5. Test on TestPyPI (recommended first)
./.github/PYPI_QUICK_START.sh test

# 6. Verify test installation
pip install --index-url https://test.pypi.org/simple/ --no-deps hpcs==0.7.0
python3 -c "import hpcs; print(hpcs.__version__)"

# 7. If test succeeds, publish to production PyPI
./.github/PYPI_QUICK_START.sh prod

# 8. Deactivate when done
deactivate
```

---

## Alternative: One-Line Activation + Publication

```bash
# Test on TestPyPI
source test-env/bin/activate && ./.github/PYPI_QUICK_START.sh test

# Publish to production PyPI
source test-env/bin/activate && ./.github/PYPI_QUICK_START.sh prod
```

---

## Troubleshooting

### Issue: "command not found: activate"
**Solution:**
```bash
# Make sure you're in the project root
cd /mnt/c/Users/Samkelo/OneDrive\ -\ iqlab/Clients/HPCSeriesCore

# Use source, not ./
source test-env/bin/activate
```

### Issue: "Permission denied"
**Solution:**
```bash
# Make activation script executable
chmod +x test-env/bin/activate
source test-env/bin/activate
```

### Issue: Virtual environment not found
**Solution:**
```bash
# Recreate virtual environment
python3 -m venv test-env
source test-env/bin/activate
pip3 install --upgrade pip setuptools wheel
pip3 install -r requirements.txt  # If you have one
```

---

## Verifying Active Environment

When the virtual environment is active, you should see:

```bash
# 1. Prompt shows (test-env)
(test-env) user@host:~/HPCSeriesCore$

# 2. Python path points to virtual environment
$ which python3
/mnt/c/Users/Samkelo/OneDrive - iqlab/Clients/HPCSeriesCore/test-env/bin/python3

# 3. pip installs to virtual environment
$ which pip3
/mnt/c/Users/Samkelo/OneDrive - iqlab/Clients/HPCSeriesCore/test-env/bin/pip3
```

---

## Why Activate Virtual Environment?

1. **Isolated Dependencies**: Build tools won't interfere with system Python
2. **Reproducible Builds**: Same environment every time
3. **Clean Testing**: Can test package installation in isolated environment
4. **No sudo Required**: Install packages without root permissions

---

## Deactivating

```bash
# When done, deactivate the virtual environment
deactivate
```

Your prompt will return to normal (no `(test-env)` prefix).
