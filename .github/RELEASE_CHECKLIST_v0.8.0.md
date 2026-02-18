# HPCSeries Core v0.8.0 — Release Checklist

## Release Intent (v0.8.0)

**Goal:** Publish the **execution-mode, pipeline, and workspace API release** of HPCSeries Core with:
- Three-mode execution system (SAFE / FAST / DETERMINISTIC)
- Pipeline API with 24 composable stages
- Workspace API for zero-allocation hot paths
- Exponential weighted statistics (EWMA, EWVAR, EWSTD)
- Advanced robust statistics (trimmed mean, winsorized mean)
- FIR convolution with template specialisations

**Release date:** 2026-02-18

**Distribution scope for v0.8.0:**
- GitHub Release (source + git tag) — YES
- PyPI (pip install hpcs) — DEFERRED (no PyPI credentials available)
- Pyodide / WebAssembly browser wheel — DEFERRED (see future work)

---

## 1 — Version Consistency

All files must show `0.8.0` and `2026-02-18`.

- [ ] `CMakeLists.txt` — `VERSION 0.8.0`
- [ ] `pyproject.toml` — `version = "0.8.0"`
- [ ] `python/hpcs/__init__.py` — `__version__ = "0.8.0"`
- [ ] `CHANGELOG.md` — `## [0.8.0] - 2026-02-18 (Current)`
- [ ] `CITATION.cff` — `version: 0.8.0` and `date-released: 2026-02-18`
- [ ] `README.md` — badge and current version reference show `0.8.0`
- [ ] `.github/RELEASE_TEMPLATE_v0.8.0.md` — `**Release Date:** 2026-02-18`

### Quick check

```bash
grep -r "0\.7\.0\|0\.6\.0\|2025-01-06\|2026-01-14" \
  CMakeLists.txt pyproject.toml CHANGELOG.md CITATION.cff
# Should return no results
```

---

## 2 — CHANGELOG Completeness

- [ ] Date is `2026-02-18` (not `2025-01-06`)
- [ ] Contains **Execution Mode System** section
- [ ] Contains **Pipeline API** section (24 stages listed)
- [ ] Contains **Workspace API** section
- [ ] Contains **Exponential Weighted Statistics** section
- [ ] Contains **FIR Filters** section
- [ ] Contains **Advanced Robust Statistics** section
- [ ] Testing section references `test_transforms_v08.py` and `test_execution_modes_v08.py`

---

## 3 — Ulwazi Files NOT Committed

The uLwazi lab notebook files must remain untracked. Confirm they are not staged or committed.

```bash
git status --short | grep "notebooks/"
# Must NOT show any notebooks/ files as staged (A) or committed
```

- [ ] `notebooks/ulwazi_saws_climate_analysis_2000_2024.ipynb` — untracked only
- [ ] `notebooks/data/saws_daily_temp_2000_2024.csv` — untracked only
- [ ] `notebooks/data/generate_saws_data.py` — untracked only

---

## 4 — All v0.8.0 Code Files Committed

- [ ] `include/hpcs_core.h` — 12 new `pipeline_add_*` declarations + `out_n` in `pipeline_execute`
- [ ] `python/hpcs/_core.pyx` — Cython bindings for 12 new stages + updated `execute()` signature
- [ ] `src/c/hpcs_plan.c` — Implementation of 12 new pipeline stages
- [ ] `tests/python/test_transforms_v08.py` — Updated tests for new stages

```bash
git log --oneline -5
# Should show the pipeline completion commit near the top
```

---

## 5 — Docker Build

```bash
docker compose -f docker-compose.python.yml up --build hpcs-jupyter
```

- [ ] Build completes without errors
- [ ] Jupyter accessible at `http://localhost:8888`
- [ ] `import hpcs` works in a notebook cell

---

## 6 — Python Smoke Tests

```python
import hpcs
import numpy as np

# Version
assert hpcs.__version__ == '0.8.0', f"Got {hpcs.__version__}"
print(f"Version OK: {hpcs.__version__}")

# Execution modes
hpcs.set_execution_mode(hpcs.MODE_SAFE)
assert hpcs.get_execution_mode() == hpcs.MODE_SAFE
hpcs.set_execution_mode(hpcs.MODE_FAST)
assert hpcs.get_execution_mode() == hpcs.MODE_FAST
hpcs.set_execution_mode(hpcs.MODE_SAFE)
print("Execution modes OK")

# EWMA / EWVAR / EWSTD
x = np.random.randn(10000).astype(float)
e = hpcs.ewma(x, alpha=0.1)
assert len(e) == len(x) and e[0] == x[0]
v = hpcs.ewvar(x, alpha=0.1)
assert v[0] == 0.0, f"EWVAR v[0] should be 0, got {v[0]}"
print("EWMA/EWVAR/EWSTD OK")
```

- [ ] Version assertion passes
- [ ] Execution mode round-trip passes
- [ ] EWMA/EWVAR/EWSTD assertions pass

---

## 7 — Pipeline API Smoke Test

```python
import hpcs, numpy as np

x = np.random.randn(1000).astype(float)

# Basic pipeline
p = hpcs.pipeline()
p.ewma(0.1).zscore()
out = p.execute(x)
assert len(out) == len(x), f"Expected {len(x)}, got {len(out)}"
print("Basic pipeline OK, shape:", out.shape)

# New stage: log_return
p2 = hpcs.pipeline()
p2.log_return()
out2 = p2.execute(np.abs(x) + 1.0)
print("log_return OK, shape:", out2.shape)

# New stage: convolve (output shortens)
kernel = np.array([1.0, 2.0, 1.0]) / 4.0
p3 = hpcs.pipeline()
p3.convolve(kernel)
out3 = p3.execute(x)
assert len(out3) < len(x), "convolve should shorten output"
print("convolve OK, input:", len(x), "output:", len(out3))
```

- [ ] Basic pipeline executes
- [ ] `log_return()` stage works
- [ ] `convolve()` stage shortens output correctly

---

## 8 — Workspace API Smoke Test

```python
import hpcs, numpy as np

ws = hpcs.workspace()
assert ws.size > 0, "Workspace size should be positive"
print(f"Workspace size: {ws.size:,} bytes")

x = np.random.randn(1000).astype(float)
p = hpcs.pipeline(ws=ws)
p.ewma(0.1).rolling_mean(10)
out = p.execute(x)
assert len(out) == len(x)
print("Workspace pipeline OK")

# reserve
ws.reserve(128 * 1024 * 1024)  # 128 MB
assert ws.size >= 128 * 1024 * 1024
print("Workspace reserve OK")
```

- [ ] `workspace()` creates with positive size
- [ ] Pipeline with workspace executes correctly
- [ ] `.reserve()` expands allocation

---

## 9 — Git Repository Status

```bash
git status
# Should show: working tree clean (except 3 untracked ulwazi files)

git log --oneline -5
# Should show meaningful commit messages
```

- [ ] On `feature/execution-modes` branch (pre-merge) or `main` (post-merge)
- [ ] No unexpected staged/modified files
- [ ] Only the 3 ulwazi files are untracked (not staged)
- [ ] Recent commits have meaningful messages

---

## 10 — Repository URLs

All URLs must point to `hpcseries/HPCSeriesCore`.

```bash
grep -r "github.com/your-org\|github.com/nrf-samkelo" \
  . --include="*.md" --include="*.toml" --include="*.cff"
# Should return no results
```

- [ ] `CITATION.cff` — url and repository-code correct
- [ ] `RELEASE_TEMPLATE_v0.8.0.md` — clone URL correct
- [ ] `pyproject.toml` — homepage, repository, changelog URLs correct

---

## 11 — Release Template Ready

- [ ] `.github/RELEASE_TEMPLATE_v0.8.0.md` exists
- [ ] All sections filled: Major Features, Breaking Changes, Performance, Installation, Verification, Citation
- [ ] Breaking change for C API `pipeline_execute()` documented
- [ ] BibTeX citation has version `0.8.0` and year `2026`

---

## 12 — Pre-Release Smoke Test (End-to-End)

Run the full verification block from `RELEASE_TEMPLATE_v0.8.0.md`:

```bash
python3 -c "
import hpcs, numpy as np
assert hpcs.__version__ == '0.8.0'
x = np.random.randn(10000).astype(float)
p = hpcs.pipeline()
p.ewma(0.1).zscore()
out = p.execute(x)
print('Pipeline OK, output shape:', out.shape)
ws = hpcs.workspace()
print('Workspace size (bytes):', ws.size)
p2 = hpcs.pipeline(ws=ws)
p2.ewma(0.1)
out2 = p2.execute(x)
print('Workspace pipeline OK')
print('All smoke tests PASSED')
"
```

- [ ] All assertions pass
- [ ] No import errors or HPCSeries init noise printed

---

## 13 — Final Summary

### Critical items (must all be green before tagging)

- [ ] All version numbers consistent (`0.8.0`, `2026-02-18`)
- [ ] CHANGELOG complete (all API sections present, correct date)
- [ ] Ulwazi files NOT committed
- [ ] All 4 code files committed
- [ ] Docker build passes
- [ ] Python version assertion passes
- [ ] Pipeline API smoke test passes
- [ ] Workspace API smoke test passes
- [ ] Git working tree clean (only ulwazi untracked)
- [ ] Repository URLs correct
- [ ] Release template ready
- [ ] End-to-end smoke test passes

---

## 14 — Release Execution Steps

When all checklist items are green:

### 1. Merge to main

```bash
git checkout main
git merge --no-ff feature/execution-modes \
  -m "Merge v0.8.0: Execution Modes, Pipeline & Workspace APIs"
git push origin main
```

### 2. Tag v0.8.0

```bash
git tag -a v0.8.0 \
  -m "HPCSeries Core v0.8.0 — Execution Modes, Pipeline & Workspace APIs"
# Verify tag
git show v0.8.0 --stat
# Push tag (ONLY WHEN RELEASE IS FINAL)
git push origin v0.8.0
```

### 3. Create GitHub Release

- Go to: https://github.com/hpcseries/HPCSeriesCore/releases/new
- Tag: `v0.8.0`
- Title: `HPCSeries Core v0.8.0 — Execution Modes, Pipeline & Workspace APIs`
- Description: copy from `.github/RELEASE_TEMPLATE_v0.8.0.md`
- Check "Set as the latest release"
- Click "Publish release"

### 4. Post-release verification

- [ ] GitHub citation button works
- [ ] Release appears on main page
- [ ] Download links work
- [ ] `CITATION.cff` auto-citation renders correctly

---

## 15 — Post-Release Discipline

Once v0.8.0 is published:

- Do NOT modify v0.8.0 behaviour
- Do NOT change ABI/API without a version bump
- All new work goes to `main` or a `dev/v0.9` branch
- Bug fixes only as patch releases (`v0.8.1`, `v0.8.2`)

---

## 16 — Deferred: PyPI & Browser Support

### PyPI publishing (deferred — credentials unavailable)

When PyPI access is restored:
```bash
python -m build
twine upload dist/hpcs-0.8.0*
```
Users will then be able to `pip install hpcs` directly.

### Browser / Pyodide support (v0.9.x work item)

HPCSeries uses compiled C/Fortran/C++ with SIMD intrinsics and OpenMP — none of which run natively in WebAssembly. Two paths exist:

**Path A — Not viable:** Pure Python `py3-none-any` wheel would remove all C/Fortran code and lose the performance advantage entirely.

**Path B — Pyodide WASM wheel (`*-pyodide_*_wasm32.whl`):**
- Compile C/C++/Fortran to WASM32 using emscripten toolchain
- Drop or stub out OpenMP (WASM is single-threaded)
- Replace AVX-512/AVX2/NEON intrinsics with WebAssembly SIMD128
- Build inside Pyodide's Docker build environment
- Estimated effort: 2–4 weeks engineering

**Short-term workaround for browser notebooks (e.g. uLwazi):**
- Run notebooks locally in Docker (`hpcs-jupyter` on port 8888) where the native wheel works
- Or create a `hpcs-lite` pure Python stub with numpy-backed implementations for demo/teaching use cases where correctness matters more than speed

---

## Sign-off

**Release Engineer:** ___________________
**Date:** 2026-02-18
**Version:** v0.8.0
**Status:** [ ] READY FOR RELEASE

---

**This checklist ensures HPCSeries Core v0.8.0 meets professional HPC library standards.**
