# ✅ HPCSeries Core v0.7.0 — Release Checklist

## 🎯 Release Intent (v0.7.0)

**Goal:**
Publish the **architecture-aware, production-ready CPU release** of HPCSeries Core with:
- Cross-architecture optimization (x86 Intel/AMD, ARM Graviton)
- SIMD vectorization (AVX-512, AVX2, NEON)
- OpenMP parallelization with empirically validated thread configuration
- Python bindings with zero-copy NumPy integration
- AWS deployment infrastructure
- Publication-quality performance documentation

**This is the production foundation release.**

---

## 1️⃣ Version Consistency Verification

### Critical version files (must all show v0.7.0 and 2025-12-17)

- [ ] `CMakeLists.txt:4` - `VERSION 0.7.0`
- [ ] `pyproject.toml:12` - `version = "0.7.0"`
- [ ] `CHANGELOG.md:8` - `## [0.7.0] - 2025-12-17 (Current)`
- [ ] `CITATION.cff:5-6` - `version: 0.7.0` and `date-released: 2025-12-17`
- [ ] `README.md:5` - Badge shows `0.7.0`
- [ ] `README.md:147` - `v0.7.0 (Current - 2025-12-17)`
- [ ] `.github/RELEASE_TEMPLATE.md:3` - `**Release Date:** 2025-12-17`

### Quick verification

```bash
grep -r "0\.6\.0\|0\.5\.0\|0\.4\.0" CMakeLists.txt pyproject.toml README.md CITATION.cff CHANGELOG.md
# Should return no results
```

---

## 2️⃣ Repository URLs Verification

### All URLs must point to `hpcseries/HPCSeriesCore`

- [ ] `pyproject.toml:64-67` - Homepage, Repository, Changelog URLs
- [ ] `CITATION.cff:7-8` - url and repository-code
- [ ] `README.md` - All GitHub links
- [ ] `.github/RELEASE_TEMPLATE.md:76` - Clone URL

### Quick verification

```bash
grep -r "github.com/your-org\|github.com/nrf-samkelo" . --include="*.md" --include="*.toml" --include="*.cff"
# Should return no results
```

---

## 3️⃣ Documentation Links Validation

### Core documentation files (must exist and be current)

- [ ] `CHANGELOG.md` - Complete v0.7.0 entry
- [ ] `LICENSE` - MIT license with correct year
- [ ] `CITATION.cff` - Valid CFF 1.2.0 format
- [ ] `CONTRIBUTING.md` - Contribution guidelines
- [ ] `SECURITY.md` - Security policy
- [ ] `docs/PERFORMANCE.md` - Empirical benchmarks
- [ ] `docs/PERFORMANCE_SUMMARY.md` - One-page summary
- [ ] `docs/AWS_DEPLOYMENT_GUIDE.md` - Production deployment
- [ ] `docs/BUILD_AND_TEST.md` - Build instructions
- [ ] `docs/CALIBRATION_GUIDE.md` - Auto-tuning guide
- [ ] `docs/NUMA_AFFINITY_GUIDE.md` - Multi-socket optimization

### README link verification

```bash
# Check all README links point to existing files
python3 -c "
import re
with open('README.md') as f:
    for match in re.finditer(r'\[.*?\]\((.*?\.md)\)', f.read()):
        path = match.group(1)
        if not path.startswith('http'):
            print(f'Checking: {path}')
"
```

---

## 4️⃣ Build System Validation

### CMake configuration

- [ ] Architecture detection works: `cmake/DetectArchitecture.cmake` exists
- [ ] Compiler flags module works: `cmake/CompilerFlags.cmake` exists
- [ ] CMake options correct: `BUILD_TESTS`, `BUILD_BENCHMARKS` (not `HPCS_*`)
- [ ] Version extracted from `CMakeLists.txt` matches `0.7.0`

### Build test (clean environment)

```bash
# Test clean build
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=ON
cmake --build . -j$(nproc)
ctest --output-on-failure
cd ..
```

- [ ] Build completes without errors
- [ ] Tests pass
- [ ] Benchmarks compile

### Python bindings test

```bash
pip install -e .
python3 -c "import hpcs; print(hpcs.__version__)"
# Should print: 0.7.0
```

- [ ] Python package installs
- [ ] Version matches
- [ ] Basic import works

---

## 5️⃣ CI/CD Workflow Validation

### GitHub Actions CI

File: `.github/workflows/ci.yml`

- [ ] Uses correct CMake flags: `-DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF`
- [ ] Workflow runs on push and pull_request
- [ ] All dependencies listed (gfortran, cmake, ninja-build)
- [ ] Tests run with `ctest --test-dir build --output-on-failure`

### Test CI workflow

```bash
# Simulate CI locally
docker run --rm -v $(pwd):/workspace -w /workspace ubuntu:latest bash -c "
  apt-get update && apt-get install -y gfortran cmake ninja-build
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DBUILD_BENCHMARKS=OFF
  cmake --build build
  ctest --test-dir build --output-on-failure
"
```

- [ ] CI simulation passes

---

## 6️⃣ Performance Documentation Validation

### Thread scaling documentation

- [ ] `docs/PERFORMANCE.md` includes results for:
  - AMD EPYC Genoa (c7a.xlarge)
  - Intel Ice Lake (m6i.2xlarge, c6i.4xlarge)
  - ARM Graviton3 (c7g.xlarge)
- [ ] OMP_NUM_THREADS=2 recommendation documented with rationale
- [ ] Performance degradation with 4+ threads quantified (5-18%)

### AWS deployment documentation

- [ ] EC2 instance metadata collection documented
- [ ] IMDSv2 authentication explained
- [ ] Instance family detection covered
- [ ] Benchmark reproduction steps included

---

## 7️⃣ Architecture-Aware Compilation Validation

### Detection scripts

- [ ] `scripts/detect_architecture.sh` exists and is executable
- [ ] Detects x86 vs ARM correctly
- [ ] Returns: `x86_intel`, `x86_amd`, `arm_graviton3`, etc.

### Test architecture detection

```bash
./scripts/detect_architecture.sh
# Should return valid architecture string
```

- [ ] Script executes without errors
- [ ] Returns expected architecture

### Compilation flags

- [ ] SAFE profile (default): no `-ffast-math`
- [ ] FAST profile (optional): includes `-ffast-math`
- [ ] x86: uses `-march=native -mtune=native`
- [ ] ARM: uses `-mcpu=native -mtune=native`

---

## 8️⃣ Git Repository Hygiene

### Branch status

```bash
git status
# Should show: "nothing to commit, working tree clean"
```

- [ ] No uncommitted changes
- [ ] No untracked files (except .gitignore exclusions)
- [ ] On `main` branch

### Git history

```bash
git log --oneline -10
```

- [ ] Commits have meaningful messages
- [ ] No "WIP" or "fix" commits in recent history
- [ ] Co-authored tags present on relevant commits

### Remote status

```bash
git remote -v
# Should show: origin https://github.com/hpcseries/HPCSeriesCore.git
```

- [ ] Remote points to correct repository
- [ ] Ready to push

---

## 9️⃣ Python Package Metadata

### pyproject.toml validation

- [ ] Version: `0.7.0`
- [ ] URLs point to `hpcseries/HPCSeriesCore`
- [ ] Dependencies listed: `numpy>=1.20`
- [ ] Python versions: `3.8+`
- [ ] License: `MIT`
- [ ] Authors: `HPCSeries Core Team`

### setup.py validation

- [ ] References v0.7 in docstring
- [ ] Cython extensions defined correctly
- [ ] Library dependencies listed (hpcs_core, gfortran, stdc++)

---

## 🔟 License & Citation

### License file

- [ ] `LICENSE` exists
- [ ] Contains Apache License 2.0 text
- [ ] Copyright year: 2025
- [ ] Copyright holder: HPCSeries Core Contributors

### Citation file

- [ ] `CITATION.cff` validates against CFF 1.2.0 schema
- [ ] Contains all required fields
- [ ] Authors, version, date match other files

### Test citation

```bash
# GitHub will auto-generate citation - verify format
cat CITATION.cff
```

---

## 1️⃣1️⃣ Release Assets Preparation

### GitHub Release Template

- [ ] `.github/RELEASE_TEMPLATE.md` ready
- [ ] All sections filled out:
  - Major Features
  - Performance Results
  - Installation instructions
  - Documentation links
  - Citation
  - Known Issues (if any)

### Tag creation (DO NOT RUN UNTIL READY)

```bash
# Create annotated tag
git tag -a v0.7.0 -m "HPCSeries Core v0.7.0 - Architecture-Aware Compilation & Performance Validation"

# Verify tag
git show v0.7.0

# Push tag (ONLY WHEN RELEASE IS FINAL)
# git push origin v0.7.0
```

---

## 1️⃣2️⃣ Pre-Release Smoke Tests

### Quick functionality test

```bash
python3 << 'EOF'
import hpcs
import numpy as np

# Test basic operations
x = np.random.randn(1000000)
assert abs(hpcs.mean(x) - np.mean(x)) < 1e-10
assert abs(hpcs.sum(x) - np.sum(x)) < 1e-8

# Test rolling operations
rolling = hpcs.rolling_mean(x[:10000], window=50)
assert len(rolling) == 10000

# Test SIMD detection
info = hpcs.simd_info()
assert 'isa' in info

print("✓ All smoke tests passed")
EOF
```

- [ ] Smoke tests pass

---

## 1️⃣3️⃣ Final Checklist Summary

### Critical items (must be green)

- [ ] All version numbers consistent (0.7.0, 2025-12-17)
- [ ] All URLs point to `hpcseries/HPCSeriesCore`
- [ ] All documentation links valid
- [ ] Build system works (clean build test passed)
- [ ] CI workflow validated
- [ ] Performance documentation complete
- [ ] Architecture detection works
- [ ] Git repository clean
- [ ] Python package metadata correct
- [ ] License and citation files valid
- [ ] Release template ready
- [ ] Smoke tests passed

---

## 1️⃣4️⃣ Release Execution Steps

### When all checklist items are ✅:

1. **Create and push git tag**
   ```bash
   git tag -a v0.7.0 -m "HPCSeries Core v0.7.0 - Architecture-Aware Compilation & Performance Validation"
   git push origin v0.7.0
   ```

2. **Create GitHub Release**
   - Go to: https://github.com/hpcseries/HPCSeriesCore/releases/new
   - Tag: `v0.7.0`
   - Title: `HPCSeries Core v0.7.0 - Architecture-Aware Compilation & Performance Validation`
   - Description: Copy from `.github/RELEASE_TEMPLATE.md`
   - Check "Set as the latest release"
   - Click "Publish release"

3. **Verify release assets**
   - GitHub automatically generates source archives (`.tar.gz`, `.zip`)
   - Verify they download correctly
   - Verify citation button appears

4. **Post-release verification**
   - GitHub citation button works
   - Release shows up on main page
   - Download links work

---

## 1️⃣5️⃣ Post-Release Communication (Optional)

### Announcement channels

- [ ] Update repository description
- [ ] Post to GitHub Discussions (if enabled)
- [ ] Update any external references
- [ ] Close any milestone issues for v0.7.0

---

## 🔒 1️⃣6️⃣ Post-Release Discipline

### Once v0.7.0 is published:

- ❌ Do NOT modify v0.7.0 behaviour
- ❌ Do NOT change ABI/API without version bump
- ❌ Do NOT backport features to v0.7.0

### All new work goes to:

- `main` branch (towards v0.8.0)
- Or `dev/v0.8` branch

### Bug fixes for v0.7.0:

- Create patch releases: `v0.7.1`, `v0.7.2`, etc.
- Only critical bugs and security fixes
- No new features

---

## 🧭 What v0.7.0 Communicates

**To the world:**
- Production-ready, cross-architecture HPC library
- Serious about performance validation
- Professional software engineering practices
- Clear scope and empirical documentation
- Ready for production deployments

**Technical credibility:**
- Architecture-aware compilation
- Empirical performance validation across AMD, Intel, ARM
- Publication-quality documentation
- Clean ABI/API design
- Reproducible benchmarks

---

## 📝 Release Notes Preview

```markdown
## HPCSeries Core v0.7.0

Production-ready release with architecture-aware compilation and comprehensive performance validation.

### Major Features
- **Architecture-Aware Compilation**: Automatic optimization for x86 (Intel/AMD) and ARM (Graviton)
- **SIMD Vectorization**: AVX-512, AVX2, NEON with runtime dispatch
- **OpenMP Parallelization**: Empirically validated thread configuration (OMP_NUM_THREADS=2)
- **AWS Deployment**: Production infrastructure with EC2 metadata integration
- **Performance Validation**: Tested on AMD EPYC Genoa, Intel Ice Lake, ARM Graviton3

### Performance Results
- **2-5x faster** reductions vs NumPy
- **50-100x faster** rolling operations vs Pandas
- **Universal finding**: 2 threads saturate memory bandwidth across all architectures

### Documentation
- [Performance Methodology](docs/PERFORMANCE.md) - Full empirical analysis
- [AWS Deployment Guide](docs/AWS_DEPLOYMENT_GUIDE.md) - Production deployment
- 12 comprehensive Jupyter notebooks

### Installation
```bash
pip install hpcs
```

See [CHANGELOG.md](CHANGELOG.md) for complete details.
```

---

## ✅ Sign-off

**Release Engineer:** ___________________
**Date:** 2025-12-17
**Version:** v0.7.0
**Status:** [ ] READY FOR RELEASE

---

**This checklist ensures HPCSeries Core v0.7.0 meets professional HPC library standards.**
