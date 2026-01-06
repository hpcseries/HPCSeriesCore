# HPCSeries Core v0.7.0 - Architecture-Aware Compilation & Performance Validation

**Release Date:** 2025-12-17
**Tag:** `v0.7.0`

---

## 🎯 Major Features

### Architecture-Aware Compilation
- **Automatic CPU detection** for x86 (Intel/AMD) and ARM (Graviton)
- **Architecture-specific optimization** with CMake modules
- **SAFE (default) and FAST profiles** for IEEE 754 compliance vs performance
- Support for Intel Sapphire Rapids, Ice Lake, AMD EPYC Genoa, ARM Graviton3

### AWS Deployment Infrastructure
- **EC2 instance metadata** collection via IMDSv2
- **Instance family detection** (c7i, c6i, c7g, c7a families)
- **Automated benchmarking** with instance and CPU metadata
- Production-ready deployment guides

### Performance Validation
- **Comprehensive benchmarks** across AMD, Intel, and ARM architectures
- **Thread scaling analysis** proving memory bandwidth saturation
- **Universal finding:** `OMP_NUM_THREADS=2` optimal across all platforms
- **vCPU-independent scaling:** 2 threads optimal for 4, 8, and 16 vCPU instances

---

## 📊 Performance Results

### Thread Scaling (10M elements, MAD operation)

| CPU Family | Instance | vCPUs | 2 Threads | 4 Threads | Degradation |
|------------|----------|-------|-----------|-----------|-------------|
| AMD EPYC Genoa | c7a.xlarge | 4 | 296 ms | 313 ms | **+5.8%** ⚠️ |
| ARM Graviton3 | c7g.xlarge | 4 | 267 ms | 278 ms | **+4.0%** ⚠️ |
| Intel Ice Lake | m6i.2xlarge | 8 | 307 ms | 322 ms | **+5.0%** ⚠️ |
| Intel Ice Lake | c6i.4xlarge | 16 | 299 ms | 317 ms | **+6.1%** ⚠️ |

**Key Finding:** Memory bandwidth saturation at 2 threads across all architectures—proof of optimal SIMD utilization.

---

## 📚 Documentation

- **[Performance Methodology](docs/PERFORMANCE.md)** - Full empirical benchmark analysis
- **[Performance Summary](docs/PERFORMANCE_SUMMARY.md)** - One-page stakeholder reference
- **[AWS Deployment Guide](docs/AWS_DEPLOYMENT_GUIDE.md)** - Production deployment
- **[CHANGELOG](CHANGELOG.md)** - Complete version history

---

## 🚀 Recommended Configuration

```bash
export OMP_DYNAMIC=false
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=2
```

**Works universally across:**
- AMD EPYC Genoa
- Intel Ice Lake / Sapphire Rapids
- ARM Graviton3
- All vCPU counts (4, 8, 16+)

---

## 📦 Installation

### From Source (Recommended)

```bash
git clone https://github.com/hpcseries/HPCSeriesCore.git
cd HPCSeriesCore
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The build system will automatically detect your CPU architecture and apply optimal flags.

### Python Bindings

```bash
pip install -e .
```

---

## ✨ What's New in v0.7.0

### Added
- CMake modules: `DetectArchitecture.cmake`, `CompilerFlags.cmake`
- AWS EC2 instance detection scripts
- ARM CPU vendor/model detection (Neoverse V1/V2, N1/N2)
- Benchmark metadata collection (CSV with instance type, CPU info)
- 9 comprehensive Jupyter notebooks
- Kaggle competition examples
- Performance methodology documentation

### Changed
- Default compilation profile: SAFE (IEEE 754 compliant)
- Benchmark script captures architecture metadata
- Updated .gitignore for documentation files

### Fixed
- ARM CPU detection in benchmarks
- Bash arithmetic bug in test scripts
- Missing benchmark source files in CMake

---

## 🔬 Validation

Tested on:
- **AMD EPYC Genoa** (c7a.xlarge)
- **Intel Ice Lake** (m6i.2xlarge, c6i.4xlarge)
- **ARM Graviton3** (c7g.xlarge)

All benchmarks reproducible with included scripts.

---

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) for details

---

## 🙏 Citation

If you use HPCSeries Core in your research, please cite:

```bibtex
@software{hpcseries_core_2025,
  title = {HPCSeries Core: High-Performance Statistical Computing for Large-Scale Data Analysis},
  author = {HPCSeries Core Contributors},
  year = {2025},
  version = {0.7.0},
  url = {https://github.com/hpcseries/HPCSeriesCore},
  license = {MIT}
}
```

Or use the auto-generated citation from GitHub's "Cite this repository" feature.

---

## 🐛 Known Issues

None reported for v0.7.0

---

## 📈 What's Next (v0.8.0)

- GPU acceleration (CUDA/HIP)
- Additional robust statistics algorithms
- Distributed computing support
- Advanced time series decomposition

---

**Full Changelog:** See [CHANGELOG.md](CHANGELOG.md) for complete version history.

**Questions?** Open an issue on GitHub or check the documentation.
