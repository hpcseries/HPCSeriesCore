# HPCSeries Core v0.8.0 — Execution Modes, Pipeline & Workspace APIs

**Release Date:** 2026-02-18
**Tag:** `v0.8.0`

---

## Major Features

### Execution Mode System — Safety vs Performance Trade-offs

- Global API: `set_execution_mode()`, `get_execution_mode()`
- Three modes: `MODE_SAFE` (IEEE 754, full validation), `MODE_FAST` (relaxed math), `MODE_DETERMINISTIC` (bit-exact reproducibility)
- Per-call mode override via `mode` parameter
- Thread-safe via OpenMP threadprivate storage
- Zero overhead via compile-time dispatcher pattern
- 8 functions with mode support: `ewma`, `ewvar`, `ewstd`, `cumulative_min`, `cumulative_max`, `convolve_valid`, `trimmed_mean`, `winsorized_mean`

### Pipeline API — Composable Kernel Execution

- 24 pipeline stages (12 original + 12 new in v0.8.0)
- New stages: `cumulative_min()`, `cumulative_max()`, `fill_forward()`, `prefix_sum()`, `convolve(kernel)`, `lag(k)`, `log_return()`, `pct_change()`, `scale(factor)`, `shift(offset)`, `abs()`, `sqrt()`
- Chained method syntax: `p.ewma(0.1).zscore().execute(x)`
- `pipeline_execute()` gains `out_n` parameter for variable-length output (convolve)
- Optional `workspace` for zero-allocation execution on hot paths

### Workspace API — Pre-Allocated SIMD-Aligned Memory

- `workspace(bytes=67108864)` — 64-byte aligned, 64 MB default
- `.size` property and `.reserve(bytes)` method
- Passed to `pipeline(ws=...)` for zero-allocation execution

### Exponential Weighted Statistics (Group A) — 15–60x faster than pandas

- `ewma()` — Exponentially weighted moving average
- `ewvar()` — Exponentially weighted variance (Welford's method, v[0]=0)
- `ewstd()` — Exponentially weighted standard deviation
- Single-pass O(n), numerically stable, matches pandas API

### Time Series Transforms (Group B) — 2–4x faster than NumPy

- `diff()` — Finite differencing (arbitrary order)
- `cumulative_min()` — Running minimum with execution modes
- `cumulative_max()` — Running maximum with execution modes

### FIR Filters (Group C) — Template-specialised convolution

- `convolve_valid()` — 1D convolution with execution modes
- Template specialisations for kernel sizes 3, 5, 7, 9, 11, 13, 15
- OpenMP parallelisation in SAFE/FAST modes

### Advanced Robust Statistics (Group D) — 10–15x faster than SciPy

- `trimmed_mean()` — Mean after discarding extremes
- `winsorized_mean()` — Mean after clamping extremes
- Deterministic O(n) selection (introselect)

---

## Breaking Changes

**C API only** — `pipeline_execute()` signature gains `size_t *out_n` parameter:

```c
// v0.7.x
int pipeline_execute(hpcs_pipeline_t *p, const double *in, size_t n, double *out);

// v0.8.0
int pipeline_execute(hpcs_pipeline_t *p, const double *in, size_t n, double *out, size_t *out_n);
```

**Python API is unchanged.** The `execute()` method handles `out_n` internally.

---

## Performance

| Operation | vs pandas/NumPy/SciPy |
|-----------|----------------------|
| EWMA | 15–60x faster |
| EWVAR | 15–60x faster |
| Cumulative min/max | 2–4x faster |
| Trimmed/winsorized mean | 10–15x faster |
| FIR convolution | 2–4x faster |

Thread configuration: `OMP_NUM_THREADS=2` optimal across all tested architectures (AMD EPYC Genoa, Intel Ice Lake, ARM Graviton3).

---

## Documentation

- `docs/GETTING_STARTED.md` — investor-ready guide
- `notebooks/10_exponential_weighted_statistics.ipynb` — comprehensive EWMA/EWVAR examples
- `notebooks/00_getting_started.ipynb` — updated for v0.8.0
- Execution mode API fully documented

---

## Installation

### Python (from source)

```bash
git clone https://github.com/hpcseries/HPCSeriesCore.git
cd HPCSeriesCore
pip install -e .
python3 -c "import hpcs; print(hpcs.__version__)"  # → 0.8.0
```

### Docker (recommended for notebooks)

```bash
docker compose -f docker-compose.python.yml up --build hpcs-jupyter
# Open: http://localhost:8888
```

### From Source (C library)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build . -j$(nproc)
ctest --output-on-failure
```

---

## Verification

```python
import hpcs
import numpy as np

# Version check
assert hpcs.__version__ == '0.8.0'

# Pipeline API
x = np.random.randn(10000).astype(float)
p = hpcs.pipeline()
p.ewma(0.1).zscore()
out = p.execute(x)
print('Pipeline OK, output shape:', out.shape)

# Workspace API
ws = hpcs.workspace()
print('Workspace size (bytes):', ws.size)
p2 = hpcs.pipeline(ws=ws)
p2.ewma(0.1)
out2 = p2.execute(x)
print('Workspace pipeline OK')
```

---

## License

Apache License 2.0 — See [LICENSE](../LICENSE) for details.

---

## Citation

```bibtex
@software{hpcseries_core_2026,
  title   = {HPCSeries Core: High-Performance Statistical Computing for Large-Scale Data Analysis},
  author  = {HPCSeries Core Contributors},
  year    = {2026},
  month   = {2},
  version = {0.8.0},
  url     = {https://github.com/hpcseries/HPCSeriesCore},
  license = {Apache-2.0}
}
```

Or use the auto-generated citation from GitHub's "Cite this repository" feature.

---

## Known Issues

None reported for v0.8.0.

---

## What's Next (v0.9.0)

- Performance optimisation phase (benchmarks deferred from v0.8.0)
- GPU acceleration (CUDA/HIP)
- Additional robust statistics algorithms

---

**Full Changelog:** See [CHANGELOG.md](../CHANGELOG.md) for complete version history.
