# HPC Series Core - Design Notes

## Architecture Overview

### Language Choice: Fortran + C Interoperability

**Why Fortran for kernels?**
- Native array operations and intrinsics
- Excellent compiler optimizations for numerical code
- Built-in support for multidimensional arrays
- ISO_C_BINDING for seamless C interoperability

**C/C++ Integration**
- C headers provide stable ABI for linking
- Future C++ wrappers for object-oriented interface
- Compatible with Python (ctypes, cffi) and Julia (ccall)

### Module Organization

```
hpcs_core_1d.f90        → Rolling operations, transforms
hpcs_core_reductions.f90 → Aggregations, groupby
hpcs_core_utils.f90      → (future) Helper functions
```

### Build System

CMake provides:
- Cross-platform builds (Linux, macOS, Windows)
- Compiler detection (GFortran, Intel, MSVC)
- Automated testing (CTest)
- Installation and packaging

## Optimization Strategy

### Phase 1: Correctness (v0.1)
- Straightforward implementations
- Focus on algorithmic correctness
- Comprehensive test coverage

### Phase 2: Performance (v0.2)
- SIMD vectorization
- Cache optimization
- Loop unrolling
- OpenMP parallelization for large arrays

### Phase 3: Advanced (v0.3+)
- GPU offloading (CUDA/HIP)
- Distributed computing (MPI)
- Adaptive algorithms based on array size

## Memory Management

- **Principle**: Caller allocates, library computes
- No dynamic allocation in hot paths
- Predictable memory footprint
- Cache-line alignment for large arrays (future)

## Testing Strategy

### Unit Tests
- Correctness on small, hand-verified arrays
- Edge cases (empty, single-element, large windows)
- Numerical accuracy validation

### Integration Tests
- C ABI compatibility
- C++ wrapper functionality
- Cross-platform consistency

### Performance Tests
- Throughput benchmarks (GB/s)
- Scalability analysis
- Comparison with reference implementations

## Future Directions

### Short-term (v0.2)
- Complete reduction implementations
- Optimize rolling operations with Welford's algorithm
- Add OpenMP directives

### Medium-term (v0.3)
- 2D kernel support (e.g., rolling over matrices)
- Additional statistical functions (median, quantiles)
- SIMD intrinsics for critical loops

### Long-term (v1.0)
- GPU acceleration
- Distributed array processing
- Integration with popular frameworks (NumPy, Pandas, Arrow)

## References

- Modern Fortran: https://fortran-lang.org/
- ISO_C_BINDING: Fortran 2003/2008 standards
- CMake Best Practices: https://cmake.org/cmake/help/latest/
- Numerical Recipes: Press et al.

---

**Last Updated**: 2025-11-19
**Maintainer**: HPC Series Core Development Team
