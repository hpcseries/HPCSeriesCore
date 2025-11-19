# HPC Series Core Kernel Specification v0.1

## Overview

This document specifies the computational kernels implemented in the HPC Series Core Library. Each kernel is designed for high-performance numerical computing with a focus on time series analysis and statistical operations.

## Kernel Categories

### 1. Rolling Window Operations

Rolling operations apply a function over a sliding window of fixed size.

#### 1.1 Rolling Mean
- **Signature**: `hpcs_rolling_mean(input, n, window, output)`
- **Description**: Computes the arithmetic mean over a sliding window
- **Complexity**: O(n)
- **Memory**: O(1) auxiliary

#### 1.2 Rolling Standard Deviation
- **Signature**: `hpcs_rolling_std(input, n, window, output)`
- **Description**: Computes standard deviation over a sliding window
- **Complexity**: O(n × window)
- **Memory**: O(1) auxiliary

#### 1.3 Rolling Min/Max
- **Signature**: `hpcs_rolling_min/max(input, n, window, output)`
- **Description**: Computes minimum/maximum over a sliding window
- **Complexity**: O(n × window)
- **Memory**: O(1) auxiliary

#### 1.4 Rolling Sum
- **Signature**: `hpcs_rolling_sum(input, n, window, output)`
- **Description**: Computes cumulative sum over a sliding window
- **Complexity**: O(n)
- **Memory**: O(1) auxiliary

### 2. Statistical Transformations

#### 2.1 Z-Score Normalization
- **Signature**: `hpcs_zscore(input, n, output)`
- **Description**: Normalizes data to zero mean and unit variance
- **Formula**: `z = (x - μ) / σ`
- **Complexity**: O(n)

#### 2.2 Rank Transform
- **Signature**: `hpcs_rank(input, n, output)`
- **Description**: Converts values to their ordinal ranks
- **Complexity**: O(n²) naive, O(n log n) with sorting
- **Memory**: O(n) for indices

### 3. Reductions

#### 3.1 Simple Reductions
- **Mean**: `hpcs_mean(input, n) → scalar`
- **Sum**: `hpcs_sum(input, n) → scalar`
- **Std**: `hpcs_std(input, n) → scalar`

#### 3.2 Grouped Reductions
- **Group Mean**: `hpcs_groupby_mean(input, n, groups, num_groups, output)`
- **Group Sum**: `hpcs_groupby_sum(input, n, groups, num_groups, output)`

## Implementation Notes

### Numerical Stability
- Use Kahan summation for long accumulations
- Welford's method for variance computation
- IEEE 754 compliance

### Performance Considerations
- Vectorization opportunities in rolling operations
- Cache-friendly memory access patterns
- OpenMP parallelization for large arrays (future)

### Edge Cases
- Window size larger than array: treated as full array
- Empty arrays: return NaN or 0 as appropriate
- Single-element window: identity operation

## Version History

- **v0.1.0**: Initial specification with core kernels
