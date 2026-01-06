! ==============================================================================
! HPC Series Core Library - Data Quality and Robust Scoring Module (v0.3)
!
! This module provides simple elementwise data quality operations (value
! clipping and winsorisation) and robust z-score computation based on the
! median and MAD.  These routines operate on contiguous 1-D arrays of
! double precision values.  NaN values propagate through the operations as
! specified.  All loops are structured for easy OpenMP insertion.
!
! Kernels implemented:
!   - hpcs_clip
!   - hpcs_winsorize_by_quantiles
!   - hpcs_robust_zscore
!   - hpcs_detect_anomalies_robust
!   - hpcs_remove_outliers_iterative
!
! The routines follow the v0.3 specification.  Quantile bounds are computed
! via hpcs_quantile.  Degeneracy in the MAD during robust z-score computation
! results in status = HPCS_ERR_NUMERIC_FAIL and NaN outputs.  Optional
! scaling can be applied to match the scale of a normal distribution.
! ============================================================================

module hpcs_core_quality
  use iso_c_binding,  only: c_int, c_double
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  use hpcs_constants
  use hpcs_core_stats, only: hpcs_quantile, hpcs_median, hpcs_mad
  implicit none
  private
  public :: hpcs_clip
  public :: hpcs_winsorize_by_quantiles
  public :: hpcs_robust_zscore
  public :: hpcs_detect_anomalies_robust
  public :: hpcs_remove_outliers_iterative

contains

  !--------------------------------------------------------------------------
  ! hpcs_clip
  !
  ! Clamp each element of x into the range [min_val, max_val].  NaN values
  ! remain unchanged.  Invalid arguments (n<=0 or min_val > max_val) trigger
  ! status = HPCS_ERR_INVALID_ARGS.  This operation is in-place on the input
  ! array.
  !
  ! Arguments (C view):
  !   x       - double*         in/out array of length n
  !   n       - int            number of elements
  !   min_val - double         lower bound
  !   max_val - double         upper bound
  !   status  - int*           output status code
  !
  !--------------------------------------------------------------------------
  subroutine hpcs_clip(x, n, min_val, max_val, status) &
       bind(C, name="hpcs_clip")
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
    implicit none
    real(c_double), intent(inout) :: x(*)
    integer(c_int),  value        :: n
    real(c_double),  value        :: min_val
    real(c_double),  value        :: max_val
    integer(c_int),  intent(out)  :: status

    integer(c_int) :: i, n_eff
    logical :: is_nan

    n_eff = n
    if (n_eff <= 0_c_int .or. min_val > max_val) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! SIMD-optimized clipping with explicit NaN handling
    ! Skip NaN values to preserve them (minimal branching)
    !$omp simd
    do i = 1_c_int, n_eff
      is_nan = ieee_is_nan(x(i))
      if (.not. is_nan) then
        x(i) = min(max(x(i), min_val), max_val)
      end if
    end do
    !$omp end simd
    status = HPCS_SUCCESS
  end subroutine hpcs_clip

  !--------------------------------------------------------------------------
  ! hpcs_winsorize_by_quantiles
  !
  ! Clip values to quantile-based bounds.  Computes lowBound = quantile(x,
  ! q_low) and highBound = quantile(x, q_high) using Type 7 quantiles.  Each
  ! element x(i) is then clamped into [lowBound, highBound], leaving NaNs
  ! unchanged.  Invalid arguments (n<=0, q_low<0, q_high>1 or q_low>q_high)
  ! produce status = HPCS_ERR_INVALID_ARGS.  Failures in quantile
  ! computations propagate their status codes.
  !
  ! Arguments (C view):
  !   x       - double*      in/out array of length n
  !   n       - int         number of elements
  !   q_low   - double      lower quantile in [0,1]
  !   q_high  - double      upper quantile in [0,1]
  !   status  - int*        output status code
  !
  !--------------------------------------------------------------------------
  subroutine hpcs_winsorize_by_quantiles(x, n, q_low, q_high, status) &
       bind(C, name="hpcs_winsorize_by_quantiles")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(inout) :: x(*)
    integer(c_int),  value        :: n
    real(c_double),  value        :: q_low
    real(c_double),  value        :: q_high
    integer(c_int),  intent(out)  :: status

    integer(c_int) :: n_eff, st
    real(c_double) :: lowBound, highBound
    integer(c_int) :: i

    n_eff = n
    ! Validate arguments
    if (n_eff <= 0_c_int .or. q_low < 0.0_c_double .or. q_high > 1.0_c_double .or. q_low > q_high) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Compute lower bound
    call hpcs_quantile(x, n_eff, q_low, lowBound, st)
    if (st /= HPCS_SUCCESS) then
      status = st
      return
    end if
    ! Compute upper bound
    call hpcs_quantile(x, n_eff, q_high, highBound, st)
    if (st /= HPCS_SUCCESS) then
      status = st
      return
    end if

    ! Apply clamping based on quantile bounds (branchless for vectorization)
    do i = 1_c_int, n_eff
      ! Preserve NaNs
      if (x(i) /= x(i)) cycle
      ! Branchless: compiler can vectorize this
      x(i) = min(max(x(i), lowBound), highBound)
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_winsorize_by_quantiles

  !--------------------------------------------------------------------------
  ! hpcs_robust_zscore
  !
  ! Compute a robust z-score for each element of x using the median and MAD.
  ! The z-score is defined as (x(i) - median) / (mad * scale), where scale
  ! defaults to 1.4826 to make the MAD comparable to the standard deviation of
  ! a normal distribution.  This C-bound version uses the default scale.
  ! If the MAD degenerates (≈0), the routine sets status =
  ! HPCS_ERR_NUMERIC_FAIL and fills y with NaNs.  NaN values in x propagate
  ! to y.  Invalid arguments (n<=0) trigger status = HPCS_ERR_INVALID_ARGS.
  !
  ! Arguments (C view):
  !   x      - const double*    input array of length n
  !   n      - int             number of elements
  !   y      - double*         output array of length n
  !   status - int*            output status code
  !
  !--------------------------------------------------------------------------
  subroutine hpcs_robust_zscore(x, n, y, status) &
       bind(C, name="hpcs_robust_zscore")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    ! Call implementation with default scale
    call robust_zscore_impl(x, n, y, status, 1.4826_c_double)
  end subroutine hpcs_robust_zscore

  !--------------------------------------------------------------------------
  ! robust_zscore_impl
  !
  ! Internal implementation of robust z-score computation with explicit scale.
  !--------------------------------------------------------------------------
  subroutine robust_zscore_impl(x, n, y, status, scale)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status
    real(c_double),  value      :: scale

    integer(c_int) :: n_eff, st
    real(c_double) :: med, mad_val
    integer(c_int) :: i

    n_eff = n
    if (n_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Compute median
    call hpcs_median(x, n_eff, med, st)
    if (st /= HPCS_SUCCESS) then
      status = st
      return
    end if
    ! Compute MAD
    call hpcs_mad(x, n_eff, mad_val, st)
    if (st == HPCS_ERR_NUMERIC_FAIL) then
      ! Degenerate distribution: fill y with NaNs
      do i = 1_c_int, n_eff
        y(i) = ieee_value(0.0_c_double, ieee_quiet_nan)
      end do
      status = HPCS_ERR_NUMERIC_FAIL
      return
    else if (st /= HPCS_SUCCESS) then
      status = st
      return
    end if
    ! Compute robust z-scores
    do i = 1_c_int, n_eff
      if (x(i) /= x(i)) then
        y(i) = ieee_value(0.0_c_double, ieee_quiet_nan)
      else
        y(i) = (x(i) - med) / (mad_val * scale)
      end if
    end do
    status = HPCS_SUCCESS
  end subroutine robust_zscore_impl

  !--------------------------------------------------------------------------
  ! hpcs_detect_anomalies_robust
  !
  ! Detect anomalies using robust z-score based on median and MAD instead of
  ! mean and standard deviation. This method is resistant to outliers because
  ! the median and MAD are not affected by extreme values.
  !
  ! The robust z-score is computed as:
  !   z_robust = (x - median) / (MAD × 1.4826)
  !
  ! Values where |z_robust| > threshold are flagged as anomalies.
  ! The scale factor 1.4826 makes the MAD comparable to standard deviation
  ! for normal distributions.
  !
  ! This is the recommended method for outlier detection in real-world data
  ! as it avoids the "masking" problem where outliers affect the detection
  ! threshold.
  !
  ! Arguments (C view):
  !   x         - const double*    input array of length n
  !   n         - int             number of elements
  !   threshold - double          z-score threshold (e.g., 3.0 for 3-sigma)
  !   anomaly   - int*            output: 1=anomaly, 0=normal
  !   status    - int*            output status code
  !
  !--------------------------------------------------------------------------
  subroutine hpcs_detect_anomalies_robust(x, n, threshold, anomaly, status) &
       bind(C, name="hpcs_detect_anomalies_robust")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double),  value      :: threshold
    integer(c_int),  intent(out):: anomaly(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, st, i
    real(c_double) :: med, mad_val, z_robust
    real(c_double), parameter :: scale = 1.4826_c_double

    n_eff = n
    ! Validate arguments
    if (n_eff <= 0_c_int .or. threshold < 0.0_c_double) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Special case: n < 2 cannot compute variance, return all zeros
    if (n_eff < 2_c_int) then
      anomaly(1) = 0_c_int
      status = HPCS_SUCCESS
      return
    end if

    ! Compute median
    call hpcs_median(x, n_eff, med, st)
    if (st /= HPCS_SUCCESS) then
      status = st
      return
    end if

    ! Compute MAD
    call hpcs_mad(x, n_eff, mad_val, st)
    if (st == HPCS_ERR_NUMERIC_FAIL) then
      ! MAD = 0: all values equal median, no anomalies possible
      do i = 1_c_int, n_eff
        anomaly(i) = 0_c_int
      end do
      status = HPCS_SUCCESS
      return
    else if (st /= HPCS_SUCCESS) then
      status = st
      return
    end if

    ! Detect anomalies using robust z-score
    do i = 1_c_int, n_eff
      if (x(i) /= x(i)) then
        ! NaN values: flag as not anomaly (or could flag as anomaly)
        anomaly(i) = 0_c_int
      else
        z_robust = abs((x(i) - med) / (mad_val * scale))
        if (z_robust > threshold) then
          anomaly(i) = 1_c_int  ! Anomaly
        else
          anomaly(i) = 0_c_int  ! Normal
        end if
      end if
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_detect_anomalies_robust

  !--------------------------------------------------------------------------
  ! hpcs_remove_outliers_iterative
  !
  ! Iteratively remove outliers from data using robust z-score detection.
  ! This function repeatedly:
  !   1. Detects outliers using median/MAD-based robust z-score
  !   2. Removes detected outliers
  !   3. Recomputes statistics on remaining data
  !   4. Repeats until convergence (no new outliers) or max iterations
  !
  ! This iterative approach is more effective than single-pass outlier
  ! detection for heavily contaminated datasets.
  !
  ! The function returns:
  !   - cleaned: array containing non-outlier values (user must allocate)
  !   - n_clean: number of non-outlier values
  !   - iterations: actual number of iterations performed
  !
  ! Arguments (C view):
  !   x          - const double*    input array of length n
  !   n          - int             number of input elements
  !   threshold  - double          z-score threshold (e.g., 3.0)
  !   max_iter   - int             maximum iterations (e.g., 10)
  !   cleaned    - double*         output: cleaned data (size >= n)
  !   n_clean    - int*            output: number of cleaned values
  !   iterations - int*            output: actual iterations performed
  !   status     - int*            output status code
  !
  ! Notes:
  !   - The cleaned array must be pre-allocated with size >= n
  !   - Convergence occurs when no new outliers are detected
  !   - If max_iter is reached, the current cleaned data is returned
  !   - NaN values are treated as outliers and removed
  !
  !--------------------------------------------------------------------------
  subroutine hpcs_remove_outliers_iterative(x, n, threshold, max_iter, &
                                             cleaned, n_clean, iterations, status) &
       bind(C, name="hpcs_remove_outliers_iterative")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double),  value      :: threshold
    integer(c_int),  value      :: max_iter
    real(c_double), intent(out) :: cleaned(*)
    integer(c_int),  intent(out):: n_clean
    integer(c_int),  intent(out):: iterations
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, iter, st, i, j, n_current, n_next
    integer(c_int), allocatable :: anomaly(:)
    real(c_double), allocatable :: current_data(:), next_data(:)
    logical :: converged

    n_eff = n
    ! Validate arguments
    if (n_eff <= 0_c_int .or. threshold < 0.0_c_double .or. max_iter <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Allocate working arrays
    allocate(anomaly(n_eff), current_data(n_eff), next_data(n_eff), stat=st)
    if (st /= 0) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Initialize with input data
    do i = 1_c_int, n_eff
      current_data(i) = x(i)
    end do
    n_current = n_eff

    ! Iterative outlier removal
    converged = .false.
    do iter = 1_c_int, max_iter
      ! Detect anomalies in current data
      call hpcs_detect_anomalies_robust(current_data, n_current, threshold, anomaly, st)
      if (st /= HPCS_SUCCESS) then
        status = st
        deallocate(anomaly, current_data, next_data)
        return
      end if

      ! Remove outliers and NaN values
      n_next = 0_c_int
      do i = 1_c_int, n_current
        ! Keep only non-anomalies and non-NaN values
        if (anomaly(i) == 0_c_int .and. current_data(i) == current_data(i)) then
          n_next = n_next + 1_c_int
          next_data(n_next) = current_data(i)
        end if
      end do

      ! Check for convergence (no outliers removed)
      if (n_next == n_current) then
        converged = .true.
        iterations = iter
        exit
      end if

      ! Check if all data was removed
      if (n_next == 0_c_int) then
        ! All data flagged as outliers - return empty result
        n_clean = 0_c_int
        iterations = iter
        status = HPCS_SUCCESS
        deallocate(anomaly, current_data, next_data)
        return
      end if

      ! Copy next_data to current_data for next iteration
      do i = 1_c_int, n_next
        current_data(i) = next_data(i)
      end do
      n_current = n_next

      ! Set iterations in case we reach max_iter
      iterations = iter
    end do

    ! Copy cleaned data to output
    n_clean = n_current
    do i = 1_c_int, n_current
      cleaned(i) = current_data(i)
    end do

    ! Clean up
    deallocate(anomaly, current_data, next_data)
    status = HPCS_SUCCESS
  end subroutine hpcs_remove_outliers_iterative

end module hpcs_core_quality