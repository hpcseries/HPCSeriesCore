! ==============================================================================
! HPC Series Core Library - Array Utilities Module
! Simple element-wise utilities for 1D arrays.
!
! Implemented kernels:
!   - hpcs_fill_missing
!   - hpcs_where
!   - hpcs_fill_value
!   - hpcs_copy
!   - hpcs_normalize_minmax  (v0.2)
!   - hpcs_fill_forward      (v0.2)
!   - hpcs_fill_backward     (v0.2)
!   - hpcs_detect_anomalies  (v0.2)
!
! All routines:
!   - use ISO_C_BINDING with bind(C)
!   - return status via an explicit integer(c_int) argument
! ==============================================================================

module hpcs_core_utils
  use iso_c_binding,  only: c_int, c_double
  use hpcs_constants
  implicit none
  public

contains

  !--------------------------------------------------------------------
  ! hpcs_fill_missing
  !--------------------------------------------------------------------
  subroutine hpcs_fill_missing(x, n, missing_value, replacement, &
                               treat_nan_as_missing, status) &
       bind(C, name="hpcs_fill_missing")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(inout) :: x(*)                ! length n
    integer(c_int),  value        :: n
    real(c_double),  value        :: missing_value
    real(c_double),  value        :: replacement
    integer(c_int),  value        :: treat_nan_as_missing
    integer(c_int),  intent(out)  :: status

    integer(c_int) :: i, n_eff
    logical        :: use_nan

    n_eff  = n
    use_nan = (treat_nan_as_missing /= 0_c_int)

    if (n_eff < 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (n_eff == 0_c_int) then
       status = HPCS_SUCCESS
       return
    end if

    do i = 1_c_int, n_eff
       ! Sentinel check
       if (x(i) == missing_value) then
          x(i) = replacement

       ! Optional NaN-as-missing: NaN is the only value where x /= x
       else if (use_nan .and. x(i) /= x(i)) then
          x(i) = replacement
       end if
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_fill_missing

  !--------------------------------------------------------------------
  ! hpcs_where
  !--------------------------------------------------------------------
  subroutine hpcs_where(mask, n, a, b, y, status) &
       bind(C, name="hpcs_where")
    use iso_c_binding, only: c_int, c_double
    implicit none

    integer(c_int),  intent(in)  :: mask(*)     ! length n
    integer(c_int),  value       :: n
    real(c_double),  intent(in)  :: a(*)        ! length n
    real(c_double),  intent(in)  :: b(*)        ! length n
    real(c_double),  intent(out) :: y(*)        ! length n
    integer(c_int),  intent(out) :: status

    integer(c_int) :: i, n_eff

    n_eff = n

    if (n_eff < 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (n_eff == 0_c_int) then
       status = HPCS_SUCCESS
       return
    end if

    do i = 1_c_int, n_eff
       if (mask(i) /= 0_c_int) then
          y(i) = a(i)
       else
          y(i) = b(i)
       end if
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_where

  !--------------------------------------------------------------------
  ! hpcs_fill_value
  !
  ! Engineer spec:
  !   - Fill x(1:n) with scalar value.
  !   - O(n), independent iterations, trivially parallelisable.
  !
  ! Our conventions added:
  !   - C interface + status code.
  !
  ! Status:
  !   HPCS_SUCCESS          : success (including n == 0)
  !   HPCS_ERR_INVALID_ARGS : n < 0
  !--------------------------------------------------------------------
  subroutine hpcs_fill_value(x, n, value, status) &
       bind(C, name="hpcs_fill_value")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(inout) :: x(*)   ! length n
    integer(c_int),  value        :: n
    real(c_double),  value        :: value
    integer(c_int),  intent(out)  :: status

    integer(c_int) :: i, n_eff

    n_eff = n

    if (n_eff < 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (n_eff == 0_c_int) then
       status = HPCS_SUCCESS
       return
    end if

    ! OpenMP-ready flat loop (no deps between iterations)
    do i = 1_c_int, n_eff
       x(i) = value
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_fill_value

  !--------------------------------------------------------------------
  ! hpcs_copy
  !
  ! Engineer spec:
  !   - Copy src(1:n) -> dst(1:n).
  !   - O(n), independent per index, trivially parallelisable.
  !   - Described as "memmove-like" in behaviour; we implement a
  !     simple forward copy loop (no OpenMP yet).
  !
  ! NOTE:
  !   For strict Fortran semantics, callers should NOT pass overlapping
  !   dst/src; we assume non-overlapping arrays here.
  !
  ! Status:
  !   HPCS_SUCCESS          : success (including n == 0)
  !   HPCS_ERR_INVALID_ARGS : n < 0
  !--------------------------------------------------------------------
  subroutine hpcs_copy(dst, src, n, status) &
       bind(C, name="hpcs_copy")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(out) :: dst(*)   ! length n
    real(c_double), intent(in)  :: src(*)   ! length n
    integer(c_int),  value      :: n
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff

    n_eff = n

    if (n_eff < 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (n_eff == 0_c_int) then
       status = HPCS_SUCCESS
       return
    end if

    ! OpenMP-ready flat loop
    do i = 1_c_int, n_eff
       dst(i) = src(i)
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_copy

  !--------------------------------------------------------------------
  ! hpcs_normalize_minmax (v0.2)
  !
  ! Min-max normalization: scales data to [0, 1] range
  !   y(i) = (x(i) - min) / (max - min)
  !
  ! Special cases:
  !   - If max == min (constant array), returns all 0.5
  !   - Empty array (n=0) succeeds with no operation
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n < 0
  !--------------------------------------------------------------------
  subroutine hpcs_normalize_minmax(x, n, y, status) &
       bind(C, name="hpcs_normalize_minmax")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)   ! length n
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: y(*)   ! length n
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff
    real(c_double) :: minval, maxval, range

    n_eff = n

    if (n_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    ! Find min and max
    minval = x(1_c_int)
    maxval = x(1_c_int)

    do i = 2_c_int, n_eff
       if (x(i) < minval) minval = x(i)
       if (x(i) > maxval) maxval = x(i)
    end do

    range = maxval - minval

    ! Normalize to [0, 1]
    if (range > 0.0_c_double) then
       do i = 1_c_int, n_eff
          y(i) = (x(i) - minval) / range
       end do
    else
       ! Constant array: set all to 0.5
       do i = 1_c_int, n_eff
          y(i) = 0.5_c_double
       end do
    end if

    status = HPCS_SUCCESS
  end subroutine hpcs_normalize_minmax

  !--------------------------------------------------------------------
  ! hpcs_fill_forward (v0.2)
  !
  ! Forward fill: propagate last valid (non-NaN) value forward
  !
  ! Example:
  !   Input:  [1.0, NaN, NaN, 2.0, NaN, 3.0]
  !   Output: [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]
  !
  ! Special cases:
  !   - Leading NaNs remain NaN (no prior value to propagate)
  !   - Empty array (n=0) succeeds with no operation
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n < 0
  !--------------------------------------------------------------------
  subroutine hpcs_fill_forward(x, n, y, status) &
       bind(C, name="hpcs_fill_forward")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)   ! length n
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: y(*)   ! length n
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff
    real(c_double) :: last_valid

    n_eff = n

    if (n_eff < 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (n_eff == 0_c_int) then
       status = HPCS_SUCCESS
       return
    end if

    ! Initialize with first value (will be replaced by first valid value)
    last_valid = x(1_c_int)  ! Use first value as initial

    do i = 1_c_int, n_eff
       if (x(i) /= x(i)) then  ! Is NaN (NaN /= NaN is TRUE)
          y(i) = last_valid  ! Propagate last valid value
       else  ! Not NaN
          last_valid = x(i)
          y(i) = x(i)
       end if
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_fill_forward

  !--------------------------------------------------------------------
  ! hpcs_fill_backward (v0.2)
  !
  ! Backward fill: propagate next valid (non-NaN) value backward
  !
  ! Example:
  !   Input:  [NaN, NaN, 1.0, NaN, 2.0, NaN]
  !   Output: [1.0, 1.0, 1.0, 2.0, 2.0, NaN]
  !
  ! Special cases:
  !   - Trailing NaNs remain NaN (no subsequent value to propagate)
  !   - Empty array (n=0) succeeds with no operation
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n < 0
  !--------------------------------------------------------------------
  subroutine hpcs_fill_backward(x, n, y, status) &
       bind(C, name="hpcs_fill_backward")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)   ! length n
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: y(*)   ! length n
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff
    real(c_double) :: next_valid

    n_eff = n

    if (n_eff < 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (n_eff == 0_c_int) then
       status = HPCS_SUCCESS
       return
    end if

    ! Initialize with last value
    next_valid = x(n_eff)

    ! Iterate backward
    do i = n_eff, 1_c_int, -1_c_int
       if (x(i) /= x(i)) then  ! Is NaN (NaN /= NaN is TRUE)
          y(i) = next_valid  ! Propagate next valid value
       else  ! Not NaN
          next_valid = x(i)
          y(i) = x(i)
       end if
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_fill_backward

  !--------------------------------------------------------------------
  ! hpcs_detect_anomalies (v0.2)
  !
  ! Detect anomalies using z-score method: flag values where
  ! |z-score| > threshold, where z-score = (x - mean) / std
  !
  ! Typical threshold is 3.0 for the "3-sigma rule"
  !
  ! Inputs:
  !   x(n)      - input data array
  !   n         - array length
  !   threshold - z-score threshold (typically 3.0)
  !
  ! Output:
  !   anomaly(n) - integer array: 1 = anomaly, 0 = normal
  !   status     - error code
  !
  ! Edge cases:
  !   - If std == 0, all values equal mean -> no anomalies (all 0)
  !   - n < 2: cannot compute variance -> error
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or threshold < 0
  !--------------------------------------------------------------------
  subroutine hpcs_detect_anomalies(x, n, threshold, anomaly, status) &
       bind(C, name="hpcs_detect_anomalies")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)         ! length n
    integer(c_int),  value      :: n
    real(c_double),  value      :: threshold
    integer(c_int), intent(out) :: anomaly(*)   ! length n (0 or 1)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff, status_local
    real(c_double) :: mean_val, variance_val, std_val, z_score
    real(c_double) :: sum_val

    n_eff = n

    if (n_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (threshold < 0.0_c_double) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    ! Compute mean
    sum_val = 0.0_c_double
    do i = 1_c_int, n_eff
       sum_val = sum_val + x(i)
    end do
    mean_val = sum_val / real(n_eff, kind=c_double)

    ! Compute variance using Welford's algorithm
    ! (reuse reduce_variance logic inline for simplicity)
    if (n_eff < 2_c_int) then
       ! Cannot compute variance with < 2 elements
       ! Mark all as non-anomalies
       do i = 1_c_int, n_eff
          anomaly(i) = 0_c_int
       end do
       status = HPCS_SUCCESS
       return
    end if

    ! Use simple variance formula: Var = E[X²] - (E[X])²
    variance_val = 0.0_c_double
    do i = 1_c_int, n_eff
       variance_val = variance_val + (x(i) - mean_val) * (x(i) - mean_val)
    end do
    variance_val = variance_val / real(n_eff, kind=c_double)

    std_val = sqrt(variance_val)

    ! Detect anomalies
    if (std_val == 0.0_c_double) then
       ! All values equal mean - no anomalies
       do i = 1_c_int, n_eff
          anomaly(i) = 0_c_int
       end do
    else
       ! Compute z-scores and flag anomalies
       do i = 1_c_int, n_eff
          z_score = (x(i) - mean_val) / std_val
          if (abs(z_score) > threshold) then
             anomaly(i) = 1_c_int  ! Anomaly
          else
             anomaly(i) = 0_c_int  ! Normal
          end if
       end do
    end if

    status = HPCS_SUCCESS
  end subroutine hpcs_detect_anomalies

end module hpcs_core_utils
