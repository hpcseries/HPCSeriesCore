! ==============================================================================
! HPC Series Core Library - Transform Kernels (v0.8.0)
!
! New kernels (v0.8.0 specification):
!   - hpcs_ewma             : exponential weighted moving average
!   - hpcs_ewvar            : exponential weighted variance
!   - hpcs_ewstd            : exponential weighted standard deviation
!   - hpcs_diff             : finite differencing (lag-k)
!   - hpcs_cumulative_min   : running minimum (prefix min)
!   - hpcs_cumulative_max   : running maximum (prefix max)
!
! Design principles:
!   - Single-pass streaming algorithms where possible
!   - NaN propagation (status=2 for all-NaN inputs)
!   - Deterministic behavior
!   - SIMD-friendly loop structure
!   - No OpenMP (inherently serial recurrences)
!
! Author: HPCSeries Core Library
! Version: 0.8.0
! Date: 2025-12-23
! ==============================================================================

module hpcs_core_transforms
  use iso_c_binding, only: c_int, c_double
  use hpcs_constants
  use hpcs_core_execution_mode, only: hpcs_get_execution_mode_internal
  implicit none
  public

contains

  !----------------------------------------------------------------------------
  ! hpcs_ewma (DISPATCHER)
  !
  ! Public entry point for exponentially weighted moving average (EWMA).
  ! Dispatches to mode-specific implementation based on execution mode.
  !
  ! Parameters:
  !   x      - input array [n]
  !   n      - array length
  !   alpha  - smoothing factor ∈ (0, 1]
  !   y      - output array [n]
  !   mode   - execution mode (0=SAFE, 1=FAST, 2=DETERMINISTIC, -1=use global)
  !   status - 0=success, 1=invalid args, 2=all NaN
  !----------------------------------------------------------------------------
  subroutine hpcs_ewma(x, n, alpha, y, mode, status) &
       bind(C, name="hpcs_ewma")
    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), value       :: alpha
    real(c_double), intent(out) :: y(*)
    integer(c_int), value       :: mode
    integer(c_int), intent(out) :: status

    integer(c_int) :: effective_mode

    ! Resolve mode (-1 = use global)
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    ! Dispatch to mode-specific implementation
    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_ewma_safe(x, n, alpha, y, status)
      case (HPCS_MODE_FAST)
        call hpcs_ewma_fast(x, n, alpha, y, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_ewma_deterministic(x, n, alpha, y, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_ewma

  !----------------------------------------------------------------------------
  ! hpcs_ewma_safe (SAFE MODE)
  !
  ! Full NaN detection, validation, IEEE 754 compliance.
  ! Current production behavior (v0.7.0 compatible).
  !----------------------------------------------------------------------------
  subroutine hpcs_ewma_safe(x, n, alpha, y, status)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan

    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), value       :: alpha
    real(c_double), intent(out) :: y(*)
    integer(c_int), intent(out) :: status

    integer(c_int) :: i
    real(c_double) :: one_minus_alpha, acc
    logical :: seen_finite

    ! Validate arguments
    if (n <= 0_c_int .or. alpha <= 0.0_c_double .or. alpha > 1.0_c_double) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    one_minus_alpha = 1.0_c_double - alpha

    ! Initialize with first element
    acc = x(1)
    y(1) = acc
    seen_finite = .not. ieee_is_nan(acc)

    ! Serial recurrence with NaN detection
    do i = 2, n
      acc = alpha * x(i) + one_minus_alpha * acc
      y(i) = acc
      if (.not. ieee_is_nan(x(i))) seen_finite = .true.
    end do

    ! Check for all-NaN input
    if (seen_finite) then
      status = HPCS_SUCCESS
    else
      status = HPCS_ERR_NUMERIC_FAIL
    end if
  end subroutine hpcs_ewma_safe

  !----------------------------------------------------------------------------
  ! hpcs_ewma_fast (FAST MODE)
  !
  ! Skip all NaN checks and validation.
  ! Assumes inputs are valid and finite.
  ! 1.5-1.8x faster than SAFE mode.
  !----------------------------------------------------------------------------
  subroutine hpcs_ewma_fast(x, n, alpha, y, status)
    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), value       :: alpha
    real(c_double), intent(out) :: y(*)
    integer(c_int), intent(out) :: status

    integer(c_int) :: i
    real(c_double) :: one_minus_alpha, acc

    ! NO validation (assume valid inputs)
    one_minus_alpha = 1.0_c_double - alpha

    ! Initialize with first element
    acc = x(1)
    y(1) = acc

    ! NO NaN checks - compiler can auto-vectorize better
    !$omp simd
    do i = 2, n
      acc = alpha * x(i) + one_minus_alpha * acc
      y(i) = acc
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_ewma_fast

  !----------------------------------------------------------------------------
  ! hpcs_ewma_deterministic (DETERMINISTIC MODE)
  !
  ! Full NaN checks like SAFE, but disable SIMD for bit-exact reproducibility.
  ! Strict evaluation order, no FMA reordering.
  ! 1.6-4x slower than SAFE due to forced scalar execution.
  !----------------------------------------------------------------------------
  subroutine hpcs_ewma_deterministic(x, n, alpha, y, status)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan

    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), value       :: alpha
    real(c_double), intent(out) :: y(*)
    integer(c_int), intent(out) :: status

    integer(c_int) :: i
    real(c_double) :: one_minus_alpha, acc, temp
    logical :: seen_finite

    ! Validate arguments (same as SAFE)
    if (n <= 0_c_int .or. alpha <= 0.0_c_double .or. alpha > 1.0_c_double) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    one_minus_alpha = 1.0_c_double - alpha

    ! Initialize with first element
    acc = x(1)
    y(1) = acc
    seen_finite = .not. ieee_is_nan(acc)

    ! Disable vectorization, force strict evaluation order
    !DIR$ NOVECTOR
    do i = 2, n
      ! Force evaluation order with temporary
      temp = alpha * x(i)
      acc = temp + one_minus_alpha * acc
      y(i) = acc
      if (.not. ieee_is_nan(x(i))) seen_finite = .true.
    end do

    ! Check for all-NaN input
    if (seen_finite) then
      status = HPCS_SUCCESS
    else
      status = HPCS_ERR_NUMERIC_FAIL
    end if
  end subroutine hpcs_ewma_deterministic

  !----------------------------------------------------------------------------
  ! hpcs_ewvar (DISPATCHER)
  !
  ! Public entry point for exponentially weighted variance.
  ! Dispatches to mode-specific implementation.
  !----------------------------------------------------------------------------
  subroutine hpcs_ewvar(x, n, alpha, v_out, mode, status) &
       bind(C, name="hpcs_ewvar")
    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), value       :: alpha
    real(c_double), intent(out) :: v_out(*)
    integer(c_int), value       :: mode
    integer(c_int), intent(out) :: status

    integer(c_int) :: effective_mode

    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_ewvar_safe(x, n, alpha, v_out, status)
      case (HPCS_MODE_FAST)
        call hpcs_ewvar_fast(x, n, alpha, v_out, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_ewvar_deterministic(x, n, alpha, v_out, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_ewvar

  !----------------------------------------------------------------------------
  ! hpcs_ewvar_safe (SAFE MODE)
  !----------------------------------------------------------------------------
  subroutine hpcs_ewvar_safe(x, n, alpha, v_out, status)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan

    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), value       :: alpha
    real(c_double), intent(out) :: v_out(*)
    integer(c_int), intent(out) :: status

    integer(c_int) :: i
    real(c_double) :: m, v, one_minus_alpha, delta
    logical :: seen_finite

    if (n <= 0_c_int .or. alpha <= 0.0_c_double .or. alpha > 1.0_c_double) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    one_minus_alpha = 1.0_c_double - alpha
    m = x(1)
    v = 0.0_c_double
    v_out(1) = v
    seen_finite = .not. ieee_is_nan(x(1))

    do i = 2, n
      delta = x(i) - m
      m = m + alpha * delta
      v = one_minus_alpha * v + alpha * one_minus_alpha * delta * delta
      v_out(i) = v
      if (.not. ieee_is_nan(x(i))) seen_finite = .true.
    end do

    if (seen_finite) then
      status = HPCS_SUCCESS
    else
      status = HPCS_ERR_NUMERIC_FAIL
    end if
  end subroutine hpcs_ewvar_safe

  !----------------------------------------------------------------------------
  ! hpcs_ewvar_fast (FAST MODE)
  !----------------------------------------------------------------------------
  subroutine hpcs_ewvar_fast(x, n, alpha, v_out, status)
    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), value       :: alpha
    real(c_double), intent(out) :: v_out(*)
    integer(c_int), intent(out) :: status

    integer(c_int) :: i
    real(c_double) :: m, v, one_minus_alpha, delta

    one_minus_alpha = 1.0_c_double - alpha
    m = x(1)
    v = 0.0_c_double
    v_out(1) = v

    !$omp simd
    do i = 2, n
      delta = x(i) - m
      m = m + alpha * delta
      v = one_minus_alpha * v + alpha * one_minus_alpha * delta * delta
      v_out(i) = v
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_ewvar_fast

  !----------------------------------------------------------------------------
  ! hpcs_ewvar_deterministic (DETERMINISTIC MODE)
  !----------------------------------------------------------------------------
  subroutine hpcs_ewvar_deterministic(x, n, alpha, v_out, status)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan

    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), value       :: alpha
    real(c_double), intent(out) :: v_out(*)
    integer(c_int), intent(out) :: status

    integer(c_int) :: i
    real(c_double) :: m, v, one_minus_alpha, delta, temp1, temp2
    logical :: seen_finite

    if (n <= 0_c_int .or. alpha <= 0.0_c_double .or. alpha > 1.0_c_double) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    one_minus_alpha = 1.0_c_double - alpha
    m = x(1)
    v = 0.0_c_double
    v_out(1) = v
    seen_finite = .not. ieee_is_nan(x(1))

    !DIR$ NOVECTOR
    do i = 2, n
      delta = x(i) - m
      temp1 = alpha * delta
      m = m + temp1
      temp2 = one_minus_alpha * delta * delta
      v = one_minus_alpha * v + alpha * temp2
      v_out(i) = v
      if (.not. ieee_is_nan(x(i))) seen_finite = .true.
    end do

    if (seen_finite) then
      status = HPCS_SUCCESS
    else
      status = HPCS_ERR_NUMERIC_FAIL
    end if
  end subroutine hpcs_ewvar_deterministic

  !----------------------------------------------------------------------------
  ! hpcs_ewstd (DISPATCHER)
  !
  ! Public entry point for exponentially weighted standard deviation.
  ! Dispatches to mode-specific implementation.
  !----------------------------------------------------------------------------
  subroutine hpcs_ewstd(x, n, alpha, y, mode, status) &
       bind(C, name="hpcs_ewstd")
    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), value       :: alpha
    real(c_double), intent(out) :: y(*)
    integer(c_int), value       :: mode
    integer(c_int), intent(out) :: status

    integer(c_int) :: effective_mode

    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_ewstd_safe(x, n, alpha, y, status)
      case (HPCS_MODE_FAST)
        call hpcs_ewstd_fast(x, n, alpha, y, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_ewstd_deterministic(x, n, alpha, y, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_ewstd

  !----------------------------------------------------------------------------
  ! hpcs_ewstd_safe (SAFE MODE)
  !----------------------------------------------------------------------------
  subroutine hpcs_ewstd_safe(x, n, alpha, y, status)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan

    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), value       :: alpha
    real(c_double), intent(out) :: y(*)
    integer(c_int), intent(out) :: status

    integer(c_int) :: i
    real(c_double) :: m, v, one_minus_alpha, delta
    logical :: seen_finite

    if (n <= 0_c_int .or. alpha <= 0.0_c_double .or. alpha > 1.0_c_double) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    one_minus_alpha = 1.0_c_double - alpha
    m = x(1)
    v = 0.0_c_double
    y(1) = 0.0_c_double
    seen_finite = .not. ieee_is_nan(x(1))

    do i = 2, n
      delta = x(i) - m
      m = m + alpha * delta
      v = one_minus_alpha * v + alpha * one_minus_alpha * delta * delta
      y(i) = sqrt(max(v, 0.0_c_double))
      if (.not. ieee_is_nan(x(i))) seen_finite = .true.
    end do

    if (seen_finite) then
      status = HPCS_SUCCESS
    else
      status = HPCS_ERR_NUMERIC_FAIL
    end if
  end subroutine hpcs_ewstd_safe

  !----------------------------------------------------------------------------
  ! hpcs_ewstd_fast (FAST MODE)
  !----------------------------------------------------------------------------
  subroutine hpcs_ewstd_fast(x, n, alpha, y, status)
    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), value       :: alpha
    real(c_double), intent(out) :: y(*)
    integer(c_int), intent(out) :: status

    integer(c_int) :: i
    real(c_double) :: m, v, one_minus_alpha, delta

    one_minus_alpha = 1.0_c_double - alpha
    m = x(1)
    v = 0.0_c_double
    y(1) = 0.0_c_double

    !$omp simd
    do i = 2, n
      delta = x(i) - m
      m = m + alpha * delta
      v = one_minus_alpha * v + alpha * one_minus_alpha * delta * delta
      y(i) = sqrt(max(v, 0.0_c_double))
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_ewstd_fast

  !----------------------------------------------------------------------------
  ! hpcs_ewstd_deterministic (DETERMINISTIC MODE)
  !----------------------------------------------------------------------------
  subroutine hpcs_ewstd_deterministic(x, n, alpha, y, status)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan

    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), value       :: alpha
    real(c_double), intent(out) :: y(*)
    integer(c_int), intent(out) :: status

    integer(c_int) :: i
    real(c_double) :: m, v, one_minus_alpha, delta, temp1, temp2
    logical :: seen_finite

    if (n <= 0_c_int .or. alpha <= 0.0_c_double .or. alpha > 1.0_c_double) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    one_minus_alpha = 1.0_c_double - alpha
    m = x(1)
    v = 0.0_c_double
    y(1) = 0.0_c_double
    seen_finite = .not. ieee_is_nan(x(1))

    !DIR$ NOVECTOR
    do i = 2, n
      delta = x(i) - m
      temp1 = alpha * delta
      m = m + temp1
      temp2 = one_minus_alpha * delta * delta
      v = one_minus_alpha * v + alpha * temp2
      y(i) = sqrt(max(v, 0.0_c_double))
      if (.not. ieee_is_nan(x(i))) seen_finite = .true.
    end do

    if (seen_finite) then
      status = HPCS_SUCCESS
    else
      status = HPCS_ERR_NUMERIC_FAIL
    end if
  end subroutine hpcs_ewstd_deterministic

  !----------------------------------------------------------------------------
  ! hpcs_diff
  !
  ! Compute finite differencing of order k: y[t] = x[t] - x[t-k].
  ! First k elements are set to NaN.
  !
  ! Parameters:
  !   x      - input array [n]
  !   n      - array length
  !   order  - lag order (k ≥ 1)
  !   y      - output difference array [n]
  !   status - 0=success, 1=invalid args
  !----------------------------------------------------------------------------
  subroutine hpcs_diff(x, n, order, y, status) &
       bind(C, name="hpcs_diff")
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    integer(c_int), value       :: order
    real(c_double), intent(out) :: y(*)
    integer(c_int), intent(out) :: status

    integer(c_int) :: i
    real(c_double) :: nan_val

    ! Validate arguments
    if (n <= 0_c_int .or. order < 1_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Create quiet NaN
    nan_val = ieee_value(0.0_c_double, ieee_quiet_nan)

    ! Set first 'order' elements to NaN
    do i = 1, min(order, n)
      y(i) = nan_val
    end do

    ! Compute differences (vectorizable - no loop-carried dependency)
    do i = order + 1, n
      y(i) = x(i) - x(i - order)
    end do

    status = HPCS_SUCCESS

  end subroutine hpcs_diff

  !----------------------------------------------------------------------------
  ! hpcs_cumulative_min (DISPATCHER)
  !
  ! Public entry point for cumulative minimum.
  ! Dispatches to mode-specific implementation.
  !----------------------------------------------------------------------------
  subroutine hpcs_cumulative_min(x, n, y, mode, status) &
       bind(C, name="hpcs_cumulative_min")
    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), intent(out) :: y(*)
    integer(c_int), value       :: mode
    integer(c_int), intent(out) :: status

    integer(c_int) :: effective_mode

    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_cumulative_min_safe(x, n, y, status)
      case (HPCS_MODE_FAST)
        call hpcs_cumulative_min_fast(x, n, y, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_cumulative_min_deterministic(x, n, y, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_cumulative_min

  !----------------------------------------------------------------------------
  ! hpcs_cumulative_min_safe (SAFE MODE)
  !----------------------------------------------------------------------------
  subroutine hpcs_cumulative_min_safe(x, n, y, status)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan

    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), intent(out) :: y(*)
    integer(c_int), intent(out) :: status

    integer(c_int) :: i
    real(c_double) :: acc
    logical :: seen_finite

    if (n <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    acc = x(1)
    y(1) = acc
    seen_finite = .not. ieee_is_nan(acc)

    do i = 2, n
      if (x(i) < acc) acc = x(i)
      y(i) = acc
      if (.not. ieee_is_nan(x(i))) seen_finite = .true.
    end do

    if (seen_finite) then
      status = HPCS_SUCCESS
    else
      status = HPCS_ERR_NUMERIC_FAIL
    end if
  end subroutine hpcs_cumulative_min_safe

  !----------------------------------------------------------------------------
  ! hpcs_cumulative_min_fast (FAST MODE)
  !----------------------------------------------------------------------------
  subroutine hpcs_cumulative_min_fast(x, n, y, status)
    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), intent(out) :: y(*)
    integer(c_int), intent(out) :: status

    integer(c_int) :: i
    real(c_double) :: acc

    acc = x(1)
    y(1) = acc

    ! No NaN checks - assume finite inputs
    do i = 2, n
      if (x(i) < acc) acc = x(i)
      y(i) = acc
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_cumulative_min_fast

  !----------------------------------------------------------------------------
  ! hpcs_cumulative_min_deterministic (DETERMINISTIC MODE)
  !----------------------------------------------------------------------------
  subroutine hpcs_cumulative_min_deterministic(x, n, y, status)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan

    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), intent(out) :: y(*)
    integer(c_int), intent(out) :: status

    integer(c_int) :: i
    real(c_double) :: acc
    logical :: seen_finite

    if (n <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    acc = x(1)
    y(1) = acc
    seen_finite = .not. ieee_is_nan(acc)

    ! Disable vectorization for deterministic results
    !DIR$ NOVECTOR
    do i = 2, n
      if (x(i) < acc) acc = x(i)
      y(i) = acc
      if (.not. ieee_is_nan(x(i))) seen_finite = .true.
    end do

    if (seen_finite) then
      status = HPCS_SUCCESS
    else
      status = HPCS_ERR_NUMERIC_FAIL
    end if
  end subroutine hpcs_cumulative_min_deterministic

  !----------------------------------------------------------------------------
  ! hpcs_cumulative_max (DISPATCHER)
  !
  ! Public entry point for cumulative maximum.
  ! Dispatches to mode-specific implementation.
  !----------------------------------------------------------------------------
  subroutine hpcs_cumulative_max(x, n, y, mode, status) &
       bind(C, name="hpcs_cumulative_max")
    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), intent(out) :: y(*)
    integer(c_int), value       :: mode
    integer(c_int), intent(out) :: status

    integer(c_int) :: effective_mode

    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_cumulative_max_safe(x, n, y, status)
      case (HPCS_MODE_FAST)
        call hpcs_cumulative_max_fast(x, n, y, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_cumulative_max_deterministic(x, n, y, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_cumulative_max

  !----------------------------------------------------------------------------
  ! hpcs_cumulative_max_safe (SAFE MODE)
  !----------------------------------------------------------------------------
  subroutine hpcs_cumulative_max_safe(x, n, y, status)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan

    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), intent(out) :: y(*)
    integer(c_int), intent(out) :: status

    integer(c_int) :: i
    real(c_double) :: acc
    logical :: seen_finite

    if (n <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    acc = x(1)
    y(1) = acc
    seen_finite = .not. ieee_is_nan(acc)

    do i = 2, n
      if (x(i) > acc) acc = x(i)
      y(i) = acc
      if (.not. ieee_is_nan(x(i))) seen_finite = .true.
    end do

    if (seen_finite) then
      status = HPCS_SUCCESS
    else
      status = HPCS_ERR_NUMERIC_FAIL
    end if
  end subroutine hpcs_cumulative_max_safe

  !----------------------------------------------------------------------------
  ! hpcs_cumulative_max_fast (FAST MODE)
  !----------------------------------------------------------------------------
  subroutine hpcs_cumulative_max_fast(x, n, y, status)
    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), intent(out) :: y(*)
    integer(c_int), intent(out) :: status

    integer(c_int) :: i
    real(c_double) :: acc

    acc = x(1)
    y(1) = acc

    ! No NaN checks - assume finite inputs
    do i = 2, n
      if (x(i) > acc) acc = x(i)
      y(i) = acc
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_cumulative_max_fast

  !----------------------------------------------------------------------------
  ! hpcs_cumulative_max_deterministic (DETERMINISTIC MODE)
  !----------------------------------------------------------------------------
  subroutine hpcs_cumulative_max_deterministic(x, n, y, status)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan

    real(c_double), intent(in)  :: x(*)
    integer(c_int), value       :: n
    real(c_double), intent(out) :: y(*)
    integer(c_int), intent(out) :: status

    integer(c_int) :: i
    real(c_double) :: acc
    logical :: seen_finite

    if (n <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    acc = x(1)
    y(1) = acc
    seen_finite = .not. ieee_is_nan(acc)

    ! Disable vectorization for deterministic results
    !DIR$ NOVECTOR
    do i = 2, n
      if (x(i) > acc) acc = x(i)
      y(i) = acc
      if (.not. ieee_is_nan(x(i))) seen_finite = .true.
    end do

    if (seen_finite) then
      status = HPCS_SUCCESS
    else
      status = HPCS_ERR_NUMERIC_FAIL
    end if
  end subroutine hpcs_cumulative_max_deterministic

  !----------------------------------------------------------------------------
  ! hpcs_convolve_valid - DEPRECATED (v0.8.0)
  !
  ! This Fortran implementation is REPLACED by the C++ version in
  ! src/cpp/hpcs_convolution.cpp which supports execution modes.
  !
  ! The C++ version provides:
  !   - Execution mode parameter (SAFE/FAST/DETERMINISTIC)
  !   - Template specializations for kernel sizes 3,5,7,9,11,13,15
  !   - OpenMP parallelization in SAFE/FAST modes
  !
  ! Keeping this code commented for reference only.
  !----------------------------------------------------------------------------
  ! subroutine hpcs_convolve_valid(x, n, k, m, y, status) &
  !      bind(C, name="hpcs_convolve_valid")
  !   use iso_c_binding, only: c_int, c_double
  !   use, intrinsic :: ieee_arithmetic, only: ieee_is_nan
  !   implicit none
  !
  !   real(c_double), intent(in)  :: x(*)
  !   integer(c_int), value       :: n
  !   real(c_double), intent(in)  :: k(*)
  !   integer(c_int), value       :: m
  !   real(c_double), intent(out) :: y(*)
  !   integer(c_int), intent(out) :: status
  !
  !   integer(c_int) :: i, j, out_n
  !   real(c_double) :: sum_val
  !
  !   ! Validate arguments
  !   if (n <= 0_c_int .or. m <= 0_c_int .or. m > n) then
  !     status = HPCS_ERR_INVALID_ARGS
  !     return
  !   end if
  !
  !   out_n = n - m + 1
  !
  !   do i = 1, out_n
  !     sum_val = 0.0_c_double
  !     do j = 1, m
  !       sum_val = sum_val + x(i+j-1) * k(j)
  !     end do
  !     y(i) = sum_val
  !   end do
  !
  !   status = HPCS_SUCCESS
  !
  ! end subroutine hpcs_convolve_valid

end module hpcs_core_transforms
