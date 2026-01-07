! ==============================================================================
! HPC Series Core Library - Reductions Module
! Simple 1D reductions and grouped reductions.
!
! Implemented kernels:
!   - hpcs_reduce_sum
!   - hpcs_reduce_min
!   - hpcs_reduce_max
!   - hpcs_reduce_mean       (v0.2)
!   - hpcs_reduce_variance   (v0.2)
!   - hpcs_reduce_std        (v0.2)
!   - hpcs_group_reduce_sum
!   - hpcs_group_reduce_mean
!   - hpcs_group_reduce_variance (v0.2)
!
! All routines:
!   - use ISO_C_BINDING with bind(C)
!   - return status via an explicit integer(c_int) argument
! ==============================================================================

module hpcs_core_reductions
  use iso_c_binding,  only: c_int, c_double
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  use hpcs_constants
  implicit none
  public

contains

  !--------------------------------------------------------------------
  ! hpcs_reduce_sum (v0.8.0 - with execution mode support)
  !
  ! out = sum(x(1:n))
  !
  ! Parameters:
  !   mode : Execution mode (SAFE=0, FAST=1, DETERMINISTIC=2, USE_GLOBAL=-1)
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or invalid mode
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_sum(x, n, out, mode, status) &
       bind(C, name="hpcs_reduce_sum")
    use iso_c_binding, only: c_int, c_double
    use hpcs_core_execution_mode, only: hpcs_get_execution_mode_internal
    implicit none

    real(c_double), intent(in)  :: x(*)      ! length n
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    ! Resolve mode (use global if MODE_USE_GLOBAL)
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    ! Dispatch to mode-specific implementation
    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_reduce_sum_safe(x, n, out, status)
      case (HPCS_MODE_FAST)
        call hpcs_reduce_sum_fast(x, n, out, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_reduce_sum_deterministic(x, n, out, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
        return
    end select
  end subroutine hpcs_reduce_sum

  !--------------------------------------------------------------------
  ! hpcs_reduce_sum_safe - SAFE mode implementation
  !
  ! Full validation + NaN detection + OpenMP parallelization
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_sum_safe(x, n, out, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    real(c_double) :: acc

    ! 1. Full input validation
    if (n <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! 2. NaN detection
    do i = 1_c_int, n
      if (ieee_is_nan(x(i))) then
        out = ieee_value(0.0_c_double, ieee_quiet_nan)
        status = HPCS_SUCCESS
        return
      end if
    end do

    ! 3. Computation with OpenMP
    acc = 0.0_c_double
    !$omp parallel do reduction(+:acc) if(n >= 10000)
    do i = 1_c_int, n
      acc = acc + x(i)
    end do
    !$omp end parallel do

    out = acc
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_sum_safe

  !--------------------------------------------------------------------
  ! hpcs_reduce_sum_fast - FAST mode implementation
  !
  ! No validation + No NaN checks + OpenMP parallelization
  ! ~1.3-1.6x faster than SAFE for large arrays
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_sum_fast(x, n, out, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    real(c_double) :: acc

    ! No validation, no NaN checks (assume n > 0, finite inputs)

    acc = 0.0_c_double
    !$omp parallel do reduction(+:acc) if(n >= 10000)
    do i = 1_c_int, n
      acc = acc + x(i)
    end do
    !$omp end parallel do

    out = acc
    status = HPCS_SUCCESS  ! Always success in FAST mode
  end subroutine hpcs_reduce_sum_fast

  !--------------------------------------------------------------------
  ! hpcs_reduce_sum_deterministic - DETERMINISTIC mode implementation
  !
  ! Full validation + NaN detection + NO OpenMP (bit-exact reproducibility)
  ! ~1.5-2.5x slower than SAFE (no parallelization)
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_sum_deterministic(x, n, out, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    real(c_double) :: acc

    ! 1. Full input validation (same as SAFE)
    if (n <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! 2. NaN detection (same as SAFE)
    do i = 1_c_int, n
      if (ieee_is_nan(x(i))) then
        out = ieee_value(0.0_c_double, ieee_quiet_nan)
        status = HPCS_SUCCESS
        return
      end if
    end do

    ! 3. Sequential computation (NO OpenMP, bit-exact reproducibility)
    !DIR$ NOVECTOR
    acc = 0.0_c_double
    do i = 1_c_int, n
      acc = acc + x(i)  ! Explicit evaluation order
    end do

    out = acc
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_sum_deterministic

  !--------------------------------------------------------------------
  ! hpcs_reduce_min (v0.8.0 - with execution mode support)
  !
  ! out = min(x(1:n))
  !
  ! Parameters:
  !   mode : Execution mode (SAFE=0, FAST=1, DETERMINISTIC=2, USE_GLOBAL=-1)
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or invalid mode
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_min(x, n, out, mode, status) &
       bind(C, name="hpcs_reduce_min")
    use iso_c_binding, only: c_int, c_double
    use hpcs_core_execution_mode, only: hpcs_get_execution_mode_internal
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_reduce_min_safe(x, n, out, status)
      case (HPCS_MODE_FAST)
        call hpcs_reduce_min_fast(x, n, out, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_reduce_min_deterministic(x, n, out, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_reduce_min

  subroutine hpcs_reduce_min_safe(x, n, out, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    real(c_double) :: minval

    if (n <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! NaN detection
    do i = 1_c_int, n
      if (ieee_is_nan(x(i))) then
        out = ieee_value(0.0_c_double, ieee_quiet_nan)
        status = HPCS_SUCCESS
        return
      end if
    end do

    minval = x(1_c_int)
    do i = 2_c_int, n
      if (x(i) < minval) minval = x(i)
    end do

    out = minval
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_min_safe

  subroutine hpcs_reduce_min_fast(x, n, out, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    real(c_double) :: minval

    minval = x(1_c_int)
    do i = 2_c_int, n
      if (x(i) < minval) minval = x(i)
    end do

    out = minval
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_min_fast

  subroutine hpcs_reduce_min_deterministic(x, n, out, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    real(c_double) :: minval

    if (n <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! NaN detection
    do i = 1_c_int, n
      if (ieee_is_nan(x(i))) then
        out = ieee_value(0.0_c_double, ieee_quiet_nan)
        status = HPCS_SUCCESS
        return
      end if
    end do

    !DIR$ NOVECTOR
    minval = x(1_c_int)
    do i = 2_c_int, n
      if (x(i) < minval) minval = x(i)
    end do

    out = minval
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_min_deterministic

  !--------------------------------------------------------------------
  ! hpcs_reduce_max (v0.8.0 - with execution mode support)
  !
  ! out = max(x(1:n))
  !
  ! Parameters:
  !   mode : Execution mode (SAFE=0, FAST=1, DETERMINISTIC=2, USE_GLOBAL=-1)
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or invalid mode
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_max(x, n, out, mode, status) &
       bind(C, name="hpcs_reduce_max")
    use iso_c_binding, only: c_int, c_double
    use hpcs_core_execution_mode, only: hpcs_get_execution_mode_internal
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_reduce_max_safe(x, n, out, status)
      case (HPCS_MODE_FAST)
        call hpcs_reduce_max_fast(x, n, out, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_reduce_max_deterministic(x, n, out, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_reduce_max

  subroutine hpcs_reduce_max_safe(x, n, out, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    real(c_double) :: maxval

    if (n <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! NaN detection
    do i = 1_c_int, n
      if (ieee_is_nan(x(i))) then
        out = ieee_value(0.0_c_double, ieee_quiet_nan)
        status = HPCS_SUCCESS
        return
      end if
    end do

    maxval = x(1_c_int)
    do i = 2_c_int, n
      if (x(i) > maxval) maxval = x(i)
    end do

    out = maxval
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_max_safe

  subroutine hpcs_reduce_max_fast(x, n, out, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    real(c_double) :: maxval

    maxval = x(1_c_int)
    do i = 2_c_int, n
      if (x(i) > maxval) maxval = x(i)
    end do

    out = maxval
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_max_fast

  subroutine hpcs_reduce_max_deterministic(x, n, out, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    real(c_double) :: maxval

    if (n <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! NaN detection
    do i = 1_c_int, n
      if (ieee_is_nan(x(i))) then
        out = ieee_value(0.0_c_double, ieee_quiet_nan)
        status = HPCS_SUCCESS
        return
      end if
    end do

    !DIR$ NOVECTOR
    maxval = x(1_c_int)
    do i = 2_c_int, n
      if (x(i) > maxval) maxval = x(i)
    end do

    out = maxval
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_max_deterministic

  !--------------------------------------------------------------------
  ! hpcs_reduce_mean (v0.8.0 - with execution mode support)
  !
  ! out = mean(x(1:n)) = sum(x) / n
  !
  ! Parameters:
  !   mode : Execution mode (SAFE=0, FAST=1, DETERMINISTIC=2, USE_GLOBAL=-1)
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or invalid mode
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_mean(x, n, out, mode, status) &
       bind(C, name="hpcs_reduce_mean")
    use iso_c_binding, only: c_int, c_double
    use hpcs_core_execution_mode, only: hpcs_get_execution_mode_internal
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_reduce_mean_safe(x, n, out, status)
      case (HPCS_MODE_FAST)
        call hpcs_reduce_mean_fast(x, n, out, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_reduce_mean_deterministic(x, n, out, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_reduce_mean

  subroutine hpcs_reduce_mean_safe(x, n, out, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    real(c_double) :: sum_val

    if (n <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! NaN detection
    do i = 1_c_int, n
      if (ieee_is_nan(x(i))) then
        out = ieee_value(0.0_c_double, ieee_quiet_nan)
        status = HPCS_SUCCESS
        return
      end if
    end do

    sum_val = 0.0_c_double
    !$omp parallel do reduction(+:sum_val) if(n >= 10000)
    do i = 1_c_int, n
      sum_val = sum_val + x(i)
    end do
    !$omp end parallel do

    out = sum_val / real(n, kind=c_double)
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_mean_safe

  subroutine hpcs_reduce_mean_fast(x, n, out, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    real(c_double) :: sum_val

    sum_val = 0.0_c_double
    !$omp parallel do reduction(+:sum_val) if(n >= 10000)
    do i = 1_c_int, n
      sum_val = sum_val + x(i)
    end do
    !$omp end parallel do

    out = sum_val / real(n, kind=c_double)
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_mean_fast

  subroutine hpcs_reduce_mean_deterministic(x, n, out, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    real(c_double) :: sum_val

    if (n <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! NaN detection
    do i = 1_c_int, n
      if (ieee_is_nan(x(i))) then
        out = ieee_value(0.0_c_double, ieee_quiet_nan)
        status = HPCS_SUCCESS
        return
      end if
    end do

    !DIR$ NOVECTOR
    sum_val = 0.0_c_double
    do i = 1_c_int, n
      sum_val = sum_val + x(i)
    end do

    out = sum_val / real(n, kind=c_double)
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_mean_deterministic

  !--------------------------------------------------------------------
  ! hpcs_reduce_variance (v0.8.0 - with execution mode support)
  !
  ! out = variance(x(1:n)) = sum((x - mean)^2) / n  (population variance)
  !
  ! Uses Welford's online algorithm for numerical stability
  !
  ! Parameters:
  !   mode : Execution mode (SAFE=0, FAST=1, DETERMINISTIC=2, USE_GLOBAL=-1)
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or invalid mode
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_variance(x, n, out, mode, status) &
       bind(C, name="hpcs_reduce_variance")
    use iso_c_binding, only: c_int, c_double
    use hpcs_core_execution_mode, only: hpcs_get_execution_mode_internal
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_reduce_variance_safe(x, n, out, status)
      case (HPCS_MODE_FAST)
        call hpcs_reduce_variance_fast(x, n, out, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_reduce_variance_deterministic(x, n, out, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_reduce_variance

  subroutine hpcs_reduce_variance_safe(x, n, out, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    real(c_double) :: mean_running, s_running, delta, delta2

    if (n <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    if (n == 1_c_int) then
      out = 0.0_c_double
      status = HPCS_SUCCESS
      return
    end if

    ! NaN detection
    do i = 1_c_int, n
      if (ieee_is_nan(x(i))) then
        out = ieee_value(0.0_c_double, ieee_quiet_nan)
        status = HPCS_SUCCESS
        return
      end if
    end do

    ! Welford's algorithm (inherently sequential, no OpenMP)
    mean_running = 0.0_c_double
    s_running = 0.0_c_double

    do i = 1_c_int, n
      delta = x(i) - mean_running
      mean_running = mean_running + delta / real(i, kind=c_double)
      delta2 = x(i) - mean_running
      s_running = s_running + delta * delta2
    end do

    out = s_running / real(n, kind=c_double)
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_variance_safe

  subroutine hpcs_reduce_variance_fast(x, n, out, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    real(c_double) :: mean_running, s_running, delta, delta2

    if (n == 1_c_int) then
      out = 0.0_c_double
      status = HPCS_SUCCESS
      return
    end if

    ! Welford's algorithm (no validation)
    mean_running = 0.0_c_double
    s_running = 0.0_c_double

    do i = 1_c_int, n
      delta = x(i) - mean_running
      mean_running = mean_running + delta / real(i, kind=c_double)
      delta2 = x(i) - mean_running
      s_running = s_running + delta * delta2
    end do

    out = s_running / real(n, kind=c_double)
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_variance_fast

  subroutine hpcs_reduce_variance_deterministic(x, n, out, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    real(c_double) :: mean_running, s_running, delta, delta2

    if (n <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    if (n == 1_c_int) then
      out = 0.0_c_double
      status = HPCS_SUCCESS
      return
    end if

    ! NaN detection
    do i = 1_c_int, n
      if (ieee_is_nan(x(i))) then
        out = ieee_value(0.0_c_double, ieee_quiet_nan)
        status = HPCS_SUCCESS
        return
      end if
    end do

    ! Welford's algorithm with explicit ordering (DETERMINISTIC)
    !DIR$ NOVECTOR
    mean_running = 0.0_c_double
    s_running = 0.0_c_double

    do i = 1_c_int, n
      delta = x(i) - mean_running
      mean_running = mean_running + delta / real(i, kind=c_double)
      delta2 = x(i) - mean_running
      s_running = s_running + delta * delta2
    end do

    out = s_running / real(n, kind=c_double)
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_variance_deterministic

  !--------------------------------------------------------------------
  ! hpcs_reduce_std (v0.8.0 - with execution mode support)
  !
  ! out = sqrt(variance(x(1:n)))  (population standard deviation)
  !
  ! Uses Welford's algorithm via hpcs_reduce_variance, then takes sqrt.
  !
  ! Parameters:
  !   mode : Execution mode (SAFE=0, FAST=1, DETERMINISTIC=2, USE_GLOBAL=-1)
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or invalid mode
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_std(x, n, out, mode, status) &
       bind(C, name="hpcs_reduce_std")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    real(c_double) :: variance

    ! Compute variance first (passing mode through)
    call hpcs_reduce_variance(x, n, variance, mode, status)
    if (status /= HPCS_SUCCESS) then
      return
    end if

    ! Take square root
    out = sqrt(variance)
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_std

  !--------------------------------------------------------------------
  ! hpcs_group_reduce_sum (v0.8.0 - with execution mode support)
  !
  ! Grouped sum:
  !   x(1:n), group_ids(1:n) in [0, n_groups-1]
  !   y(0:n_groups-1) stored as y(1:n_groups) in Fortran
  !
  ! Invalid group IDs (<0 or >= n_groups) are ignored.
  !
  ! Parameters:
  !   mode : Execution mode (SAFE=0, FAST=1, DETERMINISTIC=2, USE_GLOBAL=-1)
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or n_groups <= 0 or invalid mode
  !--------------------------------------------------------------------
  subroutine hpcs_group_reduce_sum(x, n, group_ids, n_groups, y, mode, status) &
       bind(C, name="hpcs_group_reduce_sum")
    use iso_c_binding, only: c_int, c_double
    use hpcs_core_execution_mode, only: hpcs_get_execution_mode_internal
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_group_reduce_sum_safe(x, n, group_ids, n_groups, y, status)
      case (HPCS_MODE_FAST)
        call hpcs_group_reduce_sum_fast(x, n, group_ids, n_groups, y, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_group_reduce_sum_deterministic(x, n, group_ids, n_groups, y, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_group_reduce_sum

  subroutine hpcs_group_reduce_sum_safe(x, n, group_ids, n_groups, y, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, g
    real(c_double) :: nan_val

    if (n <= 0_c_int .or. n_groups <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! NaN detection
    do i = 1_c_int, n
      if (ieee_is_nan(x(i))) then
        nan_val = ieee_value(0.0_c_double, ieee_quiet_nan)
        do g = 1_c_int, n_groups
          y(g) = nan_val
        end do
        status = HPCS_SUCCESS
        return
      end if
    end do

    ! Initialize sums to zero
    do g = 1_c_int, n_groups
      y(g) = 0.0_c_double
    end do

    ! Accumulate (group IDs validation)
    do i = 1_c_int, n
      g = group_ids(i)
      if (g < 0_c_int .or. g >= n_groups) cycle  ! Skip invalid group IDs
      y(g + 1_c_int) = y(g + 1_c_int) + x(i)
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_group_reduce_sum_safe

  subroutine hpcs_group_reduce_sum_fast(x, n, group_ids, n_groups, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, g

    ! Initialize sums to zero
    do g = 1_c_int, n_groups
      y(g) = 0.0_c_double
    end do

    ! Accumulate (no validation, assume valid group IDs)
    do i = 1_c_int, n
      g = group_ids(i)
      y(g + 1_c_int) = y(g + 1_c_int) + x(i)
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_group_reduce_sum_fast

  subroutine hpcs_group_reduce_sum_deterministic(x, n, group_ids, n_groups, y, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, g
    real(c_double) :: nan_val

    if (n <= 0_c_int .or. n_groups <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! NaN detection
    do i = 1_c_int, n
      if (ieee_is_nan(x(i))) then
        nan_val = ieee_value(0.0_c_double, ieee_quiet_nan)
        do g = 1_c_int, n_groups
          y(g) = nan_val
        end do
        status = HPCS_SUCCESS
        return
      end if
    end do

    ! Initialize sums to zero
    !DIR$ NOVECTOR
    do g = 1_c_int, n_groups
      y(g) = 0.0_c_double
    end do

    ! Accumulate (deterministic, sequential)
    !DIR$ NOVECTOR
    do i = 1_c_int, n
      g = group_ids(i)
      if (g < 0_c_int .or. g >= n_groups) cycle
      y(g + 1_c_int) = y(g + 1_c_int) + x(i)
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_group_reduce_sum_deterministic

  !--------------------------------------------------------------------
  ! hpcs_group_reduce_mean (v0.8.0 - with execution mode support)
  !
  ! Grouped mean:
  !   mean_k = sum_{i in group k} x(i) / count_k
  !
  ! Groups with zero count -> NaN.
  !
  ! Parameters:
  !   mode : Execution mode (SAFE=0, FAST=1, DETERMINISTIC=2, USE_GLOBAL=-1)
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or n_groups <= 0 or invalid mode
  !   HPCS_ERR_NUMERIC_FAIL : allocation failure
  !--------------------------------------------------------------------
  subroutine hpcs_group_reduce_mean(x, n, group_ids, n_groups, y, mode, status) &
       bind(C, name="hpcs_group_reduce_mean")
    use iso_c_binding, only: c_int, c_double
    use hpcs_core_execution_mode, only: hpcs_get_execution_mode_internal
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_group_reduce_mean_safe(x, n, group_ids, n_groups, y, status)
      case (HPCS_MODE_FAST)
        call hpcs_group_reduce_mean_fast(x, n, group_ids, n_groups, y, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_group_reduce_mean_deterministic(x, n, group_ids, n_groups, y, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_group_reduce_mean

  subroutine hpcs_group_reduce_mean_safe(x, n, group_ids, n_groups, y, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, g, istat
    real(c_double), allocatable :: group_sum(:)
    integer(c_int), allocatable :: group_count(:)
    real(c_double) :: nan_val

    if (n <= 0_c_int .or. n_groups <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! NaN detection
    do i = 1_c_int, n
      if (ieee_is_nan(x(i))) then
        nan_val = ieee_value(0.0_c_double, ieee_quiet_nan)
        do g = 1_c_int, n_groups
          y(g) = nan_val
        end do
        status = HPCS_SUCCESS
        return
      end if
    end do

    allocate(group_sum(n_groups), group_count(n_groups), stat=istat)
    if (istat /= 0_c_int) then
      status = HPCS_ERR_NUMERIC_FAIL
      return
    end if

    group_sum = 0.0_c_double
    group_count = 0_c_int

    do i = 1_c_int, n
      g = group_ids(i)
      if (g < 0_c_int .or. g >= n_groups) cycle
      group_sum(g + 1_c_int) = group_sum(g + 1_c_int) + x(i)
      group_count(g + 1_c_int) = group_count(g + 1_c_int) + 1_c_int
    end do

    nan_val = ieee_value(0.0_c_double, ieee_quiet_nan)

    do g = 1_c_int, n_groups
      if (group_count(g) > 0_c_int) then
        y(g) = group_sum(g) / real(group_count(g), kind=c_double)
      else
        y(g) = nan_val
      end if
    end do

    deallocate(group_sum, group_count, stat=istat)
    status = HPCS_SUCCESS
  end subroutine hpcs_group_reduce_mean_safe

  subroutine hpcs_group_reduce_mean_fast(x, n, group_ids, n_groups, y, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, g, istat
    real(c_double), allocatable :: group_sum(:)
    integer(c_int), allocatable :: group_count(:)
    real(c_double) :: nan_val

    allocate(group_sum(n_groups), group_count(n_groups), stat=istat)
    if (istat /= 0_c_int) then
      status = HPCS_ERR_NUMERIC_FAIL
      return
    end if

    group_sum = 0.0_c_double
    group_count = 0_c_int

    do i = 1_c_int, n
      g = group_ids(i) + 1_c_int  ! Assume valid, no bounds check
      group_sum(g) = group_sum(g) + x(i)
      group_count(g) = group_count(g) + 1_c_int
    end do

    nan_val = ieee_value(0.0_c_double, ieee_quiet_nan)

    do g = 1_c_int, n_groups
      if (group_count(g) > 0_c_int) then
        y(g) = group_sum(g) / real(group_count(g), kind=c_double)
      else
        y(g) = nan_val
      end if
    end do

    deallocate(group_sum, group_count, stat=istat)
    status = HPCS_SUCCESS
  end subroutine hpcs_group_reduce_mean_fast

  subroutine hpcs_group_reduce_mean_deterministic(x, n, group_ids, n_groups, y, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, g, istat
    real(c_double), allocatable :: group_sum(:)
    integer(c_int), allocatable :: group_count(:)
    real(c_double) :: nan_val

    if (n <= 0_c_int .or. n_groups <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! NaN detection
    do i = 1_c_int, n
      if (ieee_is_nan(x(i))) then
        nan_val = ieee_value(0.0_c_double, ieee_quiet_nan)
        do g = 1_c_int, n_groups
          y(g) = nan_val
        end do
        status = HPCS_SUCCESS
        return
      end if
    end do

    allocate(group_sum(n_groups), group_count(n_groups), stat=istat)
    if (istat /= 0_c_int) then
      status = HPCS_ERR_NUMERIC_FAIL
      return
    end if

    group_sum = 0.0_c_double
    group_count = 0_c_int

    !DIR$ NOVECTOR
    do i = 1_c_int, n
      g = group_ids(i)
      if (g < 0_c_int .or. g >= n_groups) cycle
      group_sum(g + 1_c_int) = group_sum(g + 1_c_int) + x(i)
      group_count(g + 1_c_int) = group_count(g + 1_c_int) + 1_c_int
    end do

    nan_val = ieee_value(0.0_c_double, ieee_quiet_nan)

    do g = 1_c_int, n_groups
      if (group_count(g) > 0_c_int) then
        y(g) = group_sum(g) / real(group_count(g), kind=c_double)
      else
        y(g) = nan_val
      end if
    end do

    deallocate(group_sum, group_count, stat=istat)
    status = HPCS_SUCCESS
  end subroutine hpcs_group_reduce_mean_deterministic

  !--------------------------------------------------------------------
  ! hpcs_group_reduce_variance (v0.8.0 - with execution mode support)
  !
  ! Grouped variance (population):
  !   variance_k = sum_{i in group k} (x_i - mean_k)^2 / count_k
  !
  ! Uses Welford's algorithm per group for numerical stability.
  ! Groups with zero count -> NaN.
  !
  ! Parameters:
  !   mode : Execution mode (SAFE=0, FAST=1, DETERMINISTIC=2, USE_GLOBAL=-1)
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or n_groups <= 0 or invalid mode
  !   HPCS_ERR_NUMERIC_FAIL : allocation failure
  !--------------------------------------------------------------------
  subroutine hpcs_group_reduce_variance(x, n, group_ids, n_groups, y, mode, status) &
       bind(C, name="hpcs_group_reduce_variance")
    use iso_c_binding, only: c_int, c_double
    use hpcs_core_execution_mode, only: hpcs_get_execution_mode_internal
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_group_reduce_variance_safe(x, n, group_ids, n_groups, y, status)
      case (HPCS_MODE_FAST)
        call hpcs_group_reduce_variance_fast(x, n, group_ids, n_groups, y, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_group_reduce_variance_deterministic(x, n, group_ids, n_groups, y, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_group_reduce_variance

  subroutine hpcs_group_reduce_variance_safe(x, n, group_ids, n_groups, y, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, g, istat, count_k
    real(c_double), allocatable :: group_mean(:), group_m2(:)
    integer(c_int), allocatable :: group_count(:)
    real(c_double) :: nan_val, delta, delta2

    if (n <= 0_c_int .or. n_groups <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! NaN detection
    do i = 1_c_int, n
      if (ieee_is_nan(x(i))) then
        nan_val = ieee_value(0.0_c_double, ieee_quiet_nan)
        do g = 1_c_int, n_groups
          y(g) = nan_val
        end do
        status = HPCS_SUCCESS
        return
      end if
    end do

    allocate(group_mean(n_groups), group_m2(n_groups), group_count(n_groups), stat=istat)
    if (istat /= 0_c_int) then
      status = HPCS_ERR_NUMERIC_FAIL
      return
    end if

    group_mean = 0.0_c_double
    group_m2 = 0.0_c_double
    group_count = 0_c_int

    ! Welford's algorithm per group
    do i = 1_c_int, n
      g = group_ids(i)
      if (g < 0_c_int .or. g >= n_groups) cycle
      g = g + 1_c_int  ! Convert to 1-based

      group_count(g) = group_count(g) + 1_c_int
      count_k = group_count(g)

      delta = x(i) - group_mean(g)
      group_mean(g) = group_mean(g) + delta / real(count_k, kind=c_double)
      delta2 = x(i) - group_mean(g)
      group_m2(g) = group_m2(g) + delta * delta2
    end do

    nan_val = ieee_value(0.0_c_double, ieee_quiet_nan)

    do g = 1_c_int, n_groups
      if (group_count(g) > 0_c_int) then
        y(g) = group_m2(g) / real(group_count(g), kind=c_double)
      else
        y(g) = nan_val
      end if
    end do

    deallocate(group_mean, group_m2, group_count, stat=istat)
    status = HPCS_SUCCESS
  end subroutine hpcs_group_reduce_variance_safe

  subroutine hpcs_group_reduce_variance_fast(x, n, group_ids, n_groups, y, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, g, istat, count_k
    real(c_double), allocatable :: group_mean(:), group_m2(:)
    integer(c_int), allocatable :: group_count(:)
    real(c_double) :: nan_val, delta, delta2

    allocate(group_mean(n_groups), group_m2(n_groups), group_count(n_groups), stat=istat)
    if (istat /= 0_c_int) then
      status = HPCS_ERR_NUMERIC_FAIL
      return
    end if

    group_mean = 0.0_c_double
    group_m2 = 0.0_c_double
    group_count = 0_c_int

    ! Welford's algorithm (no validation)
    do i = 1_c_int, n
      g = group_ids(i) + 1_c_int  ! Assume valid

      group_count(g) = group_count(g) + 1_c_int
      count_k = group_count(g)

      delta = x(i) - group_mean(g)
      group_mean(g) = group_mean(g) + delta / real(count_k, kind=c_double)
      delta2 = x(i) - group_mean(g)
      group_m2(g) = group_m2(g) + delta * delta2
    end do

    nan_val = ieee_value(0.0_c_double, ieee_quiet_nan)

    do g = 1_c_int, n_groups
      if (group_count(g) > 0_c_int) then
        y(g) = group_m2(g) / real(group_count(g), kind=c_double)
      else
        y(g) = nan_val
      end if
    end do

    deallocate(group_mean, group_m2, group_count, stat=istat)
    status = HPCS_SUCCESS
  end subroutine hpcs_group_reduce_variance_fast

  subroutine hpcs_group_reduce_variance_deterministic(x, n, group_ids, n_groups, y, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, g, istat, count_k
    real(c_double), allocatable :: group_mean(:), group_m2(:)
    integer(c_int), allocatable :: group_count(:)
    real(c_double) :: nan_val, delta, delta2

    if (n <= 0_c_int .or. n_groups <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! NaN detection
    do i = 1_c_int, n
      if (ieee_is_nan(x(i))) then
        nan_val = ieee_value(0.0_c_double, ieee_quiet_nan)
        do g = 1_c_int, n_groups
          y(g) = nan_val
        end do
        status = HPCS_SUCCESS
        return
      end if
    end do

    allocate(group_mean(n_groups), group_m2(n_groups), group_count(n_groups), stat=istat)
    if (istat /= 0_c_int) then
      status = HPCS_ERR_NUMERIC_FAIL
      return
    end if

    group_mean = 0.0_c_double
    group_m2 = 0.0_c_double
    group_count = 0_c_int

    ! Welford's algorithm with explicit ordering (DETERMINISTIC)
    !DIR$ NOVECTOR
    do i = 1_c_int, n
      g = group_ids(i)
      if (g < 0_c_int .or. g >= n_groups) cycle
      g = g + 1_c_int

      group_count(g) = group_count(g) + 1_c_int
      count_k = group_count(g)

      delta = x(i) - group_mean(g)
      group_mean(g) = group_mean(g) + delta / real(count_k, kind=c_double)
      delta2 = x(i) - group_mean(g)
      group_m2(g) = group_m2(g) + delta * delta2
    end do

    nan_val = ieee_value(0.0_c_double, ieee_quiet_nan)

    do g = 1_c_int, n_groups
      if (group_count(g) > 0_c_int) then
        y(g) = group_m2(g) / real(group_count(g), kind=c_double)
      else
        y(g) = nan_val
      end if
    end do

    deallocate(group_mean, group_m2, group_count, stat=istat)
    status = HPCS_SUCCESS
  end subroutine hpcs_group_reduce_variance_deterministic

end module hpcs_core_reductions
