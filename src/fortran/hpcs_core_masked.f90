! ==============================================================================
! HPC Series Core Library - Masked Operations (v0.4 CPU)
!
! This module provides validity‑aware variants of basic reductions and rolling
! statistics.  A mask array of length n, with non‑zero entries indicating
! valid data and zero entries indicating invalid/masked elements, guides
! these computations.  Masked reductions skip invalid entries.  If all
! entries are masked, the result is NaN and status is set to
! HPCS_ERR_NUMERIC_FAIL (value 2).  Rolling masked operations use prefix
! sums of values and counts for efficiency.  Median and MAD computations
! allocate a buffer containing only valid values and reuse existing 1D
! kernels from hpcs_core_stats.  All routines follow the HPCSeries ABI
! convention: C‑compatible interfaces via bind(C) and status returned via
! an explicit argument.
! ============================================================================

module hpcs_core_masked
  use iso_c_binding,  only: c_int, c_double
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  use hpcs_constants
  use hpcs_core_stats, only: hpcs_median, hpcs_mad
  use hpcs_core_execution_mode, only: hpcs_get_execution_mode_internal
  implicit none
  private
  public :: hpcs_reduce_sum_masked
  public :: hpcs_reduce_mean_masked
  public :: hpcs_reduce_variance_masked
  public :: hpcs_rolling_mean_masked
  public :: hpcs_median_masked
  public :: hpcs_mad_masked
  public :: hpcs_rolling_median_masked

contains

  !--------------------------------------------------------------------------
  ! hpcs_reduce_sum_masked
  !
  ! Compute the sum of the elements of x where mask(i) != 0.  The mask
  ! array is interpreted as an integer array of the same length as x; a
  ! non‑zero value indicates that x(i) is valid.  NaN values in x propagate
  ! into the result.  If no elements are valid (mask==0 or x is NaN for
  ! all entries), the output is NaN and status=HPCS_ERR_NUMERIC_FAIL.
  ! Invalid arguments (n<0) produce status=HPCS_ERR_INVALID_ARGS.
  !--------------------------------------------------------------------------
  ! hpcs_reduce_sum_masked (v0.8.0 dispatcher with execution mode support)
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_sum_masked(x, mask, n, out, mode, status) &
       bind(C, name="hpcs_reduce_sum_masked")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int), intent(in)  :: mask(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    ! Resolve execution mode
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    ! Dispatch to appropriate implementation
    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_reduce_sum_masked_safe(x, mask, n, out, status)
      case (HPCS_MODE_FAST)
        call hpcs_reduce_sum_masked_fast(x, mask, n, out, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_reduce_sum_masked_deterministic(x, mask, n, out, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_reduce_sum_masked

  !--------------------------------------------------------------------------
  ! SAFE mode: Full NaN detection and validation
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_sum_masked_safe(x, mask, n, out, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int), intent(in)  :: mask(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, i, count
    real(c_double) :: acc
    logical        :: has_nan

    n_eff = n
    if (n_eff < 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if
    if (n_eff == 0_c_int) then
      out = 0.0_c_double
      status = HPCS_SUCCESS
      return
    end if

    acc = 0.0_c_double
    count = 0_c_int
    has_nan = .false.
    do i = 1_c_int, n_eff
      if (mask(i) /= 0_c_int) then
        if (x(i) /= x(i)) then  ! NaN check
          has_nan = .true.
        else
          acc = acc + x(i)
          count = count + 1_c_int
        end if
      end if
    end do

    if (count == 0_c_int) then
      out = ieee_value(0.0_c_double, ieee_quiet_nan)
      status = HPCS_ERR_NUMERIC_FAIL
    else if (has_nan) then
      out = ieee_value(0.0_c_double, ieee_quiet_nan)
      status = HPCS_ERR_NUMERIC_FAIL
    else
      out = acc
      status = HPCS_SUCCESS
    end if
  end subroutine hpcs_reduce_sum_masked_safe

  !--------------------------------------------------------------------------
  ! FAST mode: No validation, maximum speed
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_sum_masked_fast(x, mask, n, out, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int), intent(in)  :: mask(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, count
    real(c_double) :: acc

    ! No input validation - assume valid
    acc = 0.0_c_double
    count = 0_c_int

    ! No NaN checks - direct accumulation
    do i = 1_c_int, n
      if (mask(i) /= 0_c_int) then
        acc = acc + x(i)
        count = count + 1_c_int
      end if
    end do

    if (count > 0_c_int) then
      out = acc
      status = HPCS_SUCCESS
    else
      out = ieee_value(0.0_c_double, ieee_quiet_nan)
      status = HPCS_ERR_NUMERIC_FAIL
    end if
  end subroutine hpcs_reduce_sum_masked_fast

  !--------------------------------------------------------------------------
  ! DETERMINISTIC mode: Full validation, no SIMD (delegates to SAFE)
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_sum_masked_deterministic(x, mask, n, out, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int), intent(in)  :: mask(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    ! Already sequential, same as SAFE mode
    call hpcs_reduce_sum_masked_safe(x, mask, n, out, status)
  end subroutine hpcs_reduce_sum_masked_deterministic

  !--------------------------------------------------------------------------
  ! hpcs_reduce_mean_masked (v0.8.0 dispatcher with execution mode support)
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_mean_masked(x, mask, n, out, mode, status) &
       bind(C, name="hpcs_reduce_mean_masked")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int), intent(in)  :: mask(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    ! Resolve execution mode
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    ! Dispatch to appropriate implementation
    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_reduce_mean_masked_safe(x, mask, n, out, status)
      case (HPCS_MODE_FAST)
        call hpcs_reduce_mean_masked_fast(x, mask, n, out, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_reduce_mean_masked_deterministic(x, mask, n, out, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_reduce_mean_masked

  !--------------------------------------------------------------------------
  ! SAFE mode: Full NaN detection and validation
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_mean_masked_safe(x, mask, n, out, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int), intent(in)  :: mask(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, i, count
    real(c_double) :: acc
    logical        :: has_nan

    n_eff = n
    if (n_eff < 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if
    if (n_eff == 0_c_int) then
      out = 0.0_c_double
      status = HPCS_SUCCESS
      return
    end if

    acc = 0.0_c_double
    count = 0_c_int
    has_nan = .false.
    do i = 1_c_int, n_eff
      if (mask(i) /= 0_c_int) then
        if (x(i) /= x(i)) then  ! NaN check
          has_nan = .true.
        else
          acc = acc + x(i)
          count = count + 1_c_int
        end if
      end if
    end do

    if (count == 0_c_int .or. has_nan) then
      out = ieee_value(0.0_c_double, ieee_quiet_nan)
      status = HPCS_ERR_NUMERIC_FAIL
    else
      out = acc / real(count, kind=c_double)
      status = HPCS_SUCCESS
    end if
  end subroutine hpcs_reduce_mean_masked_safe

  !--------------------------------------------------------------------------
  ! FAST mode: No validation, maximum speed
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_mean_masked_fast(x, mask, n, out, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int), intent(in)  :: mask(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, count
    real(c_double) :: acc

    ! No input validation - assume valid
    acc = 0.0_c_double
    count = 0_c_int

    ! No NaN checks - direct accumulation
    do i = 1_c_int, n
      if (mask(i) /= 0_c_int) then
        acc = acc + x(i)
        count = count + 1_c_int
      end if
    end do

    if (count > 0_c_int) then
      out = acc / real(count, kind=c_double)
      status = HPCS_SUCCESS
    else
      out = ieee_value(0.0_c_double, ieee_quiet_nan)
      status = HPCS_ERR_NUMERIC_FAIL
    end if
  end subroutine hpcs_reduce_mean_masked_fast

  !--------------------------------------------------------------------------
  ! DETERMINISTIC mode: Full validation, no SIMD (delegates to SAFE)
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_mean_masked_deterministic(x, mask, n, out, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int), intent(in)  :: mask(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    ! Already sequential, same as SAFE mode
    call hpcs_reduce_mean_masked_safe(x, mask, n, out, status)
  end subroutine hpcs_reduce_mean_masked_deterministic

  !--------------------------------------------------------------------------
  ! hpcs_reduce_variance_masked (v0.8.0 dispatcher with execution mode support)
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_variance_masked(x, mask, n, out, mode, status) &
       bind(C, name="hpcs_reduce_variance_masked")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int), intent(in)  :: mask(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    ! Resolve execution mode
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    ! Dispatch to appropriate implementation
    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_reduce_variance_masked_safe(x, mask, n, out, status)
      case (HPCS_MODE_FAST)
        call hpcs_reduce_variance_masked_fast(x, mask, n, out, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_reduce_variance_masked_deterministic(x, mask, n, out, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_reduce_variance_masked

  !--------------------------------------------------------------------------
  ! SAFE mode: Full NaN detection with Welford's algorithm
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_variance_masked_safe(x, mask, n, out, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int), intent(in)  :: mask(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, i, count
    real(c_double) :: mean, M2, delta, delta2
    logical        :: has_nan

    n_eff = n
    if (n_eff < 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if
    if (n_eff == 0_c_int) then
      out = 0.0_c_double
      status = HPCS_SUCCESS
      return
    end if

    count = 0_c_int
    mean  = 0.0_c_double
    M2    = 0.0_c_double
    has_nan = .false.

    ! Welford's algorithm with NaN detection
    do i = 1_c_int, n_eff
      if (mask(i) /= 0_c_int) then
        if (x(i) /= x(i)) then  ! NaN check
          has_nan = .true.
        else
          count = count + 1_c_int
          delta = x(i) - mean
          mean  = mean + delta / real(count, kind=c_double)
          delta2 = x(i) - mean
          M2 = M2 + delta * delta2
        end if
      end if
    end do

    if (count < 2_c_int .or. has_nan) then
      out = ieee_value(0.0_c_double, ieee_quiet_nan)
      status = HPCS_ERR_NUMERIC_FAIL
    else
      out = M2 / real(count - 1_c_int, kind=c_double)
      status = HPCS_SUCCESS
    end if
  end subroutine hpcs_reduce_variance_masked_safe

  !--------------------------------------------------------------------------
  ! FAST mode: No NaN checks, Welford's algorithm (still sequential)
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_variance_masked_fast(x, mask, n, out, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int), intent(in)  :: mask(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, count
    real(c_double) :: mean, M2, delta, delta2

    ! No input validation - assume valid
    count = 0_c_int
    mean  = 0.0_c_double
    M2    = 0.0_c_double

    ! Welford's algorithm without NaN checks
    do i = 1_c_int, n
      if (mask(i) /= 0_c_int) then
        count = count + 1_c_int
        delta = x(i) - mean
        mean  = mean + delta / real(count, kind=c_double)
        delta2 = x(i) - mean
        M2 = M2 + delta * delta2
      end if
    end do

    if (count >= 2_c_int) then
      out = M2 / real(count - 1_c_int, kind=c_double)
      status = HPCS_SUCCESS
    else
      out = ieee_value(0.0_c_double, ieee_quiet_nan)
      status = HPCS_ERR_NUMERIC_FAIL
    end if
  end subroutine hpcs_reduce_variance_masked_fast

  !--------------------------------------------------------------------------
  ! DETERMINISTIC mode: Same as SAFE (Welford's is already sequential)
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_variance_masked_deterministic(x, mask, n, out, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int), intent(in)  :: mask(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    ! Welford's algorithm is inherently sequential - same as SAFE
    call hpcs_reduce_variance_masked_safe(x, mask, n, out, status)
  end subroutine hpcs_reduce_variance_masked_deterministic

  !--------------------------------------------------------------------------
  ! hpcs_rolling_mean_masked
  !
  ! Compute a masked rolling mean with window length window on a 1D series x
  ! of length n.  The mask array marks which elements are valid.  At each
  ! position i, the result y(i) is the mean of the valid values in the
  ! window [i-window+1 .. i] (inclusive).  If no values in the window are
  ! valid, y(i) is NaN.  This implementation uses prefix sums for values
  ! and counts for O(n) complexity.  Invalid arguments (n<=0, window<=0
  ! or window>n) produce status=1.
  !--------------------------------------------------------------------------
  subroutine hpcs_rolling_mean_masked(x, mask, n, window, y, status) &
       bind(C, name="hpcs_rolling_mean_masked")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int), intent(in)  :: mask(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, w_eff
    integer(c_int) :: i
    real(c_double), allocatable :: p_val(:)
    integer(c_int), allocatable :: p_cnt(:)
    integer(c_int) :: start_idx
    real(c_double) :: sum_val
    integer(c_int) :: cnt

    n_eff = n
    w_eff = window
    if (n_eff <= 0_c_int .or. w_eff <= 0_c_int .or. w_eff > n_eff) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Allocate prefix arrays with one extra element to handle window offset
    allocate(p_val(0:n_eff))
    allocate(p_cnt(0:n_eff))
    p_val(0) = 0.0_c_double
    p_cnt(0) = 0_c_int
    do i = 1_c_int, n_eff
      if (mask(i) /= 0_c_int .and. x(i) == x(i)) then
        p_val(i) = p_val(i-1_c_int) + x(i)
        p_cnt(i) = p_cnt(i-1_c_int) + 1_c_int
      else
        p_val(i) = p_val(i-1_c_int)
        p_cnt(i) = p_cnt(i-1_c_int)
      end if
    end do
    ! Compute rolling means
    do i = 1_c_int, n_eff
      if (i > w_eff) then
        start_idx = i - w_eff
        sum_val = p_val(i) - p_val(start_idx)
        cnt     = p_cnt(i) - p_cnt(start_idx)
      else
        sum_val = p_val(i)
        cnt     = p_cnt(i)
      end if
      if (cnt > 0_c_int) then
        y(i) = sum_val / real(cnt, kind=c_double)
      else
        y(i) = ieee_value(0.0_c_double, ieee_quiet_nan)
      end if
    end do
    deallocate(p_val)
    deallocate(p_cnt)
    status = HPCS_SUCCESS
  end subroutine hpcs_rolling_mean_masked

  !--------------------------------------------------------------------------
  ! hpcs_median_masked
  !
  ! Compute the median of the valid entries in x as indicated by mask.  A
  ! temporary buffer is allocated to store the valid values, which is
  ! passed to hpcs_median.  If there are no valid values, the output is
  ! NaN and status=2.  NaN values in x among valid entries propagate to
  ! the result (hpcs_median will return NaN).  Invalid arguments (n<0)
  ! yield status=1.
  !--------------------------------------------------------------------------
  subroutine hpcs_median_masked(x, mask, n, out, mode, status) &
       bind(C, name="hpcs_median_masked")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int), intent(in)  :: mask(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff
    integer(c_int) :: i, cnt, st
    real(c_double), allocatable :: buf(:)
    integer(c_int) :: j
    integer(c_int) :: effective_mode

    n_eff = n
    if (n_eff < 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if
    if (n_eff == 0_c_int) then
      out = 0.0_c_double
      status = HPCS_SUCCESS
      return
    end if

    ! Resolve execution mode
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    ! Count valid entries
    cnt = 0_c_int
    do i = 1_c_int, n_eff
      if (mask(i) /= 0_c_int .and. x(i) == x(i)) cnt = cnt + 1_c_int
    end do
    if (cnt == 0_c_int) then
      out = ieee_value(0.0_c_double, ieee_quiet_nan)
      status = HPCS_ERR_NUMERIC_FAIL
      return
    end if
    allocate(buf(cnt))
    j = 1_c_int
    do i = 1_c_int, n_eff
      if (mask(i) /= 0_c_int .and. x(i) == x(i)) then
        buf(j) = x(i)
        j = j + 1_c_int
      end if
    end do
    call hpcs_median(buf, cnt, out, effective_mode, st)
    deallocate(buf)
    if (st /= HPCS_SUCCESS) then
      status = st
    else
      status = HPCS_SUCCESS
    end if
  end subroutine hpcs_median_masked

  !--------------------------------------------------------------------------
  ! hpcs_mad_masked
  !
  ! Compute the median absolute deviation (MAD) of the valid entries in x
  ! indicated by mask.  A buffer of the valid values is passed to the
  ! existing hpcs_mad routine.  If no valid entries exist, the result is
  ! NaN and status=2.  Degenerate distributions (MAD≈0) propagate
  ! status=2 from hpcs_mad.  Invalid arguments (n<0) yield status=1.
  !--------------------------------------------------------------------------
  subroutine hpcs_mad_masked(x, mask, n, out, mode, status) &
       bind(C, name="hpcs_mad_masked")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int), intent(in)  :: mask(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff
    integer(c_int) :: i, cnt, st
    real(c_double), allocatable :: buf(:)
    integer(c_int) :: j
    integer(c_int) :: effective_mode

    n_eff = n
    if (n_eff < 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if
    if (n_eff == 0_c_int) then
      out = 0.0_c_double
      status = HPCS_SUCCESS
      return
    end if

    ! Resolve execution mode
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    cnt = 0_c_int
    do i = 1_c_int, n_eff
      if (mask(i) /= 0_c_int .and. x(i) == x(i)) cnt = cnt + 1_c_int
    end do
    if (cnt == 0_c_int) then
      out = ieee_value(0.0_c_double, ieee_quiet_nan)
      status = HPCS_ERR_NUMERIC_FAIL
      return
    end if
    allocate(buf(cnt))
    j = 1_c_int
    do i = 1_c_int, n_eff
      if (mask(i) /= 0_c_int .and. x(i) == x(i)) then
        buf(j) = x(i)
        j = j + 1_c_int
      end if
    end do
    call hpcs_mad(buf, cnt, out, effective_mode, st)
    deallocate(buf)
    status = st
  end subroutine hpcs_mad_masked

  !--------------------------------------------------------------------------
  ! hpcs_rolling_median_masked
  !
  ! Compute a masked rolling median on a 1D series x of length n.  For each
  ! position i, the result y(i) is the median of the valid elements in the
  ! window [i-window+1 .. i].  If there are no valid elements in the window
  ! the output is NaN and status=2.  A naive implementation is used: for
  ! each position, a temporary buffer of valid values in the current window
  ! is constructed and the median is computed via hpcs_median.  Complexity
  ! is O(n*window).  Invalid arguments (n<=0 or window<=0 or window>n)
  ! produce status=1.
  !--------------------------------------------------------------------------
  subroutine hpcs_rolling_median_masked(x, mask, n, window, y, status) &
       bind(C, name="hpcs_rolling_median_masked")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int), intent(in)  :: mask(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, w_eff
    integer(c_int) :: i, j, start_idx
    integer(c_int) :: cnt, st
    real(c_double), allocatable :: buf(:)
    integer(c_int) :: k
    real(c_double) :: val
    integer(c_int) :: max_status

    n_eff = n
    w_eff = window
    if (n_eff <= 0_c_int .or. w_eff <= 0_c_int .or. w_eff > n_eff) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if
    max_status = HPCS_SUCCESS

    ! Allocate buffer once with maximum size (w_eff) instead of per iteration
    ! This fixes the critical performance issue: O(1) allocation vs O(n) allocations
    allocate(buf(w_eff))

    do i = 1_c_int, n_eff
      ! Determine window bounds [start_idx .. i]
      start_idx = i - w_eff + 1_c_int
      if (start_idx < 1_c_int) start_idx = 1_c_int
      ! Count valid elements in the window
      cnt = 0_c_int
      do j = start_idx, i
        if (mask(j) /= 0_c_int .and. x(j) == x(j)) cnt = cnt + 1_c_int
      end do
      if (cnt == 0_c_int) then
        y(i) = ieee_value(0.0_c_double, ieee_quiet_nan)
        if (HPCS_ERR_NUMERIC_FAIL > max_status) max_status = HPCS_ERR_NUMERIC_FAIL
        cycle
      end if
      ! Reuse the pre-allocated buffer - only use first cnt elements
      k = 1_c_int
      do j = start_idx, i
        if (mask(j) /= 0_c_int .and. x(j) == x(j)) then
          buf(k) = x(j)
          k = k + 1_c_int
        end if
      end do
      call hpcs_median(buf, cnt, val, HPCS_MODE_SAFE, st)
      y(i) = val
      if (st > max_status) max_status = st
    end do

    deallocate(buf)
    status = max_status
  end subroutine hpcs_rolling_median_masked

end module hpcs_core_masked