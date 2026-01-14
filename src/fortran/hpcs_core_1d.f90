! ==============================================================================
! HPC Series Core Library - 1D Operations Module
! Rolling window operations and statistical transformations
!
! Implemented kernels:
!   - hpcs_rolling_sum:      O(n) sliding window summation
!   - hpcs_rolling_mean:     O(n) sliding window mean
!   - hpcs_rolling_variance: O(n) sliding window variance (v0.2)
!   - hpcs_rolling_std:      O(n) sliding window std deviation (v0.2)
!   - hpcs_zscore:           Z-score normalization using Welford's algorithm
!
! All routines use C-compatible interfaces via iso_c_binding and return
! status via an explicit integer(c_int) argument.
! ==============================================================================

module hpcs_core_1d
  use iso_c_binding,  only: c_int, c_double
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  use hpcs_constants
  use hpcs_core_execution_mode, only: hpcs_get_execution_mode_internal
  implicit none
  public

contains

  !--------------------------------------------------------------------
  ! Rolling sum
  !
  ! y(i) = sum of last window elements up to i (truncated near start)
  !
  ! Arguments (C view):
  !   x   : const double*  (length >= n)
  !   n   : int
  !   w   : int  (window)
  !   y   : double*        (length >= n)
  !   st  : int*           (status)
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or window <= 0
  !--------------------------------------------------------------------
  !--------------------------------------------------------------------
  ! Rolling sum (v0.8.0 - with execution mode support)
  !
  ! y(i) = sum of elements in window ending at position i
  !
  ! Parameters:
  !   mode: Execution mode (SAFE/FAST/DETERMINISTIC/USE_GLOBAL)
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or window <= 0 or invalid mode
  !--------------------------------------------------------------------
  subroutine hpcs_rolling_sum(x, n, window, y, mode, status) &
       bind(C, name="hpcs_rolling_sum")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)      ! length n
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)      ! length n
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    ! Resolve mode
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    ! Dispatch to appropriate implementation
    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_rolling_sum_safe(x, n, window, y, status)
      case (HPCS_MODE_FAST)
        call hpcs_rolling_sum_fast(x, n, window, y, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_rolling_sum_deterministic(x, n, window, y, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_rolling_sum

  ! SAFE mode: Full validation with NaN checking
  subroutine hpcs_rolling_sum_safe(x, n, window, y, status)
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, j, n_eff, w_eff, k, valid_count
    real(c_double) :: sum

    n_eff = n
    w_eff = window

    if (n_eff <= 0_c_int .or. w_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    ! Process each position
    do i = 1_c_int, n_eff
       sum = 0.0_c_double
       valid_count = 0_c_int

       ! Determine window bounds
       if (i < w_eff) then
          k = i
       else
          k = w_eff
       end if

       ! Sum valid (non-NaN) elements in window
       do j = i - k + 1_c_int, i
          if (x(j) == x(j)) then  ! NaN check
             sum = sum + x(j)
             valid_count = valid_count + 1_c_int
          end if
       end do

       ! Return sum or NaN if all values were NaN
       if (valid_count > 0_c_int) then
          y(i) = sum
       else
          y(i) = ieee_value(0.0_c_double, ieee_quiet_nan)
       end if
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_rolling_sum_safe

  ! FAST mode: No validation, maximum speed with sliding window
  subroutine hpcs_rolling_sum_fast(x, n, window, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff, w_eff
    real(c_double) :: sum

    n_eff = n
    w_eff = window

    sum = 0.0_c_double

    ! Sliding window optimization: add new element, subtract old
    do i = 1_c_int, n_eff
       sum = sum + x(i)    ! Add new element

       if (i > w_eff) then
          sum = sum - x(i - w_eff)  ! Subtract element leaving window
       end if

       y(i) = sum
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_rolling_sum_fast

  ! DETERMINISTIC mode: Same as SAFE (no OpenMP in original)
  subroutine hpcs_rolling_sum_deterministic(x, n, window, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    ! Delegate to SAFE mode
    call hpcs_rolling_sum_safe(x, n, window, y, status)
  end subroutine hpcs_rolling_sum_deterministic

  !--------------------------------------------------------------------
  ! Rolling mean (v0.8.0 - with execution mode support)
  !
  ! y(i) = rolling_sum(i) / min(i, window)
  !
  ! Parameters:
  !   mode: Execution mode (SAFE/FAST/DETERMINISTIC/USE_GLOBAL)
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or window <= 0 or invalid mode
  !--------------------------------------------------------------------
  subroutine hpcs_rolling_mean(x, n, window, y, mode, status) &
       bind(C, name="hpcs_rolling_mean")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)       ! length n
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)       ! length n
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    ! Resolve mode
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    ! Dispatch to appropriate implementation
    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_rolling_mean_safe(x, n, window, y, status)
      case (HPCS_MODE_FAST)
        call hpcs_rolling_mean_fast(x, n, window, y, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_rolling_mean_deterministic(x, n, window, y, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_rolling_mean

  ! SAFE mode: Full validation with NaN checking
  subroutine hpcs_rolling_mean_safe(x, n, window, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, j, n_eff, w_eff, k, valid_count
    real(c_double) :: sum

    n_eff = n
    w_eff = window

    if (n_eff <= 0_c_int .or. w_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    ! Process each position
    do i = 1_c_int, n_eff
       sum = 0.0_c_double
       valid_count = 0_c_int

       ! Determine window bounds
       if (i < w_eff) then
          k = i
       else
          k = w_eff
       end if

       ! Sum valid (non-NaN) elements in window
       do j = i - k + 1_c_int, i
          if (x(j) == x(j)) then  ! NaN check
             sum = sum + x(j)
             valid_count = valid_count + 1_c_int
          end if
       end do

       ! Compute mean or NaN if all values were NaN
       if (valid_count > 0_c_int) then
          y(i) = sum / real(valid_count, kind=c_double)
       else
          y(i) = ieee_value(0.0_c_double, ieee_quiet_nan)
       end if
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_rolling_mean_safe

  ! FAST mode: No validation, maximum speed
  subroutine hpcs_rolling_mean_fast(x, n, window, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff, w_eff, k
    real(c_double) :: sum

    n_eff = n
    w_eff = window

    sum = 0.0_c_double

    do i = 1_c_int, n_eff
       sum = sum + x(i)
       if (i > w_eff) then
          sum = sum - x(i - w_eff)
       end if

       if (i < w_eff) then
          k = i
       else
          k = w_eff
       end if

       y(i) = sum / real(k, kind=c_double)
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_rolling_mean_fast

  ! DETERMINISTIC mode: Same as SAFE (no OpenMP in original)
  subroutine hpcs_rolling_mean_deterministic(x, n, window, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    ! Same as SAFE mode since no parallelization
    call hpcs_rolling_mean_safe(x, n, window, y, status)
  end subroutine hpcs_rolling_mean_deterministic

  !--------------------------------------------------------------------
  ! Rolling variance (v0.8.0 - with execution mode support)
  !
  ! Computes rolling window population variance using the formula:
  !   variance = E[X²] - (E[X])²
  !
  ! Maintains rolling sum and rolling sum-of-squares for O(n) complexity.
  !
  ! Parameters:
  !   mode: Execution mode (SAFE/FAST/DETERMINISTIC/USE_GLOBAL)
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or window <= 0 or invalid mode
  !--------------------------------------------------------------------
  subroutine hpcs_rolling_variance(x, n, window, y, mode, status) &
       bind(C, name="hpcs_rolling_variance")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    ! Resolve mode
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    ! Dispatch to appropriate implementation
    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_rolling_variance_safe(x, n, window, y, status)
      case (HPCS_MODE_FAST)
        call hpcs_rolling_variance_fast(x, n, window, y, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_rolling_variance_deterministic(x, n, window, y, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_rolling_variance

  ! SAFE mode: Full validation with NaN checking
  subroutine hpcs_rolling_variance_safe(x, n, window, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, j, n_eff, w_eff, k, valid_count
    real(c_double) :: sum, sum_sq, mean, mean_sq, var

    n_eff = n
    w_eff = window

    if (n_eff <= 0_c_int .or. w_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    ! Process each position
    do i = 1_c_int, n_eff
       sum = 0.0_c_double
       sum_sq = 0.0_c_double
       valid_count = 0_c_int

       ! Determine window bounds
       if (i < w_eff) then
          k = i
       else
          k = w_eff
       end if

       ! Sum valid (non-NaN) elements in window
       do j = i - k + 1_c_int, i
          if (x(j) == x(j)) then  ! NaN check
             sum = sum + x(j)
             sum_sq = sum_sq + x(j) * x(j)
             valid_count = valid_count + 1_c_int
          end if
       end do

       ! Compute variance or NaN if all values were NaN
       if (valid_count > 0_c_int) then
          mean = sum / real(valid_count, kind=c_double)
          mean_sq = sum_sq / real(valid_count, kind=c_double)
          var = mean_sq - mean * mean
          if (var < 0.0_c_double) var = 0.0_c_double
          y(i) = var
       else
          y(i) = ieee_value(0.0_c_double, ieee_quiet_nan)
       end if
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_rolling_variance_safe

  ! FAST mode: No validation, maximum speed
  subroutine hpcs_rolling_variance_fast(x, n, window, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff, w_eff, k
    real(c_double) :: sum, sum_sq, mean, mean_sq, var

    n_eff = n
    w_eff = window

    sum    = 0.0_c_double
    sum_sq = 0.0_c_double

    do i = 1_c_int, n_eff
       ! Add new element
       sum    = sum    + x(i)
       sum_sq = sum_sq + x(i) * x(i)

       ! Remove element leaving window
       if (i > w_eff) then
          sum    = sum    - x(i - w_eff)
          sum_sq = sum_sq - x(i - w_eff) * x(i - w_eff)
       end if

       ! Determine current window size
       if (i < w_eff) then
          k = i
       else
          k = w_eff
       end if

       ! Compute variance: Var(X) = E[X²] - (E[X])²
       mean     = sum / real(k, kind=c_double)
       mean_sq  = sum_sq / real(k, kind=c_double)
       var      = mean_sq - mean * mean

       ! Guard against numerical errors (variance should be >= 0)
       if (var < 0.0_c_double) var = 0.0_c_double

       y(i) = var
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_rolling_variance_fast

  ! DETERMINISTIC mode: Same as SAFE (no OpenMP in original)
  subroutine hpcs_rolling_variance_deterministic(x, n, window, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    ! Same as SAFE mode since no parallelization
    call hpcs_rolling_variance_safe(x, n, window, y, status)
  end subroutine hpcs_rolling_variance_deterministic

  !--------------------------------------------------------------------
  ! Rolling standard deviation (v0.8.0 - with execution mode support)
  !
  ! Computes rolling window standard deviation as sqrt(rolling_variance)
  !
  ! Parameters:
  !   mode: Execution mode (SAFE/FAST/DETERMINISTIC/USE_GLOBAL)
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or window <= 0 or invalid mode
  !--------------------------------------------------------------------
  subroutine hpcs_rolling_std(x, n, window, y, mode, status) &
       bind(C, name="hpcs_rolling_std")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    ! Resolve mode
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    ! Dispatch to appropriate implementation
    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_rolling_std_safe(x, n, window, y, status)
      case (HPCS_MODE_FAST)
        call hpcs_rolling_std_fast(x, n, window, y, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_rolling_std_deterministic(x, n, window, y, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_rolling_std

  ! SAFE mode: Call rolling_variance_safe + sqrt
  subroutine hpcs_rolling_std_safe(x, n, window, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff

    n_eff = n

    ! Compute rolling variance first (SAFE mode)
    call hpcs_rolling_variance_safe(x, n, window, y, status)
    if (status /= HPCS_SUCCESS) then
       return
    end if

    ! Take square root of each variance value
    do i = 1_c_int, n_eff
       y(i) = sqrt(y(i))
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_rolling_std_safe

  ! FAST mode: Call rolling_variance_fast + sqrt
  subroutine hpcs_rolling_std_fast(x, n, window, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff

    n_eff = n

    ! Compute rolling variance first (FAST mode)
    call hpcs_rolling_variance_fast(x, n, window, y, status)
    if (status /= HPCS_SUCCESS) then
       return
    end if

    ! Take square root of each variance value
    do i = 1_c_int, n_eff
       y(i) = sqrt(y(i))
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_rolling_std_fast

  ! DETERMINISTIC mode: Same as SAFE (no OpenMP in original)
  subroutine hpcs_rolling_std_deterministic(x, n, window, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    ! Same as SAFE mode since no parallelization
    call hpcs_rolling_std_safe(x, n, window, y, status)
  end subroutine hpcs_rolling_std_deterministic

  !--------------------------------------------------------------------
  ! Z-score transform
  !
  ! Uses two-pass Welford-style algorithm (serial).
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0
  !   HPCS_ERR_NUMERIC_FAIL : stddev == 0 -> y set to 0
  !--------------------------------------------------------------------
  subroutine hpcs_zscore(x, n, y, status) &
       bind(C, name="hpcs_zscore")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)      ! length n
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: y(*)      ! length n
    integer(c_int),  intent(out):: status

    integer(c_int)  :: i, n_eff
    real(c_double)  :: mean, M, S, variance, std
    real(c_double)  :: oldM

    n_eff = n

    if (n_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    ! First pass: Welford's method for mean and variance
    M = 0.0_c_double
    S = 0.0_c_double

    do i = 1_c_int, n_eff
       oldM = M
       M = M + (x(i) - M) / real(i, kind=c_double)
       S = S + (x(i) - M) * (x(i) - oldM)
    end do

    mean = M
    variance = S / real(n_eff, kind=c_double)
    if (variance < 0.0_c_double) variance = 0.0_c_double
    std = sqrt(variance)

    if (std == 0.0_c_double) then
       ! All values identical (or numerically zero variance)
       do i = 1_c_int, n_eff
          y(i) = 0.0_c_double
       end do
       status = HPCS_ERR_NUMERIC_FAIL
       return
    end if

    ! Second pass: z-scores
    do i = 1_c_int, n_eff
       y(i) = (x(i) - mean) / std
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_zscore

end module hpcs_core_1d
