! ==============================================================================
! HPC Series Core Library - 1D Operations Module
! Rolling window operations and statistical transformations
!
! Implemented kernels:
!   - hpcs_rolling_sum:  O(n) sliding window summation
!   - hpcs_rolling_mean: O(n) sliding window mean
!   - hpcs_zscore:       Z-score normalization using Welford's algorithm
!
! All functions use C-compatible interfaces via iso_c_binding
! ==============================================================================

module hpcs_core_1d
  use iso_c_binding, only: c_int, c_double
  use hpcs_constants
  implicit none

contains

  !--------------------------------------------------------------------
  ! Rolling sum
  ! y(i) = sum of last window elements up to i (truncated near start)
  !--------------------------------------------------------------------
  subroutine hpcs_rolling_sum(x, n, window, y, status) &
       bind(C, name="hpcs_rolling_sum")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)      ! length n
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)      ! length n
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    integer(c_int) :: n_eff, w_eff
    real(c_double) :: sum

    n_eff = n
    w_eff = window

    if (n_eff <= 0_c_int .or. w_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    sum = 0.0_c_double

    do i = 1_c_int, n_eff
       sum = sum + x(i)    ! add new element

       if (i > w_eff) then
          sum = sum - x(i - w_eff)  ! subtract element leaving the window
       end if

       y(i) = sum
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_rolling_sum

  !--------------------------------------------------------------------
  ! Rolling mean
  ! y(i) = rolling_sum(i) / min(i, window)
  !--------------------------------------------------------------------
  subroutine hpcs_rolling_mean(x, n, window, y, status) &
       bind(C, name="hpcs_rolling_mean")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)       ! length n
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)       ! length n
    integer(c_int),  intent(out):: status

    integer(c_int) :: i
    integer(c_int) :: n_eff, w_eff, k
    real(c_double) :: sum

    n_eff = n
    w_eff = window

    if (n_eff <= 0_c_int .or. w_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

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
  end subroutine hpcs_rolling_mean

  !--------------------------------------------------------------------
  ! Z-score transform
  ! Uses two-pass Welford-style algorithm (serial)
  ! status = 0: success
  ! status = 2: zero stddev (numeric failure) – y set to 0
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

    if (n_eff < 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (n_eff == 0_c_int) then
       ! Empty input: nothing to do, consider it "success"
       status = HPCS_SUCCESS
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
