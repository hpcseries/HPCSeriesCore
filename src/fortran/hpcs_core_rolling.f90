! ==============================================================================
! HPC Series Core Library - Rolling Robust Statistics Module (v0.3)
!
! This module implements sliding window (rolling) robust statistics over
! contiguous 1-D arrays.  Functions compute the median and median absolute
! deviation (MAD) within each window of a specified length.  The outputs
! contain NaN values for indices before the window is fully defined.
!
! Kernels implemented:
!   - hpcs_rolling_median
!   - hpcs_rolling_mad
!   - hpcs_rolling_anomalies (v0.3+)
!
! The algorithms here follow the v0.3 specification but use a simple copy
! approach for each window: for every window position, a temporary buffer is
! filled with the window contents and the median (and MAD) is computed via
! the hpcs_median routine.  Although this is O(n*window) complexity, it
! satisfies the functional requirements.  A more efficient heap-based
! implementation may be introduced in future versions.
!
! All routines use ISO_C_BINDING with bind(C) and report status codes via
! integer(c_int).  Loops are flat and ready for OpenMP parallelisation.
! ============================================================================

module hpcs_core_rolling
  use iso_c_binding, only: c_int, c_double
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  use hpcs_constants
  use hpcs_core_stats, only: hpcs_median
  implicit none
  private
  public :: hpcs_rolling_median
  public :: hpcs_rolling_mad
  public :: hpcs_rolling_anomalies

contains

  !--------------------------------------------------------------------------
  ! hpcs_rolling_median
  !
  ! Compute the median within a sliding window of length window across x.
  ! The output y has length n.  For indices i < window the output is set to
  ! NaN (undefined).  A copy-based method is used: each window segment is
  ! copied into a temporary buffer and the median is computed via hpcs_median.
  ! Invalid arguments (n<=0, window<=0 or window>n) trigger status
  ! HPCS_ERR_INVALID_ARGS.
  !
  ! Arguments (C view):
  !   x      - const double*    input array of length n
  !   n      - int             number of elements
  !   window - int             window length
  !   y      - double*         output array of length n
  !   status - int*            output status code
  !
  !--------------------------------------------------------------------------
  subroutine hpcs_rolling_median(x, n, window, y, status) &
       bind(C, name="hpcs_rolling_median")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int)             :: n_eff, w_eff
    integer(c_int)             :: i, j, start
    real(c_double), allocatable :: buf(:)
    integer(c_int)             :: st

    n_eff = n
    w_eff = window
    ! Validate arguments
    if (n_eff <= 0_c_int .or. w_eff <= 0_c_int .or. w_eff > n_eff) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Allocate buffer for window contents
    allocate(buf(w_eff))

    ! Initialise the first (window-1) outputs to NaN
    do i = 1_c_int, w_eff - 1_c_int
      y(i) = ieee_value(0.0_c_double, ieee_quiet_nan)
    end do

    ! Compute median for each full window
    do i = w_eff, n_eff
      start = i - w_eff + 1_c_int
      ! copy current window into buffer
      do j = 1_c_int, w_eff
        buf(j) = x(start + j - 1_c_int)
      end do
      ! compute median of buffer
      call hpcs_median(buf, w_eff, y(i), st)
      ! If hpcs_median returns invalid args it should not happen here
    end do

    deallocate(buf)
    status = HPCS_SUCCESS
  end subroutine hpcs_rolling_median

  !--------------------------------------------------------------------------
  ! hpcs_rolling_mad
  !
  ! Compute the median absolute deviation within a sliding window.  For
  ! i < window the output y(i) is NaN.  For i >= window, y(i) is the MAD of
  ! x(i-window+1:i).  The algorithm copies each window into a temporary
  ! buffer, computes its median, computes absolute deviations and then
  ! computes the median of deviations.  Degenerate windows (MAD ≈ 0) are
  ! recorded as y(i) = 0.  Invalid arguments (n<=0, window<=0 or window>n)
  ! produce status = HPCS_ERR_INVALID_ARGS.
  !
  ! Arguments (C view):
  !   x      - const double*    input array of length n
  !   n      - int             number of elements
  !   window - int             window length
  !   y      - double*         output array of length n
  !   status - int*            output status code
  !
  !--------------------------------------------------------------------------
  subroutine hpcs_rolling_mad(x, n, window, y, status) &
       bind(C, name="hpcs_rolling_mad")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int)             :: n_eff, w_eff
    integer(c_int)             :: i, j, start
    real(c_double), allocatable :: buf(:), dev(:)
    real(c_double)             :: med_local, mad_local
    integer(c_int)             :: st

    n_eff = n
    w_eff = window
    ! Validate arguments
    if (n_eff <= 0_c_int .or. w_eff <= 0_c_int .or. w_eff > n_eff) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    allocate(buf(w_eff))
    allocate(dev(w_eff))

    ! Initialise undefined outputs to NaN for positions before the first full window
    do i = 1_c_int, w_eff - 1_c_int
      y(i) = ieee_value(0.0_c_double, ieee_quiet_nan)
    end do

    ! Loop over each window
    do i = w_eff, n_eff
      start = i - w_eff + 1_c_int
      ! Copy window values
      do j = 1_c_int, w_eff
        buf(j) = x(start + j - 1_c_int)
      end do
      ! Compute median of window
      call hpcs_median(buf, w_eff, med_local, st)
      if (st /= HPCS_SUCCESS) then
        ! Should not happen for valid window sizes; mark as NaN
        y(i) = ieee_value(0.0_c_double, ieee_quiet_nan)
        cycle
      end if
      if (med_local /= med_local) then
        ! median is NaN -> propagate NaN to this position
        y(i) = med_local
        cycle
      end if
      ! Compute absolute deviations from median
      do j = 1_c_int, w_eff
        dev(j) = abs(buf(j) - med_local)
      end do
      ! Compute median of deviations
      call hpcs_median(dev, w_eff, mad_local, st)
      if (st /= HPCS_SUCCESS) then
        y(i) = ieee_value(0.0_c_double, ieee_quiet_nan)
      else
        ! Check degeneracy: if mad_local is approximately zero, set to zero
        if (mad_local < 1.0e-12_c_double) then
          y(i) = 0.0_c_double
        else
          y(i) = mad_local
        end if
      end if
    end do

    deallocate(buf)
    deallocate(dev)
    status = HPCS_SUCCESS
  end subroutine hpcs_rolling_mad

  !--------------------------------------------------------------------------
  ! hpcs_rolling_anomalies
  !
  ! Detect anomalies in time series using a rolling window approach with
  ! robust statistics (median and MAD). For each position i, computes
  ! robust z-score based on the median and MAD of the window ending at i,
  ! then flags if |z_robust| > threshold.
  !
  ! This method provides adaptive outlier detection that adjusts to local
  ! trends in the data, making it suitable for non-stationary time series.
  !
  ! Algorithm for each position i >= window:
  !   1. Extract window: data[i-window+1 : i]
  !   2. Compute median and MAD of window
  !   3. Compute z_robust = (data[i] - median) / (MAD * 1.4826)
  !   4. Flag as anomaly if |z_robust| > threshold
  !
  ! For i < window, output is set to 0 (no detection possible).
  !
  ! Arguments (C view):
  !   x         - const double*    input time series of length n
  !   n         - int             number of elements
  !   window    - int             window length
  !   threshold - double          z-score threshold (e.g., 3.0)
  !   anomaly   - int*            output: 1=anomaly, 0=normal
  !   status    - int*            output status code
  !
  !--------------------------------------------------------------------------
  subroutine hpcs_rolling_anomalies(x, n, window, threshold, anomaly, status) &
       bind(C, name="hpcs_rolling_anomalies")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double),  value      :: threshold
    integer(c_int),  intent(out):: anomaly(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, window_eff, st
    integer(c_int) :: i, j, idx
    real(c_double) :: med, mad_val, z_robust
    real(c_double), allocatable :: buf(:), dev(:)
    real(c_double), parameter :: scale = 1.4826_c_double

    n_eff = n
    window_eff = window

    ! Validate arguments
    if (n_eff <= 0_c_int .or. window_eff <= 0_c_int .or. &
        window_eff > n_eff .or. threshold < 0.0_c_double) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Allocate temporary buffers for window data
    allocate(buf(window_eff), dev(window_eff), stat=st)
    if (st /= 0) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Initialize outputs for indices before full window
    do i = 1_c_int, window_eff - 1_c_int
      anomaly(i) = 0_c_int  ! Cannot detect before full window
    end do

    ! Rolling anomaly detection starting from window position
    do i = window_eff, n_eff
      ! Copy current window into buffer
      do j = 1_c_int, window_eff
        idx = i - window_eff + j
        buf(j) = x(idx)
      end do

      ! Check if window contains NaN
      if (buf(window_eff) /= buf(window_eff)) then
        ! Current value is NaN: flag as non-anomaly (or could flag as anomaly)
        anomaly(i) = 0_c_int
        cycle
      end if

      ! Compute median of window
      call hpcs_median(buf, window_eff, med, st)
      if (st /= HPCS_SUCCESS) then
        anomaly(i) = 0_c_int
        cycle
      end if

      ! Compute MAD of window
      ! First compute deviations from median
      do j = 1_c_int, window_eff
        if (buf(j) /= buf(j)) then
          dev(j) = ieee_value(0.0_c_double, ieee_quiet_nan)
        else
          dev(j) = abs(buf(j) - med)
        end if
      end do

      ! Median of absolute deviations
      call hpcs_median(dev, window_eff, mad_val, st)
      if (st /= HPCS_SUCCESS) then
        anomaly(i) = 0_c_int
        cycle
      end if

      ! Check if MAD is degenerate (all values equal)
      if (mad_val < 1.0e-12_c_double) then
        ! No variation in window: current value cannot be anomaly
        anomaly(i) = 0_c_int
      else
        ! Compute robust z-score for current value
        z_robust = abs((x(i) - med) / (mad_val * scale))

        ! Flag if exceeds threshold
        if (z_robust > threshold) then
          anomaly(i) = 1_c_int  ! Anomaly
        else
          anomaly(i) = 0_c_int  ! Normal
        end if
      end if
    end do

    deallocate(buf, dev)
    status = HPCS_SUCCESS
  end subroutine hpcs_rolling_anomalies

end module hpcs_core_rolling