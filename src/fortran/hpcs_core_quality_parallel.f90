! ==============================================================================
! HPC Series Core Library - Parallel Data Quality Module (v0.3)
!
! Provides OpenMP-parallelized version of robust z-score computation for
! large arrays (n >= 100K).
! ==============================================================================

module hpcs_core_quality_parallel
  use iso_c_binding, only: c_int, c_double
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  use hpcs_constants
  use hpcs_core_stats_parallel, only: hpcs_median_parallel, hpcs_mad_parallel
  implicit none
  private
  public :: hpcs_robust_zscore_parallel

  integer(c_int), parameter :: PARALLEL_THRESHOLD = 100000_c_int

contains

  !--------------------------------------------------------------------------
  ! hpcs_robust_zscore_parallel
  !
  ! Parallel version of robust z-score. Uses parallel median/MAD and
  ! parallelizes the final z-score computation loop.
  !--------------------------------------------------------------------------
  subroutine hpcs_robust_zscore_parallel(x, n, y, status) &
       bind(C, name="hpcs_robust_zscore_parallel")
    use hpcs_core_quality, only: hpcs_robust_zscore
    use omp_lib
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, st, i
    real(c_double) :: med, mad_val, scale

    n_eff = n

    ! Small arrays: use serial version
    if (n_eff < PARALLEL_THRESHOLD) then
      call hpcs_robust_zscore(x, n_eff, y, status)
      return
    end if

    if (n_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    scale = 1.4826_c_double

    ! Compute median (parallel)
    call hpcs_median_parallel(x, n_eff, med, st)
    if (st /= HPCS_SUCCESS) then
      status = st
      return
    end if

    ! Compute MAD (parallel)
    call hpcs_mad_parallel(x, n_eff, mad_val, st)
    if (st == HPCS_ERR_NUMERIC_FAIL) then
      ! Degenerate distribution: fill y with NaNs (parallel)
      !$OMP PARALLEL DO SCHEDULE(STATIC)
      do i = 1_c_int, n_eff
        y(i) = ieee_value(0.0_c_double, ieee_quiet_nan)
      end do
      !$OMP END PARALLEL DO
      status = HPCS_ERR_NUMERIC_FAIL
      return
    else if (st /= HPCS_SUCCESS) then
      status = st
      return
    end if

    ! Compute robust z-scores (parallel)
    !$OMP PARALLEL DO SCHEDULE(STATIC)
    do i = 1_c_int, n_eff
      if (x(i) /= x(i)) then
        y(i) = ieee_value(0.0_c_double, ieee_quiet_nan)
      else
        y(i) = (x(i) - med) / (mad_val * scale)
      end if
    end do
    !$OMP END PARALLEL DO

    status = HPCS_SUCCESS
  end subroutine hpcs_robust_zscore_parallel

end module hpcs_core_quality_parallel
