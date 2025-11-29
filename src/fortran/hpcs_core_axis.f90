! ==============================================================================
! HPC Series Core Library - Axis‑Based Reductions (v0.4 CPU)
!
! This module implements column‑ and row‑wise reductions for 2D arrays stored
! in column‑major order.  Axis‑0 reductions operate across rows and produce
! a vector of length m (one value per column).  Axis‑1 reductions operate
! across columns and produce a vector of length n.  The axis‑1 sum and mean
! routines are implemented in hpcs_core_batched to reuse buffer logic for
! robust statistics; this module provides the axis‑0 counterparts.  All
! routines follow the HPCSeries ABI: arguments are passed by value where
! appropriate, arrays are contiguous double precision, and status codes are
! returned via an explicit integer(c_int) argument.  Loops are written in
! simple, flat form to facilitate OpenMP parallelisation in a future
! revision.
! ============================================================================

module hpcs_core_axis
  use iso_c_binding, only: c_int, c_double
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  use hpcs_constants
  implicit none
  private
  public :: hpcs_reduce_sum_axis0
  public :: hpcs_reduce_mean_axis0
  public :: hpcs_reduce_min_axis0
  public :: hpcs_reduce_max_axis0

contains

  !--------------------------------------------------------------------------
  ! hpcs_reduce_sum_axis0
  !
  ! Compute the sum along axis‑0 (rows) for each column of x(n,m).  The input
  ! matrix x is stored in column‑major order with shape (n,m), so the element
  ! at row i and column j is located at x(i + (j-1)*n).  For each column j,
  ! this routine accumulates x(i,j) for i=1..n and stores the result in
  ! y(j).  NaNs propagate: if any element in column j is NaN then y(j)
  ! becomes NaN.  If invalid arguments (n<=0 or m<=0) are supplied, the
  ! status is set to HPCS_ERR_INVALID_ARGS and no computation is performed.
  !
  ! Arguments (C view):
  !   x      - const double*  input array of length n*m
  !   n      - int           number of rows
  !   m      - int           number of columns
  !   y      - double*       output array of length m
  !   status - int*          output status code
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_sum_axis0(x, n, m, y, status) &
       bind(C, name="hpcs_reduce_sum_axis0")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, m_eff
    integer(c_int) :: col, row, idx
    real(c_double) :: acc
    logical        :: has_nan

    n_eff = n
    m_eff = m
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Loop over columns
    ! Higher threshold (500k) for axis-0 sum - marginal benefit at smaller sizes
!$omp parallel do default(none) shared(x,y,n_eff,m_eff) &
!$omp private(col,row,idx,acc,has_nan) if(n_eff*m_eff > 500000)
    do col = 1_c_int, m_eff
      acc = 0.0_c_double
      has_nan = .false.
      ! Accumulate across rows
      do row = 1_c_int, n_eff
        idx = row + (col - 1_c_int) * n_eff
        if (x(idx) /= x(idx)) then
          has_nan = .true.
        else
          acc = acc + x(idx)
        end if
      end do
      if (has_nan) then
        y(col) = ieee_value(0.0_c_double, ieee_quiet_nan)
      else
        y(col) = acc
      end if
    end do
!$omp end parallel do
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_sum_axis0

  !--------------------------------------------------------------------------
  ! hpcs_reduce_mean_axis0
  !
  ! Compute the mean along axis‑0 (rows) for each column of x(n,m).
  ! NaNs propagate: if any element in a column is NaN the mean for that
  ! column is NaN.  Invalid arguments return status = 1.  The result
  ! y(j) = sum_{i=1..n} x(i,j)/n.
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_mean_axis0(x, n, m, y, status) &
       bind(C, name="hpcs_reduce_mean_axis0")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, m_eff
    integer(c_int) :: col, row, idx
    real(c_double) :: acc
    logical        :: has_nan

    n_eff = n
    m_eff = m
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Higher threshold (500k) for axis-0 mean - marginal benefit at smaller sizes
!$omp parallel do default(none) shared(x,y,n_eff,m_eff) &
!$omp private(col,row,idx,acc,has_nan) if(n_eff*m_eff > 500000)
    do col = 1_c_int, m_eff
      acc = 0.0_c_double
      has_nan = .false.
      do row = 1_c_int, n_eff
        idx = row + (col - 1_c_int) * n_eff
        if (x(idx) /= x(idx)) then
          has_nan = .true.
        else
          acc = acc + x(idx)
        end if
      end do
      if (has_nan) then
        y(col) = ieee_value(0.0_c_double, ieee_quiet_nan)
      else
        y(col) = acc / real(n_eff, kind=c_double)
      end if
    end do
!$omp end parallel do
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_mean_axis0

  !--------------------------------------------------------------------------
  ! hpcs_reduce_min_axis0
  !
  ! Compute the minimum value along axis‑0 for each column of x(n,m).  If
  ! any element in a column is NaN, the result for that column is NaN.
  ! Invalid arguments set status = 1 and no computation occurs.  For an
  ! empty column (n==0) the output is +huge(0.0).
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_min_axis0(x, n, m, y, status) &
       bind(C, name="hpcs_reduce_min_axis0")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, m_eff
    integer(c_int) :: col, row, idx
    real(c_double) :: minval
    logical        :: has_nan

    n_eff = n
    m_eff = m
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! OpenMP disabled: too fast and memory-bound, overhead exceeds benefit (1.17x speedup)
    do col = 1_c_int, m_eff
      has_nan = .false.
      minval = huge(0.0_c_double)
      do row = 1_c_int, n_eff
        idx = row + (col - 1_c_int) * n_eff
        if (x(idx) /= x(idx)) then
          has_nan = .true.
        else
          if (x(idx) < minval) minval = x(idx)
        end if
      end do
      if (has_nan) then
        y(col) = ieee_value(0.0_c_double, ieee_quiet_nan)
      else
        y(col) = minval
      end if
    end do
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_min_axis0

  !--------------------------------------------------------------------------
  ! hpcs_reduce_max_axis0
  !
  ! Compute the maximum value along axis‑0 for each column of x(n,m).  NaNs
  ! propagate: if any element in a column is NaN, the result for that
  ! column is NaN.  Invalid arguments set status = 1.  For an empty
  ! column the result is -huge(0.0).
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_max_axis0(x, n, m, y, status) &
       bind(C, name="hpcs_reduce_max_axis0")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, m_eff
    integer(c_int) :: col, row, idx
    real(c_double) :: maxval
    logical        :: has_nan

    n_eff = n
    m_eff = m
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! OpenMP disabled: too fast and memory-bound, overhead causes slowdown (0.68x speedup)
    do col = 1_c_int, m_eff
      has_nan = .false.
      maxval = -huge(0.0_c_double)
      do row = 1_c_int, n_eff
        idx = row + (col - 1_c_int) * n_eff
        if (x(idx) /= x(idx)) then
          has_nan = .true.
        else
          if (x(idx) > maxval) maxval = x(idx)
        end if
      end do
      if (has_nan) then
        y(col) = ieee_value(0.0_c_double, ieee_quiet_nan)
      else
        y(col) = maxval
      end if
    end do
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_max_axis0

end module hpcs_core_axis