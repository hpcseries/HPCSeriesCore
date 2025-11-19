! ==============================================================================
! HPC Series Core Library - Array Utilities Module
! Simple element-wise utilities for 1D arrays.
!
! Implemented kernels:
!   - hpcs_fill_missing
!   - hpcs_where
!   - hpcs_fill_value
!   - hpcs_copy
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

end module hpcs_core_utils
