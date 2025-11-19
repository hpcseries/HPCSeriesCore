! ==============================================================================
! HPC Series Core Library - Array Utilities Module
! Simple element-wise utilities for 1D arrays
!
! Implemented kernels:
!   - hpcs_fill_missing:
!       Replace missing values (by sentinel) and optionally NaNs with a
!       replacement value, in place.
!
!   - hpcs_where:
!       Element-wise conditional select:
!         y(i) = a(i) if mask(i) != 0
!                b(i) otherwise
!
!   - hpcs_fill_value:
!       In-place fill with a constant value
!
!   - hpcs_copy:
!       Copy array elements from source to destination
!
! All functions use C-compatible interfaces via iso_c_binding and shared
! status codes from hpcs_constants.
! ==============================================================================

module hpcs_core_utils
  use iso_c_binding, only: c_int, c_double
  use hpcs_constants
  implicit none
  public

contains

  !--------------------------------------------------------------------
  ! hpcs_fill_missing
  !
  ! In-place fill of "missing" values.
  !
  ! Arguments
  !   x                     : inout array of length n
  !   n                     : number of elements
  !   missing_value         : sentinel value to be treated as missing
  !   replacement           : value used to replace missing entries
  !   treat_nan_as_missing  : if non-zero, NaNs are also treated as missing
  !
  ! Behavior
  !   For i = 1..n:
  !     - if x(i) == missing_value       -> x(i) = replacement
  !     - if treat_nan_as_missing != 0 and x(i) is NaN -> x(i) = replacement
  !
  ! Status codes
  !   HPCS_SUCCESS          : success (including n == 0)
  !   HPCS_ERR_INVALID_ARGS : n < 0
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
    logical        :: use_nan, missing_is_nan

    n_eff  = n
    use_nan = (treat_nan_as_missing /= 0_c_int)
    missing_is_nan = (missing_value /= missing_value)  ! Check if sentinel is NaN

    if (n_eff < 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (n_eff == 0_c_int) then
       status = HPCS_SUCCESS
       return
    end if

    do i = 1_c_int, n_eff
       if (missing_is_nan) then
          ! Sentinel is NaN: check for any NaN
          if (x(i) /= x(i)) then
             x(i) = replacement
          end if
       else
          ! Normal sentinel: check exact match
          if (x(i) == missing_value) then
             x(i) = replacement

          ! Optional NaN-as-missing: NaN is the only value where x /= x
          else if (use_nan .and. x(i) /= x(i)) then
             x(i) = replacement
          end if
       end if
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_fill_missing

  !--------------------------------------------------------------------
  ! hpcs_where
  !
  ! Element-wise conditional select (mask/where)
  !
  ! Arguments
  !   mask  : integer mask array of length n (0 = false, non-zero = true)
  !   n     : number of elements
  !   a     : input array a, length n
  !   b     : input array b, length n
  !   y     : output array y, length n
  !
  ! Behavior
  !   For i = 1..n:
  !     if mask(i) /= 0:
  !        y(i) = a(i)
  !     else
  !        y(i) = b(i)
  !
  ! Status codes
  !   HPCS_SUCCESS          : success (including n == 0)
  !   HPCS_ERR_INVALID_ARGS : n < 0
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
  ! In-place fill with a constant value
  !
  ! Arguments
  !   x      : inout array of length n
  !   n      : number of elements
  !   value  : fill value
  !
  ! Behavior
  !   For i = 1..n:
  !     x(i) = value
  !
  ! Note: No status code returned (always succeeds for valid pointers)
  !--------------------------------------------------------------------
  subroutine hpcs_fill_value(x, n, value) &
       bind(C, name="hpcs_fill_value")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(inout) :: x(*)      ! length n
    integer(c_int),  value        :: n
    real(c_double),  value        :: value

    integer(c_int) :: i, n_eff

    n_eff = n

    if (n_eff <= 0_c_int) return

    do i = 1_c_int, n_eff
       x(i) = value
    end do
  end subroutine hpcs_fill_value

  !--------------------------------------------------------------------
  ! hpcs_copy
  !
  ! Copy array x to array y
  !
  ! Arguments
  !   x  : input array of length n
  !   n  : number of elements
  !   y  : output array of length n
  !
  ! Behavior
  !   For i = 1..n:
  !     y(i) = x(i)
  !
  ! Note: No status code returned (always succeeds for valid pointers)
  !--------------------------------------------------------------------
  subroutine hpcs_copy(x, n, y) &
       bind(C, name="hpcs_copy")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)      ! length n
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: y(*)      ! length n

    integer(c_int) :: i, n_eff

    n_eff = n

    if (n_eff <= 0_c_int) return

    do i = 1_c_int, n_eff
       y(i) = x(i)
    end do
  end subroutine hpcs_copy

end module hpcs_core_utils
