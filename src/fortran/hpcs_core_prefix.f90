! ==============================================================================
! HPC Series Core Library - Prefix Sum Kernels (v0.2)
!
! New kernels:
!   - hpcs_prefix_sum           : inclusive prefix sum
!   - hpcs_prefix_sum_exclusive : exclusive prefix sum
!
! ABI (from v0.2 core spec):
!   void hpcs_prefix_sum(const double *x, int n, double *y, int *status);
!   void hpcs_prefix_sum_exclusive(const double *x, int n, double *y, int *status);
!
! These are currently implemented as serial O(n) loops, with flat structure
! ready for OpenMP parallel-scan in a future version.
! ==============================================================================

module hpcs_core_prefix
  use iso_c_binding,  only: c_int, c_double
  use hpcs_constants
  implicit none
  public

contains

  !--------------------------------------------------------------------
  ! Inclusive prefix sum:
  !   y(i) = sum_{k=1..i} x(k)
  !--------------------------------------------------------------------
  subroutine hpcs_prefix_sum(x, n, y, status) &
       bind(C, name="hpcs_prefix_sum")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)      ! length >= n
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: y(*)      ! length >= n
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff
    real(c_double) :: acc

    n_eff = n

    if (n_eff < 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (n_eff == 0_c_int) then
       status = HPCS_SUCCESS
       return
    end if

    acc = 0.0_c_double

    do i = 1_c_int, n_eff
       acc  = acc + x(i)
       y(i) = acc
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_prefix_sum

  !--------------------------------------------------------------------
  ! Exclusive prefix sum:
  !   y(1) = 0
  !   y(i) = sum_{k=1..i-1} x(k)   for i > 1
  !--------------------------------------------------------------------
  subroutine hpcs_prefix_sum_exclusive(x, n, y, status) &
       bind(C, name="hpcs_prefix_sum_exclusive")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)      ! length >= n
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: y(*)      ! length >= n
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff
    real(c_double) :: acc

    n_eff = n

    if (n_eff < 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (n_eff == 0_c_int) then
       status = HPCS_SUCCESS
       return
    end if

    acc = 0.0_c_double

    do i = 1_c_int, n_eff
       y(i) = acc
       acc  = acc + x(i)
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_prefix_sum_exclusive

end module hpcs_core_prefix
