! ==============================================================================
! HPC Series Core Library - Reductions Module
! Simple and grouped reduction operations
!
! Implemented kernels:
!   - hpcs_reduce_sum:        Simple sum reduction
!   - hpcs_reduce_min:        Minimum value (returns +∞ for empty arrays)
!   - hpcs_reduce_max:        Maximum value (returns -∞ for empty arrays)
!   - hpcs_group_reduce_sum:  Group-wise sum (0-based group IDs)
!   - hpcs_group_reduce_mean: Group-wise mean (NaN for empty groups)
!
! All functions use C-compatible interfaces via iso_c_binding
! ==============================================================================

module hpcs_core_reductions
  use iso_c_binding, only: c_int, c_double
  use hpcs_constants
  implicit none

contains

  !--------------------------------------------------------------------
  ! Simple sum reduction
  ! out = sum(x(1:n))
  ! n == 0 -> out = 0.0, success
  ! n <  0 -> invalid args
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_sum(x, n, out, status) &
       bind(C, name="hpcs_reduce_sum")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)      ! length n
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff
    real(c_double) :: acc

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
    do i = 1_c_int, n_eff
       acc = acc + x(i)
    end do

    out = acc
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_sum

  !--------------------------------------------------------------------
  ! Reduce minimum
  ! n == 0 -> out = +huge, success (sentinel)
  ! n <  0 -> invalid args
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_min(x, n, out, status) &
       bind(C, name="hpcs_reduce_min")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)      ! length n
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff
    real(c_double) :: minval

    n_eff = n

    if (n_eff < 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (n_eff == 0_c_int) then
      out = huge(0.0_c_double)   ! +∞ sentinel
      status = HPCS_SUCCESS
      return
    end if

    minval = x(1_c_int)
    do i = 2_c_int, n_eff
       if (x(i) < minval) minval = x(i)
    end do

    out = minval
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_min

  !--------------------------------------------------------------------
  ! Reduce maximum
  ! n == 0 -> out = -huge, success (sentinel)
  ! n <  0 -> invalid args
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_max(x, n, out, status) &
       bind(C, name="hpcs_reduce_max")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)      ! length n
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff
    real(c_double) :: maxval

    n_eff = n

    if (n_eff < 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (n_eff == 0_c_int) then
      out = -huge(0.0_c_double)   ! -∞ sentinel
      status = HPCS_SUCCESS
      return
    end if

    maxval = x(1_c_int)
    do i = 2_c_int, n_eff
       if (x(i) > maxval) maxval = x(i)
    end do

    out = maxval
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_max

  !--------------------------------------------------------------------
  ! Grouped sum:
  !   x(1:n), group_ids(1:n) in [0, n_groups-1]
  !   y(0:n_groups-1) stored as y(1:n_groups) in Fortran
  ! Invalid group IDs are ignored.
  !--------------------------------------------------------------------
  subroutine hpcs_group_reduce_sum(x, n, group_ids, n_groups, y, status) &
       bind(C, name="hpcs_group_reduce_sum")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)           ! length n
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)   ! length n
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)           ! length n_groups
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, g
    integer(c_int) :: n_eff, ng_eff

    n_eff  = n
    ng_eff = n_groups

    if (n_eff < 0_c_int .or. ng_eff < 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (ng_eff == 0_c_int) then
       ! No groups – nothing to accumulate
       status = HPCS_SUCCESS
       return
    end if

    ! Initialise sums to zero
    do g = 1_c_int, ng_eff
       y(g) = 0.0_c_double
    end do

    do i = 1_c_int, n_eff
       g = group_ids(i)
       if (g < 0_c_int .or. g >= ng_eff) cycle
       y(g + 1_c_int) = y(g + 1_c_int) + x(i)
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_group_reduce_sum

  !--------------------------------------------------------------------
  ! Grouped mean:
  !   mean_k = sum_{i in group k} x(i) / count_k
  ! Groups with zero count -> NaN
  !--------------------------------------------------------------------
  subroutine hpcs_group_reduce_mean(x, n, group_ids, n_groups, y, status) &
       bind(C, name="hpcs_group_reduce_mean")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)           ! length n
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)   ! length n
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)           ! length n_groups
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, g
    integer(c_int) :: n_eff, ng_eff
    real(c_double), allocatable :: group_sum(:)
    integer(c_int), allocatable :: group_count(:)
    real(c_double) :: nan_val

    ! Use local pointers to automatic arrays
    n_eff  = n
    ng_eff = n_groups

    if (n_eff < 0_c_int .or. ng_eff < 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (ng_eff == 0_c_int) then
       status = HPCS_SUCCESS
       return
    end if

    ! Automatic arrays sized by number of groups
    allocate(group_sum(ng_eff))
    allocate(group_count(ng_eff))

    group_sum   = 0.0_c_double
    group_count = 0_c_int

    do i = 1_c_int, n_eff
       g = group_ids(i)
       if (g < 0_c_int .or. g >= ng_eff) cycle
       group_sum(g + 1_c_int)   = group_sum(g + 1_c_int)   + x(i)
       group_count(g + 1_c_int) = group_count(g + 1_c_int) + 1_c_int
    end do

    ! Produce a NaN value for groups with zero count
    nan_val = 0.0_c_double / 0.0_c_double

    do g = 1_c_int, ng_eff
       if (group_count(g) > 0_c_int) then
          y(g) = group_sum(g) / real(group_count(g), kind=c_double)
       else
          y(g) = nan_val
       end if
    end do

    deallocate(group_sum)
    deallocate(group_count)

    status = HPCS_SUCCESS
  end subroutine hpcs_group_reduce_mean

end module hpcs_core_reductions
