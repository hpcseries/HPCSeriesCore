! ==============================================================================
! HPC Series Core Library - Reductions Module
! Simple 1D reductions and grouped reductions.
!
! Implemented kernels:
!   - hpcs_reduce_sum
!   - hpcs_reduce_min
!   - hpcs_reduce_max
!   - hpcs_reduce_mean       (v0.2)
!   - hpcs_reduce_variance   (v0.2)
!   - hpcs_reduce_std        (v0.2)
!   - hpcs_group_reduce_sum
!   - hpcs_group_reduce_mean
!   - hpcs_group_reduce_variance (v0.2)
!
! All routines:
!   - use ISO_C_BINDING with bind(C)
!   - return status via an explicit integer(c_int) argument
! ==============================================================================

module hpcs_core_reductions
  use iso_c_binding,  only: c_int, c_double
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  use hpcs_constants
  implicit none
  public

contains

  !--------------------------------------------------------------------
  ! hpcs_reduce_sum
  !
  ! out = sum(x(1:n))
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0
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

    if (n_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    acc = 0.0_c_double
    do i = 1_c_int, n_eff
       acc = acc + x(i)
    end do

    out    = acc
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_sum

  !--------------------------------------------------------------------
  ! hpcs_reduce_min
  !
  ! out = min(x(1:n))
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0
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

    if (n_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    minval = x(1_c_int)
    do i = 2_c_int, n_eff
       if (x(i) < minval) minval = x(i)
    end do

    out    = minval
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_min

  !--------------------------------------------------------------------
  ! hpcs_reduce_max
  !
  ! out = max(x(1:n))
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0
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

    if (n_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    maxval = x(1_c_int)
    do i = 2_c_int, n_eff
       if (x(i) > maxval) maxval = x(i)
    end do

    out    = maxval
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_max

  !--------------------------------------------------------------------
  ! hpcs_reduce_mean (v0.2)
  !
  ! out = mean(x(1:n)) = sum(x) / n
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_mean(x, n, out, status) &
       bind(C, name="hpcs_reduce_mean")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)      ! length n
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff
    real(c_double) :: sum_val

    n_eff = n

    if (n_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    sum_val = 0.0_c_double
    do i = 1_c_int, n_eff
       sum_val = sum_val + x(i)
    end do

    out    = sum_val / real(n_eff, kind=c_double)
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_mean

  !--------------------------------------------------------------------
  ! hpcs_reduce_variance (v0.2)
  !
  ! out = variance(x(1:n)) = sum((x - mean)^2) / n  (population variance)
  !
  ! Uses Welford's online algorithm for numerical stability:
  !   M_k = M_{k-1} + (x_k - M_{k-1}) / k
  !   S_k = S_{k-1} + (x_k - M_{k-1}) * (x_k - M_k)
  !   variance = S_n / n
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_variance(x, n, out, status) &
       bind(C, name="hpcs_reduce_variance")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)      ! length n
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff
    real(c_double) :: mean_running, s_running, delta, delta2

    n_eff = n

    if (n_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (n_eff == 1_c_int) then
       out    = 0.0_c_double
       status = HPCS_SUCCESS
       return
    end if

    ! Welford's algorithm
    mean_running = 0.0_c_double
    s_running    = 0.0_c_double

    do i = 1_c_int, n_eff
       delta        = x(i) - mean_running
       mean_running = mean_running + delta / real(i, kind=c_double)
       delta2       = x(i) - mean_running
       s_running    = s_running + delta * delta2
    end do

    out    = s_running / real(n_eff, kind=c_double)
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_variance

  !--------------------------------------------------------------------
  ! hpcs_reduce_std (v0.2)
  !
  ! out = sqrt(variance(x(1:n)))  (population standard deviation)
  !
  ! Uses Welford's algorithm via hpcs_reduce_variance, then takes sqrt.
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_std(x, n, out, status) &
       bind(C, name="hpcs_reduce_std")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)      ! length n
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    real(c_double) :: variance

    ! Compute variance first
    call hpcs_reduce_variance(x, n, variance, status)
    if (status /= HPCS_SUCCESS) then
       return
    end if

    ! Take square root
    out = sqrt(variance)
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_std

  !--------------------------------------------------------------------
  ! hpcs_group_reduce_sum
  !
  ! Grouped sum:
  !   x(1:n), group_ids(1:n) in [0, n_groups-1]
  !   y(0:n_groups-1) stored as y(1:n_groups) in Fortran
  !
  ! Invalid group IDs (<0 or >= n_groups) are ignored.
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or n_groups <= 0
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

    if (n_eff <= 0_c_int .or. ng_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
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
  ! hpcs_group_reduce_mean
  !
  ! Grouped mean:
  !   mean_k = sum_{i in group k} x(i) / count_k
  !
  ! Groups with zero count -> NaN.
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or n_groups <= 0
  !   HPCS_ERR_NUMERIC_FAIL : allocation failure (very unlikely)
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

    integer(c_int)           :: i, g
    integer(c_int)           :: n_eff, ng_eff
    real(c_double), allocatable :: group_sum(:)
    integer(c_int), allocatable :: group_count(:)
    integer(c_int)           :: istat
    real(c_double)           :: nan_val

    n_eff  = n
    ng_eff = n_groups

    if (n_eff <= 0_c_int .or. ng_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    ! Allocate temporary accumulators
    allocate(group_sum(ng_eff), group_count(ng_eff), stat=istat)
    if (istat /= 0_c_int) then
       status = HPCS_ERR_NUMERIC_FAIL
       return
    end if

    group_sum   = 0.0_c_double
    group_count = 0_c_int

    do i = 1_c_int, n_eff
       g = group_ids(i)
       if (g < 0_c_int .or. g >= ng_eff) cycle
       group_sum(g + 1_c_int)   = group_sum(g + 1_c_int)   + x(i)
       group_count(g + 1_c_int) = group_count(g + 1_c_int) + 1_c_int
    end do

    ! NaN value for empty groups
    nan_val = ieee_value(0.0_c_double, ieee_quiet_nan)

    do g = 1_c_int, ng_eff
       if (group_count(g) > 0_c_int) then
          y(g) = group_sum(g) / real(group_count(g), kind=c_double)
       else
          y(g) = nan_val
       end if
    end do

    deallocate(group_sum, group_count, stat=istat)
    ! Ignore deallocation errors for now; we've already produced y.

    status = HPCS_SUCCESS
  end subroutine hpcs_group_reduce_mean

  !--------------------------------------------------------------------
  ! hpcs_group_reduce_variance (v0.2)
  !
  ! Grouped variance (population):
  !   variance_k = sum_{i in group k} (x_i - mean_k)^2 / count_k
  !
  ! Uses Welford's algorithm per group for numerical stability.
  ! Groups with zero count -> NaN.
  !
  ! Status:
  !   HPCS_SUCCESS          : success
  !   HPCS_ERR_INVALID_ARGS : n <= 0 or n_groups <= 0
  !   HPCS_ERR_NUMERIC_FAIL : allocation failure
  !--------------------------------------------------------------------
  subroutine hpcs_group_reduce_variance(x, n, group_ids, n_groups, y, status) &
       bind(C, name="hpcs_group_reduce_variance")
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
    real(c_double), allocatable :: group_mean(:)   ! Running mean per group
    real(c_double), allocatable :: group_m2(:)     ! Sum of squared deviations
    integer(c_int), allocatable :: group_count(:)
    integer(c_int) :: istat
    real(c_double) :: nan_val, delta, delta2
    integer(c_int) :: count_k

    n_eff  = n
    ng_eff = n_groups

    if (n_eff <= 0_c_int .or. ng_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    ! Allocate temporary accumulators for Welford's algorithm
    allocate(group_mean(ng_eff), group_m2(ng_eff), group_count(ng_eff), stat=istat)
    if (istat /= 0_c_int) then
       status = HPCS_ERR_NUMERIC_FAIL
       return
    end if

    ! Initialize Welford accumulators
    group_mean  = 0.0_c_double
    group_m2    = 0.0_c_double
    group_count = 0_c_int

    ! Welford's algorithm: update mean and M2 for each group
    do i = 1_c_int, n_eff
       g = group_ids(i)
       if (g < 0_c_int .or. g >= ng_eff) cycle

       ! Convert to 1-based Fortran indexing
       g = g + 1_c_int

       ! Update count
       group_count(g) = group_count(g) + 1_c_int
       count_k = group_count(g)

       ! Welford's online variance algorithm
       delta = x(i) - group_mean(g)
       group_mean(g) = group_mean(g) + delta / real(count_k, kind=c_double)
       delta2 = x(i) - group_mean(g)
       group_m2(g) = group_m2(g) + delta * delta2
    end do

    ! Compute variance for each group
    nan_val = ieee_value(0.0_c_double, ieee_quiet_nan)

    do g = 1_c_int, ng_eff
       if (group_count(g) > 0_c_int) then
          ! Population variance: M2 / count
          y(g) = group_m2(g) / real(group_count(g), kind=c_double)
       else
          y(g) = nan_val
       end if
    end do

    deallocate(group_mean, group_m2, group_count, stat=istat)

    status = HPCS_SUCCESS
  end subroutine hpcs_group_reduce_variance

end module hpcs_core_reductions
