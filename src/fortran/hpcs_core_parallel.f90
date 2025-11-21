! ==============================================================================
! HPC Series Core Library - OpenMP Parallel Kernels (v0.2)
!
! New kernels (ABI from HPCSeries_Core_v0.2_Specification):
!
!   Prefix sums are in hpcs_core_prefix:
!     void hpcs_prefix_sum(const double *x, int n, double *y, int *status);
!     void hpcs_prefix_sum_exclusive(const double *x, int n, double *y, int *status);
!
!   Parallel rolling:
!     void hpcs_rolling_sum_parallel(const double *x, int n, int window, double *y, int *status);
!     void hpcs_rolling_mean_parallel(const double *x, int n, int window, double *y, int *status);
!
!   Parallel reductions:
!     void hpcs_reduce_sum_parallel(const double *x, int n, double *out, int *status);
!     void hpcs_reduce_min_parallel(const double *x, int n, double *out, int *status);
!     void hpcs_reduce_max_parallel(const double *x, int n, double *out, int *status);
!
!   Parallel group reductions:
!     void hpcs_group_reduce_sum_parallel(const double *x, int n, const int *group_ids,
!                                         int n_groups, double *y, int *status);
!     void hpcs_group_reduce_mean_parallel(const double *x, int n, const int *group_ids,
!                                          int n_groups, double *y, int *status);
!
!   Parallel z-score:
!     void hpcs_zscore_parallel(const double *x, int n, double *y, int *status);
!
! Notes:
!   - Uses OpenMP pragmas where appropriate.
!   - Falls back to serial kernels below HPCS_PARALLEL_THRESHOLD.
!   - Compiles fine without OpenMP (directives are comments then).
! ==============================================================================

module hpcs_core_parallel
  use iso_c_binding,  only: c_int, c_double
  use hpcs_constants
  use hpcs_core_1d        ! serial rolling + zscore
  use hpcs_core_reductions
  implicit none
  public

  integer(c_int), parameter :: HPCS_PARALLEL_THRESHOLD = 100000_c_int

contains

  !====================================================================
  ! PARALLEL ROLLING OPERATIONS
  !====================================================================

  !--------------------------------------------------------------------
  ! Parallel rolling sum (v0.2 ABI).
  !
  ! Current implementation:
  !   - Delegates to serial hpcs_rolling_sum for correctness.
  !   - Flat call, ready to be replaced with prefix-sum + OpenMP scheme.
  !--------------------------------------------------------------------
  subroutine hpcs_rolling_sum_parallel(x, n, window, y, status) &
       bind(C, name="hpcs_rolling_sum_parallel")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)      ! length >= n
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)      ! length >= n
    integer(c_int),  intent(out):: status

    ! For v0.2 we simply delegate to the serial kernel.
    call hpcs_rolling_sum(x, n, window, y, status)
  end subroutine hpcs_rolling_sum_parallel

  !--------------------------------------------------------------------
  ! Parallel rolling mean (v0.2 ABI).
  !
  ! Same as above: currently delegates to serial rolling_mean.
  !--------------------------------------------------------------------
  subroutine hpcs_rolling_mean_parallel(x, n, window, y, status) &
       bind(C, name="hpcs_rolling_mean_parallel")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)      ! length >= n
    integer(c_int),  value      :: n
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)      ! length >= n
    integer(c_int),  intent(out):: status

    call hpcs_rolling_mean(x, n, window, y, status)
  end subroutine hpcs_rolling_mean_parallel

  !====================================================================
  ! PARALLEL REDUCTIONS
  !====================================================================

  !--------------------------------------------------------------------
  ! Parallel reduce sum
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_sum_parallel(x, n, out, status) &
       bind(C, name="hpcs_reduce_sum_parallel")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)      ! length >= n
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
       out    = 0.0_c_double
       status = HPCS_SUCCESS
       return
    end if

    if (n_eff < HPCS_PARALLEL_THRESHOLD) then
       call hpcs_reduce_sum(x, n, out, status)
       return
    end if

    acc = 0.0_c_double

!$omp parallel do default(none) shared(x, n_eff) reduction(+:acc)
    do i = 1_c_int, n_eff
       acc = acc + x(i)
    end do
!$omp end parallel do

    out    = acc
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_sum_parallel

  !--------------------------------------------------------------------
  ! Parallel reduce min
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_min_parallel(x, n, out, status) &
       bind(C, name="hpcs_reduce_min_parallel")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)      ! length >= n
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
       out    = huge(0.0_c_double)
       status = HPCS_SUCCESS
       return
    end if

    if (n_eff < HPCS_PARALLEL_THRESHOLD) then
       call hpcs_reduce_min(x, n, out, status)
       return
    end if

    minval = x(1_c_int)

!$omp parallel do default(none) shared(x, n_eff) reduction(min:minval)
    do i = 2_c_int, n_eff
       if (x(i) < minval) minval = x(i)
    end do
!$omp end parallel do

    out    = minval
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_min_parallel

  !--------------------------------------------------------------------
  ! Parallel reduce max
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_max_parallel(x, n, out, status) &
       bind(C, name="hpcs_reduce_max_parallel")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)      ! length >= n
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
       out    = -huge(0.0_c_double)
       status = HPCS_SUCCESS
       return
    end if

    if (n_eff < HPCS_PARALLEL_THRESHOLD) then
       call hpcs_reduce_max(x, n, out, status)
       return
    end if

    maxval = x(1_c_int)

!$omp parallel do default(none) shared(x, n_eff) reduction(max:maxval)
    do i = 2_c_int, n_eff
       if (x(i) > maxval) maxval = x(i)
    end do
!$omp end parallel do

    out    = maxval
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_max_parallel

  !====================================================================
  ! PARALLEL GROUP REDUCTIONS
  !====================================================================

  !--------------------------------------------------------------------
  ! Parallel group reduce sum (thread-private buffers).
  !--------------------------------------------------------------------
  subroutine hpcs_group_reduce_sum_parallel(x, n, group_ids, n_groups, y, status) &
       bind(C, name="hpcs_group_reduce_sum_parallel")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)           ! length >= n
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)   ! length >= n
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)           ! length >= n_groups
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, ng_eff
    integer(c_int) :: i, g

    real(c_double), allocatable :: group_sum(:)
    real(c_double), allocatable :: local_sum(:)
    integer(c_int)              :: istat

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

    if (n_eff < HPCS_PARALLEL_THRESHOLD) then
       call hpcs_group_reduce_sum(x, n, group_ids, n_groups, y, status)
       return
    end if

    allocate(group_sum(ng_eff), stat=istat)
    if (istat /= 0_c_int) then
       status = HPCS_ERR_NUMERIC_FAIL
       return
    end if
    group_sum = 0.0_c_double

!$omp parallel default(none) &
!$omp shared(x, group_ids, group_sum, n_eff, ng_eff) &
!$omp private(local_sum, i, g, istat)
    allocate(local_sum(ng_eff), stat=istat)
    if (istat == 0_c_int) then
       local_sum = 0.0_c_double

!$omp do
       do i = 1_c_int, n_eff
          g = group_ids(i)
          if (g < 0_c_int .or. g >= ng_eff) cycle
          local_sum(g + 1_c_int) = local_sum(g + 1_c_int) + x(i)
       end do
!$omp end do

!$omp critical
       do g = 1_c_int, ng_eff
          group_sum(g) = group_sum(g) + local_sum(g)
       end do
!$omp end critical

       deallocate(local_sum)
    end if
!$omp end parallel

    do g = 1_c_int, ng_eff
       y(g) = group_sum(g)
    end do

    deallocate(group_sum)

    status = HPCS_SUCCESS
  end subroutine hpcs_group_reduce_sum_parallel

  !--------------------------------------------------------------------
  ! Parallel group reduce mean (thread-private buffers).
  !--------------------------------------------------------------------
  subroutine hpcs_group_reduce_mean_parallel(x, n, group_ids, n_groups, y, status) &
       bind(C, name="hpcs_group_reduce_mean_parallel")
    use iso_c_binding, only: c_int, c_double
    use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
    implicit none

    real(c_double), intent(in)  :: x(*)           ! length >= n
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)   ! length >= n
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)           ! length >= n_groups
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, ng_eff
    integer(c_int) :: i, g, istat
    real(c_double), allocatable :: group_sum(:)
    integer(c_int), allocatable :: group_count(:)

    real(c_double), allocatable :: local_sum(:)
    integer(c_int), allocatable :: local_count(:)

    real(c_double) :: nan_val

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

    if (n_eff < HPCS_PARALLEL_THRESHOLD) then
       call hpcs_group_reduce_mean(x, n, group_ids, n_groups, y, status)
       return
    end if

    allocate(group_sum(ng_eff), group_count(ng_eff), stat=istat)
    if (istat /= 0_c_int) then
       status = HPCS_ERR_NUMERIC_FAIL
       return
    end if

    group_sum   = 0.0_c_double
    group_count = 0_c_int

!$omp parallel default(none) &
!$omp shared(x, group_ids, group_sum, group_count, n_eff, ng_eff) &
!$omp private(local_sum, local_count, i, g, istat)
    allocate(local_sum(ng_eff), local_count(ng_eff), stat=istat)
    if (istat == 0_c_int) then
       local_sum   = 0.0_c_double
       local_count = 0_c_int

!$omp do
       do i = 1_c_int, n_eff
          g = group_ids(i)
          if (g < 0_c_int .or. g >= ng_eff) cycle
          local_sum(g + 1_c_int)   = local_sum(g + 1_c_int)   + x(i)
          local_count(g + 1_c_int) = local_count(g + 1_c_int) + 1_c_int
       end do
!$omp end do

!$omp critical
       do g = 1_c_int, ng_eff
          group_sum(g)   = group_sum(g)   + local_sum(g)
          group_count(g) = group_count(g) + local_count(g)
       end do
!$omp end critical

       deallocate(local_sum, local_count)
    end if
!$omp end parallel

    nan_val = ieee_value(0.0_c_double, ieee_quiet_nan)

    do g = 1_c_int, ng_eff
       if (group_count(g) > 0_c_int) then
          y(g) = group_sum(g) / real(group_count(g), kind=c_double)
       else
          y(g) = nan_val
       end if
    end do

    deallocate(group_sum, group_count)

    status = HPCS_SUCCESS
  end subroutine hpcs_group_reduce_mean_parallel

  !====================================================================
  ! PARALLEL Z-SCORE
  !====================================================================

  !--------------------------------------------------------------------
  ! Parallel z-score via parallel sum/sumsq reductions.
  !--------------------------------------------------------------------
  subroutine hpcs_zscore_parallel(x, n, y, status) &
       bind(C, name="hpcs_zscore_parallel")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)      ! length >= n
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: y(*)      ! length >= n
    integer(c_int),  intent(out):: status

    integer(c_int)  :: i, n_eff
    real(c_double)  :: sum, sumsq, mean, variance, std

    n_eff = n

    if (n_eff < 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    if (n_eff == 0_c_int) then
       status = HPCS_SUCCESS
       return
    end if

    if (n_eff < HPCS_PARALLEL_THRESHOLD) then
       call hpcs_zscore(x, n, y, status)
       return
    end if

    sum   = 0.0_c_double
    sumsq = 0.0_c_double

!$omp parallel do default(none) shared(x, n_eff) reduction(+:sum, sumsq)
    do i = 1_c_int, n_eff
       sum   = sum   + x(i)
       sumsq = sumsq + x(i) * x(i)
    end do
!$omp end parallel do

    mean = sum / real(n_eff, kind=c_double)
    variance = sumsq / real(n_eff, kind=c_double) - mean * mean
    if (variance < 0.0_c_double) variance = 0.0_c_double
    std = sqrt(variance)

    if (std == 0.0_c_double) then
       do i = 1_c_int, n_eff
          y(i) = 0.0_c_double
       end do
       status = HPCS_ERR_NUMERIC_FAIL
       return
    end if

!$omp parallel do default(none) shared(x, y, n_eff, mean, std)
    do i = 1_c_int, n_eff
       y(i) = (x(i) - mean) / std
    end do
!$omp end parallel do

    status = HPCS_SUCCESS
  end subroutine hpcs_zscore_parallel

  !====================================================================
  ! PARALLEL REDUCE OPERATIONS (v0.2)
  !====================================================================

  !--------------------------------------------------------------------
  ! hpcs_reduce_mean_parallel
  !
  ! Parallel mean computation using OpenMP reduction.
  ! Falls back to serial version for n < HPCS_PARALLEL_THRESHOLD.
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_mean_parallel(x, n, out, status) &
       bind(C, name="hpcs_reduce_mean_parallel")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
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

    ! Fall back to serial for small arrays
    if (n_eff < HPCS_PARALLEL_THRESHOLD) then
       call hpcs_reduce_mean(x, n, out, status)
       return
    end if

    ! Parallel sum using OpenMP reduction
    sum_val = 0.0_c_double
!$omp parallel do default(none) shared(x, n_eff) reduction(+:sum_val)
    do i = 1_c_int, n_eff
       sum_val = sum_val + x(i)
    end do
!$omp end parallel do

    out = sum_val / real(n_eff, kind=c_double)
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_mean_parallel

  !--------------------------------------------------------------------
  ! hpcs_reduce_variance_parallel
  !
  ! Parallel population variance using two-pass algorithm with OpenMP.
  ! Falls back to serial version for n < HPCS_PARALLEL_THRESHOLD.
  !
  ! Two-pass algorithm:
  !   Pass 1: Compute mean (parallel reduction)
  !   Pass 2: Compute sum of squared deviations (parallel reduction)
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_variance_parallel(x, n, out, status) &
       bind(C, name="hpcs_reduce_variance_parallel")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    integer(c_int) :: i, n_eff
    real(c_double) :: mean_val, sum_sq_dev

    n_eff = n

    if (n_eff <= 0_c_int) then
       status = HPCS_ERR_INVALID_ARGS
       return
    end if

    ! Fall back to serial for small arrays
    if (n_eff < HPCS_PARALLEL_THRESHOLD) then
       call hpcs_reduce_variance(x, n, out, status)
       return
    end if

    ! Pass 1: Compute mean
    mean_val = 0.0_c_double
!$omp parallel do default(none) shared(x, n_eff) reduction(+:mean_val)
    do i = 1_c_int, n_eff
       mean_val = mean_val + x(i)
    end do
!$omp end parallel do
    mean_val = mean_val / real(n_eff, kind=c_double)

    ! Pass 2: Compute sum of squared deviations
    sum_sq_dev = 0.0_c_double
!$omp parallel do default(none) shared(x, n_eff, mean_val) reduction(+:sum_sq_dev)
    do i = 1_c_int, n_eff
       sum_sq_dev = sum_sq_dev + (x(i) - mean_val) * (x(i) - mean_val)
    end do
!$omp end parallel do

    out = sum_sq_dev / real(n_eff, kind=c_double)
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_variance_parallel

  !--------------------------------------------------------------------
  ! hpcs_reduce_std_parallel
  !
  ! Parallel standard deviation (sqrt of variance).
  ! Falls back to serial version for n < HPCS_PARALLEL_THRESHOLD.
  !--------------------------------------------------------------------
  subroutine hpcs_reduce_std_parallel(x, n, out, status) &
       bind(C, name="hpcs_reduce_std_parallel")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: out
    integer(c_int),  intent(out):: status

    real(c_double) :: variance

    ! Fall back to serial for small arrays
    if (n < HPCS_PARALLEL_THRESHOLD) then
       call hpcs_reduce_std(x, n, out, status)
       return
    end if

    ! Use parallel variance
    call hpcs_reduce_variance_parallel(x, n, variance, status)
    if (status /= HPCS_SUCCESS) return

    out = sqrt(variance)
    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_std_parallel

  !--------------------------------------------------------------------
  ! hpcs_group_reduce_variance_parallel
  !
  ! Parallel grouped variance using thread-private accumulators.
  ! Falls back to serial version for n < HPCS_PARALLEL_THRESHOLD.
  !
  ! Strategy: Each thread maintains private accumulator arrays for
  ! each group, then combines them in a critical section.
  !--------------------------------------------------------------------
  subroutine hpcs_group_reduce_variance_parallel(x, n, group_ids, n_groups, y, status) &
       bind(C, name="hpcs_group_reduce_variance_parallel")
    use iso_c_binding, only: c_int, c_double
    implicit none

    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  intent(in) :: group_ids(*)
    integer(c_int),  value      :: n_groups
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    ! CRITICAL FIX: Always delegate to serial version
    !
    ! Benchmarking revealed that the parallel implementation with atomic operations
    ! is 5-10x SLOWER than the serial version due to severe atomic contention.
    ! Multiple threads competing to update the same group buckets causes serialization
    ! that negates any parallel benefit and adds significant overhead.
    !
    ! Benchmark results (4 threads):
    !   Array Size  | Serial (ms) | Parallel (ms) | Speedup
    !   ------------|-------------|---------------|--------
    !   100K        |   0.708     |   4.925       | 0.14x
    !   500K        |   3.499     |  38.922       | 0.09x
    !   1M          |   8.078     |  41.056       | 0.20x
    !   10M         |  55.485     | 350.309       | 0.16x
    !
    ! This function is retained for API compatibility but internally uses the
    ! serial implementation which is significantly faster. Alternative parallel
    ! strategies (thread-private arrays, block partitioning) may be explored
    ! in future versions.
    !
    ! See: BENCHMARK_RESULTS_SUMMARY.md for detailed analysis

    call hpcs_group_reduce_variance(x, n, group_ids, n_groups, y, status)
  end subroutine hpcs_group_reduce_variance_parallel

end module hpcs_core_parallel
