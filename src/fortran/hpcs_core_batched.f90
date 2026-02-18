! ==============================================================================
! HPC Series Core Library - Batched / Multiseries Operations (v0.4 CPU)
!
! This module implements batched 1D operations and axis-1 reductions for 2D
! arrays stored in column-major order.  Batched rolling operations apply
! existing 1D rolling kernels independently to each column.  Axis-1
! reductions compute statistics for each row across all columns.  All
! routines adhere to the HPCSeries ABI: scalar inputs by value, arrays as
! assumed-size and an integer status argument at the end.  Loops are
! structured to allow future OpenMP parallelisation over independent
! columns or rows.
!
! Functions implemented:
!   Batched Rolling Operations (Section 1.1):
!     - hpcs_rolling_sum_batched
!     - hpcs_rolling_mean_batched
!     - hpcs_rolling_median_batched
!     - hpcs_rolling_mad_batched
!
!   Axis-1 Reductions and Statistics (Section 1.2):
!     - hpcs_reduce_sum_axis1
!     - hpcs_reduce_mean_axis1
!     - hpcs_median_axis1
!     - hpcs_mad_axis1
!     - hpcs_quantile_axis1
!     - hpcs_robust_zscore_axis1
!
! Error codes:
!   HPCS_SUCCESS          : 0  (successful completion)
!   HPCS_ERR_INVALID_ARGS : 1  (invalid input arguments)
!   HPCS_ERR_NUMERIC_FAIL : 2  (degenerate numeric condition)
! ==============================================================================

module hpcs_core_batched
  use iso_c_binding,  only: c_int, c_double, c_bool
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  use hpcs_constants
  use hpcs_core_stats, only: hpcs_median, hpcs_mad, hpcs_quantile
  use hpcs_core_execution_mode, only: hpcs_get_execution_mode_internal
  use hpcs_cpu_detect  ! Hardware-aware adaptive parallelization
  implicit none
  private
  public :: hpcs_rolling_sum_batched
  public :: hpcs_rolling_mean_batched
  public :: hpcs_rolling_median_batched
  public :: hpcs_rolling_mad_batched
  public :: hpcs_reduce_sum_axis1
  public :: hpcs_reduce_mean_axis1
  public :: hpcs_median_axis1
  public :: hpcs_mad_axis1
  public :: hpcs_quantile_axis1
  public :: hpcs_robust_zscore_axis1

  ! Interface to fast C++ rolling implementations (O(n log w) using balanced tree)
  ! These provide 20-40x speedup over buffer-copy approach for large windows
  interface
    subroutine hpcs_rolling_median_fast_c(x, n, window, y, status) bind(C, name="hpcs_rolling_median_fast")
      use iso_c_binding, only: c_int, c_double
      real(c_double), intent(in)  :: x(*)
      integer(c_int),  value      :: n
      integer(c_int),  value      :: window
      real(c_double), intent(out) :: y(*)
      integer(c_int),  intent(out):: status
    end subroutine hpcs_rolling_median_fast_c

    subroutine hpcs_rolling_mad_fast_c(x, n, window, y, status) bind(C, name="hpcs_rolling_mad_fast")
      use iso_c_binding, only: c_int, c_double
      real(c_double), intent(in)  :: x(*)
      integer(c_int),  value      :: n
      integer(c_int),  value      :: window
      real(c_double), intent(out) :: y(*)
      integer(c_int),  intent(out):: status
    end subroutine hpcs_rolling_mad_fast_c
  end interface

contains

  !--------------------------------------------------------------------------
  ! hpcs_rolling_sum_batched
  !
  ! Compute rolling sum for each column of a 2D array x(n,m) stored in
  ! column-major order.  For each column j, apply the rolling sum algorithm
  ! with the specified window length.  The output y(n,m) contains the
  ! rolling sums for each column.  Column j is contiguous at x + j*n.
  ! Uses prefix sum algorithm: O(n) per column, total O(n*m).
  !
  ! Arguments (C view):
  !   x      - const double*  input array of length n*m
  !   n      - int           number of rows
  !   m      - int           number of columns
  !   window - int           window length (1 <= window <= n)
  !   y      - double*       output array of length n*m
  !   status - int*          output status code
  !--------------------------------------------------------------------------
  subroutine hpcs_rolling_sum_batched(x, n, m, window, y, status) &
       bind(C, name="hpcs_rolling_sum_batched")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, m_eff, w_eff
    integer(c_int) :: col, row, idx, start, threshold
    real(c_double), allocatable :: p_sum(:)
    real(c_double) :: sum_val

    n_eff = n
    m_eff = m
    w_eff = window
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int .or. w_eff <= 0_c_int .or. w_eff > n_eff) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Get adaptive threshold for fast rolling operations
    threshold = hpcs_cpu_get_threshold(2)  ! THRESHOLD_ROLLING_SIMPLE

    ! Loop over columns (parallelisable with OpenMP)
!$omp parallel do default(none) shared(x,y,n_eff,m_eff,w_eff,threshold) &
!$omp private(col,row,idx,start,sum_val,p_sum) if(n_eff*m_eff > threshold)
    do col = 1_c_int, m_eff
      ! Allocate prefix sum array for this column
      allocate(p_sum(0:n_eff))
      p_sum(0) = 0.0_c_double

      ! Build prefix sum for column col
      do row = 1_c_int, n_eff
        idx = row + (col - 1_c_int) * n_eff
        p_sum(row) = p_sum(row - 1_c_int) + x(idx)
      end do

      ! Compute rolling sum using prefix sums
      do row = 1_c_int, n_eff
        idx = row + (col - 1_c_int) * n_eff
        if (row < w_eff) then
          ! Not enough history; set to NaN or zero (spec says NaN or zero for undefined positions)
          y(idx) = ieee_value(0.0_c_double, ieee_quiet_nan)
        else
          start = row - w_eff
          sum_val = p_sum(row) - p_sum(start)
          y(idx) = sum_val
        end if
      end do

      deallocate(p_sum)
    end do
!$omp end parallel do
    status = HPCS_SUCCESS
  end subroutine hpcs_rolling_sum_batched

  !--------------------------------------------------------------------------
  ! hpcs_rolling_mean_batched
  !
  ! Compute rolling mean for each column of x(n,m).  Uses the rolling sum
  ! algorithm and divides by window.  Undefined positions (row < window)
  ! are set to NaN.  Complexity: O(n*m).
  !--------------------------------------------------------------------------
  subroutine hpcs_rolling_mean_batched(x, n, m, window, y, status) &
       bind(C, name="hpcs_rolling_mean_batched")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, m_eff, w_eff
    integer(c_int) :: col, row, idx, start, threshold
    real(c_double), allocatable :: p_sum(:)
    real(c_double) :: sum_val

    n_eff = n
    m_eff = m
    w_eff = window
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int .or. w_eff <= 0_c_int .or. w_eff > n_eff) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    threshold = hpcs_cpu_get_threshold(2)  ! THRESHOLD_ROLLING_SIMPLE

!$omp parallel do default(none) shared(x,y,n_eff,m_eff,w_eff,threshold) &
!$omp private(col,row,idx,start,sum_val,p_sum) if(n_eff*m_eff > threshold)
    do col = 1_c_int, m_eff
      allocate(p_sum(0:n_eff))
      p_sum(0) = 0.0_c_double

      do row = 1_c_int, n_eff
        idx = row + (col - 1_c_int) * n_eff
        p_sum(row) = p_sum(row - 1_c_int) + x(idx)
      end do

      do row = 1_c_int, n_eff
        idx = row + (col - 1_c_int) * n_eff
        if (row < w_eff) then
          y(idx) = ieee_value(0.0_c_double, ieee_quiet_nan)
        else
          start = row - w_eff
          sum_val = p_sum(row) - p_sum(start)
          y(idx) = sum_val / real(w_eff, kind=c_double)
        end if
      end do

      deallocate(p_sum)
    end do
!$omp end parallel do
    status = HPCS_SUCCESS
  end subroutine hpcs_rolling_mean_batched

  !--------------------------------------------------------------------------
  ! hpcs_rolling_median_batched
  !
  ! Compute rolling median for each column of x(n,m).  Uses the fast C++
  ! balanced-tree implementation (O(n log w)) instead of buffer copy (O(n*w)).
  ! This provides 20-40x speedup for large windows.  Complexity: O(n*m*log w).
  ! Each column is contiguous in memory, allowing direct calls to the fast 1D
  ! implementation.
  !--------------------------------------------------------------------------
  subroutine hpcs_rolling_median_batched(x, n, m, window, y, status) &
       bind(C, name="hpcs_rolling_median_batched")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, m_eff, w_eff
    integer(c_int) :: col, col_offset
    integer(c_int) :: st, threshold

    n_eff = n
    m_eff = m
    w_eff = window
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int .or. w_eff <= 0_c_int .or. w_eff > n_eff) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Get adaptive threshold based on CPU capabilities
    threshold = hpcs_cpu_get_threshold(3)  ! THRESHOLD_COMPUTE_HEAVY

    ! Process each column independently (parallelisable with OpenMP)
    ! Column j is contiguous at offset j*n, so we can call the fast 1D implementation directly
!$omp parallel do default(none) shared(x,y,n_eff,m_eff,w_eff,status,threshold) &
!$omp private(col,col_offset,st) if(n_eff*m_eff > threshold)
    do col = 1_c_int, m_eff
      col_offset = (col - 1_c_int) * n_eff + 1_c_int
      ! Call fast C++ implementation on this column (pass starting address)
      call hpcs_rolling_median_fast_c(x(col_offset), n_eff, w_eff, y(col_offset), st)
      if (st /= HPCS_SUCCESS) then
        status = st
      end if
    end do
!$omp end parallel do
    status = HPCS_SUCCESS
  end subroutine hpcs_rolling_median_batched

  !--------------------------------------------------------------------------
  ! hpcs_rolling_mad_batched
  !
  ! Compute rolling MAD (median absolute deviation) for each column of x(n,m).
  ! Uses the fast C++ implementation (O(n*w log w)) instead of naive buffer
  ! approach (O(n*w²)).  Each column is processed independently.
  ! Complexity: O(n*m*w log w).
  !--------------------------------------------------------------------------
  subroutine hpcs_rolling_mad_batched(x, n, m, window, y, status) &
       bind(C, name="hpcs_rolling_mad_batched")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    integer(c_int),  value      :: window
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, m_eff, w_eff
    integer(c_int) :: col, col_offset
    integer(c_int) :: st, max_status, threshold

    n_eff = n
    m_eff = m
    w_eff = window
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int .or. w_eff <= 0_c_int .or. w_eff > n_eff) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    max_status = HPCS_SUCCESS
    threshold = hpcs_cpu_get_threshold(3)  ! THRESHOLD_COMPUTE_HEAVY

    ! Process each column independently (parallelisable with OpenMP)
!$omp parallel do default(none) shared(x,y,n_eff,m_eff,w_eff,max_status,threshold) &
!$omp private(col,col_offset,st) if(n_eff*m_eff > threshold)
    do col = 1_c_int, m_eff
      col_offset = (col - 1_c_int) * n_eff + 1_c_int
      ! Call fast C++ implementation on this column (pass starting address)
      call hpcs_rolling_mad_fast_c(x(col_offset), n_eff, w_eff, y(col_offset), st)
      if (st > max_status) max_status = st
    end do
!$omp end parallel do
    status = max_status
  end subroutine hpcs_rolling_mad_batched

  !--------------------------------------------------------------------------
  ! hpcs_reduce_sum_axis1
  !
  ! Compute the sum along axis-1 (rows) for each row of x(n,m).  For each
  ! row i, sum across all m columns to produce y(i).  Row i elements are
  ! located at offsets i + k*n for k=0..m-1 in column-major storage.
  ! NaNs propagate: if any element in a row is NaN then y(i) becomes NaN.
  ! Complexity: O(n*m).
  !
  ! Arguments (C view):
  !   x      - const double*  input array of length n*m
  !   n      - int           number of rows
  !   m      - int           number of columns
  !   y      - double*       output array of length n
  !   status - int*          output status code
  !--------------------------------------------------------------------------
  ! hpcs_reduce_sum_axis1 (v0.8.0 dispatcher with execution mode support)
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_sum_axis1(x, n, m, y, mode, status) &
       bind(C, name="hpcs_reduce_sum_axis1")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double), intent(out) :: y(*)
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    ! Resolve execution mode
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    ! Dispatch to appropriate implementation
    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_reduce_sum_axis1_safe(x, n, m, y, status)
      case (HPCS_MODE_FAST)
        call hpcs_reduce_sum_axis1_fast(x, n, m, y, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_reduce_sum_axis1_deterministic(x, n, m, y, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_reduce_sum_axis1

  !--------------------------------------------------------------------------
  ! SAFE mode: Full NaN detection
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_sum_axis1_safe(x, n, m, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, m_eff, row, col, idx
    real(c_double) :: acc
    logical        :: has_nan

    n_eff = n
    m_eff = m
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Loop over rows (no OpenMP - memory-bound operation)
    do row = 1_c_int, n_eff
      acc = 0.0_c_double
      has_nan = .false.

      ! Sum across columns for this row
      do col = 1_c_int, m_eff
        idx = row + (col - 1_c_int) * n_eff
        if (x(idx) /= x(idx)) then  ! NaN check
          has_nan = .true.
        else
          acc = acc + x(idx)
        end if
      end do

      if (has_nan) then
        y(row) = ieee_value(0.0_c_double, ieee_quiet_nan)
      else
        y(row) = acc
      end if
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_sum_axis1_safe

  !--------------------------------------------------------------------------
  ! FAST mode: No NaN checks, maximum speed
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_sum_axis1_fast(x, n, m, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: row, col, idx
    real(c_double) :: acc

    ! No input validation - assume valid

    ! Loop over rows - no NaN checks
    do row = 1_c_int, n
      acc = 0.0_c_double

      ! Direct sum across columns
      do col = 1_c_int, m
        idx = row + (col - 1_c_int) * n
        acc = acc + x(idx)
      end do

      y(row) = acc
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_sum_axis1_fast

  !--------------------------------------------------------------------------
  ! DETERMINISTIC mode: Same as SAFE (already sequential)
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_sum_axis1_deterministic(x, n, m, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    ! Already sequential - same as SAFE
    call hpcs_reduce_sum_axis1_safe(x, n, m, y, status)
  end subroutine hpcs_reduce_sum_axis1_deterministic

  !--------------------------------------------------------------------------
  ! hpcs_reduce_mean_axis1 (v0.8.0 dispatcher with execution mode support)
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_mean_axis1(x, n, m, y, mode, status) &
       bind(C, name="hpcs_reduce_mean_axis1")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double), intent(out) :: y(*)
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: effective_mode

    ! Resolve execution mode
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    ! Dispatch to appropriate implementation
    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_reduce_mean_axis1_safe(x, n, m, y, status)
      case (HPCS_MODE_FAST)
        call hpcs_reduce_mean_axis1_fast(x, n, m, y, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_reduce_mean_axis1_deterministic(x, n, m, y, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_reduce_mean_axis1

  !--------------------------------------------------------------------------
  ! SAFE mode: Full NaN detection
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_mean_axis1_safe(x, n, m, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, m_eff, row, col, idx
    real(c_double) :: acc
    logical        :: has_nan

    n_eff = n
    m_eff = m
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Loop over rows (no OpenMP - memory-bound operation)
    do row = 1_c_int, n_eff
      acc = 0.0_c_double
      has_nan = .false.

      ! Sum across columns for this row
      do col = 1_c_int, m_eff
        idx = row + (col - 1_c_int) * n_eff
        if (x(idx) /= x(idx)) then  ! NaN check
          has_nan = .true.
        else
          acc = acc + x(idx)
        end if
      end do

      if (has_nan) then
        y(row) = ieee_value(0.0_c_double, ieee_quiet_nan)
      else
        y(row) = acc / real(m_eff, kind=c_double)
      end if
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_mean_axis1_safe

  !--------------------------------------------------------------------------
  ! FAST mode: No NaN checks, maximum speed
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_mean_axis1_fast(x, n, m, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: row, col, idx
    real(c_double) :: acc

    ! No input validation - assume valid

    ! Loop over rows - no NaN checks
    do row = 1_c_int, n
      acc = 0.0_c_double

      ! Direct sum across columns
      do col = 1_c_int, m
        idx = row + (col - 1_c_int) * n
        acc = acc + x(idx)
      end do

      y(row) = acc / real(m, kind=c_double)
    end do

    status = HPCS_SUCCESS
  end subroutine hpcs_reduce_mean_axis1_fast

  !--------------------------------------------------------------------------
  ! DETERMINISTIC mode: Same as SAFE (already sequential)
  !--------------------------------------------------------------------------
  subroutine hpcs_reduce_mean_axis1_deterministic(x, n, m, y, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    ! Already sequential - same as SAFE
    call hpcs_reduce_mean_axis1_safe(x, n, m, y, status)
  end subroutine hpcs_reduce_mean_axis1_deterministic

  !--------------------------------------------------------------------------
  ! hpcs_median_axis1
  !
  ! Compute the median along axis-1 for each row of x(n,m).  For each row,
  ! copy the m column values into a buffer and call hpcs_median.  If all
  ! values in a row are NaN, the result is NaN.  Complexity: O(n*m) using
  ! Quickselect.
  !--------------------------------------------------------------------------
  subroutine hpcs_median_axis1(x, n, m, y, mode, status) &
       bind(C, name="hpcs_median_axis1")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double), intent(out) :: y(*)
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, m_eff
    integer(c_int) :: row, col, idx
    real(c_double), allocatable :: buf(:)
    integer(c_int) :: st, threshold
    integer(c_int) :: effective_mode

    n_eff = n
    m_eff = m
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Resolve execution mode
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    threshold = hpcs_cpu_get_threshold(3)  ! THRESHOLD_COMPUTE_HEAVY

!$omp parallel do default(none) shared(x,y,n_eff,m_eff,threshold,effective_mode) &
!$omp private(row,col,idx,buf,st) if(n_eff*m_eff > threshold)
    do row = 1_c_int, n_eff
      ! Copy row values into buffer
      allocate(buf(m_eff))
      do col = 1_c_int, m_eff
        idx = row + (col - 1_c_int) * n_eff
        buf(col) = x(idx)
      end do
      ! Compute median
      call hpcs_median(buf, m_eff, y(row), effective_mode, st)
      deallocate(buf)
    end do
!$omp end parallel do

    status = HPCS_SUCCESS
  end subroutine hpcs_median_axis1

  !--------------------------------------------------------------------------
  ! hpcs_mad_axis1
  !
  ! Compute the MAD (median absolute deviation) along axis-1 for each row.
  ! Copy each row into a buffer and call hpcs_mad.  Status is aggregated:
  ! if any row has MAD ≈ 0, status = HPCS_ERR_NUMERIC_FAIL.
  !--------------------------------------------------------------------------
  subroutine hpcs_mad_axis1(x, n, m, y, mode, status) &
       bind(C, name="hpcs_mad_axis1")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double), intent(out) :: y(*)
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, m_eff
    integer(c_int) :: row, col, idx
    real(c_double), allocatable :: buf(:)
    integer(c_int) :: st, max_status, threshold
    integer(c_int) :: effective_mode

    n_eff = n
    m_eff = m
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Resolve execution mode
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    max_status = HPCS_SUCCESS
    threshold = hpcs_cpu_get_threshold(3)  ! THRESHOLD_COMPUTE_HEAVY

!$omp parallel do default(none) shared(x,y,n_eff,m_eff,max_status,threshold,effective_mode) &
!$omp private(row,col,idx,buf,st) if(n_eff*m_eff > threshold)
    do row = 1_c_int, n_eff
      allocate(buf(m_eff))
      do col = 1_c_int, m_eff
        idx = row + (col - 1_c_int) * n_eff
        buf(col) = x(idx)
      end do
      call hpcs_mad(buf, m_eff, y(row), effective_mode, st)
      if (st > max_status) max_status = st
      deallocate(buf)
    end do
!$omp end parallel do

    status = max_status
  end subroutine hpcs_mad_axis1

  !--------------------------------------------------------------------------
  ! hpcs_quantile_axis1
  !
  ! Compute the q-th quantile (0 <= q <= 1) along axis-1 for each row.
  ! For each row, copy values into a buffer and call hpcs_quantile.
  ! Complexity: O(n*m).
  !--------------------------------------------------------------------------
  subroutine hpcs_quantile_axis1(x, n, m, q, y, status) &
       bind(C, name="hpcs_quantile_axis1")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double),  value      :: q
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, m_eff
    integer(c_int) :: row, col, idx
    real(c_double), allocatable :: buf(:)
    integer(c_int) :: st, threshold

    n_eff = n
    m_eff = m
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if
    if (q < 0.0_c_double .or. q > 1.0_c_double) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    threshold = hpcs_cpu_get_threshold(3)  ! THRESHOLD_COMPUTE_HEAVY

!$omp parallel do default(none) shared(x,y,n_eff,m_eff,q,threshold) &
!$omp private(row,col,idx,buf,st) if(n_eff*m_eff > threshold)
    do row = 1_c_int, n_eff
      allocate(buf(m_eff))
      do col = 1_c_int, m_eff
        idx = row + (col - 1_c_int) * n_eff
        buf(col) = x(idx)
      end do
      call hpcs_quantile(buf, m_eff, q, y(row), HPCS_MODE_SAFE, st)
      deallocate(buf)
    end do
!$omp end parallel do

    status = HPCS_SUCCESS
  end subroutine hpcs_quantile_axis1

  !--------------------------------------------------------------------------
  ! hpcs_robust_zscore_axis1
  !
  ! Compute robust z-scores for each element in x(n,m) using per-row median
  ! and MAD.  For each row i, compute median m_i and MAD d_i.  Then for
  ! each element x(i,j), compute z(i,j) = (x(i,j) - m_i) / (d_i * scale).
  ! The scale parameter defaults to 1.4826 to approximate standard deviation.
  ! If a row has MAD ≈ 0, all z-scores for that row are set to 0 and
  ! status = HPCS_ERR_NUMERIC_FAIL.  The output array y has the same shape
  ! as x.
  !
  ! Arguments (C view):
  !   x      - const double*  input array of length n*m
  !   n      - int           number of rows
  !   m      - int           number of columns
  !   scale  - double        scaling constant (typically 1.4826)
  !   y      - double*       output array of length n*m (robust z-scores)
  !   status - int*          output status code
  !--------------------------------------------------------------------------
  subroutine hpcs_robust_zscore_axis1(x, n, m, scale, y, status) &
       bind(C, name="hpcs_robust_zscore_axis1")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double),  value      :: scale
    real(c_double), intent(out) :: y(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, m_eff
    real(c_double) :: sc
    integer(c_int) :: row, col, idx
    real(c_double), allocatable :: buf(:)
    real(c_double) :: med, mad_val, xi, z
    integer(c_int) :: st, max_status, threshold

    n_eff = n
    m_eff = m
    sc = scale
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    max_status = HPCS_SUCCESS
    threshold = hpcs_cpu_get_threshold(3)  ! THRESHOLD_COMPUTE_HEAVY

!$omp parallel do default(none) shared(x,y,n_eff,m_eff,sc,max_status,threshold) &
!$omp private(row,col,idx,buf,med,mad_val,xi,z,st) if(n_eff*m_eff > threshold)
    do row = 1_c_int, n_eff
      ! Copy row into buffer
      allocate(buf(m_eff))
      do col = 1_c_int, m_eff
        idx = row + (col - 1_c_int) * n_eff
        buf(col) = x(idx)
      end do

      ! Compute median
      call hpcs_median(buf, m_eff, med, HPCS_MODE_SAFE, st)
      if (st > max_status) max_status = st

      ! Compute MAD
      call hpcs_mad(buf, m_eff, mad_val, HPCS_MODE_SAFE, st)
      if (st == HPCS_ERR_NUMERIC_FAIL) then
        ! MAD ≈ 0: set all z-scores to 0
        do col = 1_c_int, m_eff
          idx = row + (col - 1_c_int) * n_eff
          y(idx) = 0.0_c_double
        end do
        if (st > max_status) max_status = st
        deallocate(buf)
        cycle
      elseif (st > max_status) then
        max_status = st
      end if

      ! Compute robust z-scores for this row
      do col = 1_c_int, m_eff
        idx = row + (col - 1_c_int) * n_eff
        xi = x(idx)
        if (xi /= xi) then
          ! NaN propagates
          y(idx) = xi
        else
          z = (xi - med) / (mad_val * sc)
          y(idx) = z
        end if
      end do
      deallocate(buf)
    end do
!$omp end parallel do

    status = max_status
  end subroutine hpcs_robust_zscore_axis1

end module hpcs_core_batched
