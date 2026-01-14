! ==============================================================================
! HPC Series Core Library - Robust Statistics Module (v0.3)
!
! This module provides robust statistical kernels for 1D double precision arrays
! including median, median absolute deviation (MAD) and quantile computations.
! Algorithms follow the specification in the v0.3 design document.  All
! routines are ISO_C_BINDING compatible and return status codes through an
! explicit integer argument.  Temporary allocations and internal helper
! procedures are encapsulated inside the module.  Loops are structured to
! allow trivial insertion of OpenMP pragmas in future revisions.
!
! Kernels implemented:
!   - hpcs_median
!   - hpcs_mad
!   - hpcs_quantile
!
! Error codes:
!   HPCS_SUCCESS          : 0  (successful completion)
!   HPCS_ERR_INVALID_ARGS : 1  (invalid input arguments)
!
! Dependencies:
!   - hpcs_constants      : shared status codes
!
! Each procedure in this module operates on contiguous arrays of type
! real(c_double).  Arrays are passed as assumed-size (*) with an accompanying
! integer length parameter.  Quickselect-based selection is used for median
! and quantile calculations.  NaN values propagate through the statistical
! functions as specified.
! ============================================================================

module hpcs_core_stats
  use iso_c_binding, only: c_int, c_double
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
  use hpcs_constants
  use hpcs_core_execution_mode, only: hpcs_get_execution_mode_internal
  implicit none
  private
  public :: hpcs_median
  public :: hpcs_mad
  public :: hpcs_quantile

contains

  !--------------------------------------------------------------------------
  ! hpcs_median - Dispatcher with execution mode support (v0.8.0 Phase 2)
  !
  ! Compute the median of a 1-D array.  For even-length arrays the median is
  ! defined as the average of the two middle order statistics.  NaNs in the
  ! input propagate to the result (median = NaN, status = success).  Uses a
  ! copy of the input to avoid modifying caller data.  Quickselect is used to
  ! select the k-th smallest elements in expected O(n) time.
  !
  ! Arguments (C view):
  !   x      - const double*      input array of length n
  !   n      - int               number of elements
  !   median - double*           output scalar median
  !   mode   - int               execution mode (0=SAFE, 1=FAST, 2=DETERMINISTIC, -1=use global)
  !   status - int*              output status code
  !
  ! Status codes:
  !   HPCS_SUCCESS          (0) : success
  !   HPCS_ERR_INVALID_ARGS (1) : n <= 0 or invalid mode
  !
  !--------------------------------------------------------------------------
  subroutine hpcs_median(x, n, median, mode, status) bind(C, name="hpcs_median")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: median
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status
    integer(c_int)              :: effective_mode

    ! Resolve mode
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    ! Dispatch to appropriate implementation
    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_median_safe(x, n, median, status)
      case (HPCS_MODE_FAST)
        call hpcs_median_fast(x, n, median, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_median_deterministic(x, n, median, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_median

  !--------------------------------------------------------------------------
  ! hpcs_median_safe - SAFE mode implementation
  !--------------------------------------------------------------------------
  subroutine hpcs_median_safe(x, n, median, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: median
    integer(c_int),  intent(out):: status

    integer(c_int)            :: n_eff, k1, k2
    real(c_double), allocatable :: tmp(:)
    real(c_double)            :: v1, v2
    integer(c_int)            :: i
    logical                   :: has_nan

    n_eff = n
    if (n_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Allocate a local copy of the data
    allocate(tmp(n_eff))
    do i = 1_c_int, n_eff
      tmp(i) = x(i)
    end do

    ! Check for NaN in the copy.  NaN propagates to the output.
    has_nan = .false.
    do i = 1_c_int, n_eff
      if (tmp(i) /= tmp(i)) then
        has_nan = .true.
        exit
      end if
    end do
    if (has_nan) then
      median = ieee_value(0.0_c_double, ieee_quiet_nan)
      status = HPCS_SUCCESS
      deallocate(tmp)
      return
    end if

    ! Determine positions of the middle elements (1-based indexing)
    k1 = (n_eff - 1_c_int) / 2_c_int
    k2 = n_eff / 2_c_int

    ! Quickselect the k1+1-th smallest element
    call quickselect(tmp, 1_c_int, n_eff, k1 + 1_c_int)
    v1 = tmp(k1 + 1_c_int)

    ! If n is even, select the k2+1-th smallest as well
    if (k2 /= k1) then
      call quickselect(tmp, 1_c_int, n_eff, k2 + 1_c_int)
      v2 = tmp(k2 + 1_c_int)
      median = 0.5_c_double * (v1 + v2)
    else
      median = v1
    end if

    status = HPCS_SUCCESS
    deallocate(tmp)
  end subroutine hpcs_median_safe

  !--------------------------------------------------------------------------
  ! hpcs_median_fast - FAST mode implementation (no NaN checks)
  !--------------------------------------------------------------------------
  subroutine hpcs_median_fast(x, n, median, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: median
    integer(c_int),  intent(out):: status

    integer(c_int)            :: n_eff, k1, k2
    real(c_double), allocatable :: tmp(:)
    real(c_double)            :: v1, v2
    integer(c_int)            :: i

    n_eff = n
    if (n_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Allocate and copy (no NaN checks in FAST mode)
    allocate(tmp(n_eff))
    do i = 1_c_int, n_eff
      tmp(i) = x(i)
    end do

    ! Determine positions of the middle elements
    k1 = (n_eff - 1_c_int) / 2_c_int
    k2 = n_eff / 2_c_int

    ! Quickselect the k1+1-th smallest element
    call quickselect(tmp, 1_c_int, n_eff, k1 + 1_c_int)
    v1 = tmp(k1 + 1_c_int)

    ! If n is even, select the k2+1-th smallest as well
    if (k2 /= k1) then
      call quickselect(tmp, 1_c_int, n_eff, k2 + 1_c_int)
      v2 = tmp(k2 + 1_c_int)
      median = 0.5_c_double * (v1 + v2)
    else
      median = v1
    end if

    status = HPCS_SUCCESS
    deallocate(tmp)
  end subroutine hpcs_median_fast

  !--------------------------------------------------------------------------
  ! hpcs_median_deterministic - DETERMINISTIC mode (same as SAFE, no OpenMP)
  !--------------------------------------------------------------------------
  subroutine hpcs_median_deterministic(x, n, median, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: median
    integer(c_int),  intent(out):: status

    integer(c_int)            :: n_eff, k1, k2
    real(c_double), allocatable :: tmp(:)
    real(c_double)            :: v1, v2
    integer(c_int)            :: i
    logical                   :: has_nan

    n_eff = n
    if (n_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Allocate a local copy of the data
    allocate(tmp(n_eff))
    do i = 1_c_int, n_eff
      tmp(i) = x(i)
    end do

    ! Check for NaN in the copy (deterministic NaN detection)
    has_nan = .false.
    do i = 1_c_int, n_eff
      if (tmp(i) /= tmp(i)) then
        has_nan = .true.
        exit
      end if
    end do
    if (has_nan) then
      median = ieee_value(0.0_c_double, ieee_quiet_nan)
      status = HPCS_SUCCESS
      deallocate(tmp)
      return
    end if

    ! Determine positions of the middle elements
    k1 = (n_eff - 1_c_int) / 2_c_int
    k2 = n_eff / 2_c_int

    ! Quickselect the k1+1-th smallest element
    call quickselect(tmp, 1_c_int, n_eff, k1 + 1_c_int)
    v1 = tmp(k1 + 1_c_int)

    ! If n is even, select the k2+1-th smallest as well
    if (k2 /= k1) then
      call quickselect(tmp, 1_c_int, n_eff, k2 + 1_c_int)
      v2 = tmp(k2 + 1_c_int)
      median = 0.5_c_double * (v1 + v2)
    else
      median = v1
    end if

    status = HPCS_SUCCESS
    deallocate(tmp)
  end subroutine hpcs_median_deterministic

  !--------------------------------------------------------------------------
  ! hpcs_mad - Dispatcher with execution mode support (v0.8.0 Phase 2)
  !
  ! Compute the median absolute deviation (MAD) of a 1-D array.  The MAD is
  ! defined as the median of the absolute deviations from the median of the
  ! input.  MAD = 0 is a valid result for constant data (all same values).
  ! NaNs in the input propagate to the result (mad = NaN, status = success).
  ! Uses hpcs_median for computing both the median of x and the median of
  ! deviations.
  !
  ! Arguments (C view):
  !   x    - const double*      input array of length n
  !   n    - int               number of elements
  !   mad  - double*           output scalar MAD
  !   mode - int               execution mode (0=SAFE, 1=FAST, 2=DETERMINISTIC, -1=use global)
  !   st   - int*              output status code
  !
  ! Status codes:
  !   HPCS_SUCCESS          (0) : success
  !   HPCS_ERR_INVALID_ARGS (1) : n <= 0 or invalid mode
  !--------------------------------------------------------------------------
  subroutine hpcs_mad(x, n, mad, mode, status) bind(C, name="hpcs_mad")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: mad
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status
    integer(c_int)              :: effective_mode

    ! Resolve mode
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    ! Dispatch to appropriate implementation
    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_mad_safe(x, n, mad, status)
      case (HPCS_MODE_FAST)
        call hpcs_mad_fast(x, n, mad, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_mad_deterministic(x, n, mad, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_mad

  !--------------------------------------------------------------------------
  ! hpcs_mad_safe - SAFE mode implementation
  !--------------------------------------------------------------------------
  subroutine hpcs_mad_safe(x, n, mad, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: mad
    integer(c_int),  intent(out):: status

    integer(c_int)             :: n_eff, st
    real(c_double)             :: med
    real(c_double), allocatable :: dev(:)
    integer(c_int)             :: i
    logical                    :: has_nan

    n_eff = n
    if (n_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Compute the median; this also handles NaN propagation
    call hpcs_median(x, n_eff, med, HPCS_MODE_SAFE, st)
    if (st /= HPCS_SUCCESS) then
      ! invalid args propagate directly
      mad    = 0.0_c_double
      status = st
      return
    end if

    ! If median is NaN, then MAD is also NaN (propagate) and status = success
    if (med /= med) then
      mad    = med
      status = HPCS_SUCCESS
      return
    end if

    ! Allocate deviation array
    allocate(dev(n_eff))
    has_nan = .false.
    do i = 1_c_int, n_eff
      dev(i) = abs(x(i) - med)
      if (dev(i) /= dev(i)) then
        has_nan = .true.
      end if
    end do
    if (has_nan) then
      ! propagate NaN in deviations
      mad    = ieee_value(0.0_c_double, ieee_quiet_nan)
      status = HPCS_SUCCESS
      deallocate(dev)
      return
    end if

    ! Median of deviations
    call hpcs_median(dev, n_eff, mad, HPCS_MODE_SAFE, st)
    if (st /= HPCS_SUCCESS) then
      mad    = 0.0_c_double
      status = st
      deallocate(dev)
      return
    end if

    ! MAD computed successfully (MAD = 0 is valid for constant data)
    status = HPCS_SUCCESS
    deallocate(dev)
  end subroutine hpcs_mad_safe

  !--------------------------------------------------------------------------
  ! hpcs_mad_fast - FAST mode implementation (no NaN checks)
  !--------------------------------------------------------------------------
  subroutine hpcs_mad_fast(x, n, mad, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: mad
    integer(c_int),  intent(out):: status

    integer(c_int)             :: n_eff, st
    real(c_double)             :: med
    real(c_double), allocatable :: dev(:)
    integer(c_int)             :: i

    n_eff = n
    if (n_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Compute the median (no NaN checks in FAST mode)
    call hpcs_median(x, n_eff, med, HPCS_MODE_FAST, st)
    if (st /= HPCS_SUCCESS) then
      mad    = 0.0_c_double
      status = st
      return
    end if

    ! Allocate deviation array (no NaN checks)
    allocate(dev(n_eff))
    do i = 1_c_int, n_eff
      dev(i) = abs(x(i) - med)
    end do

    ! Median of deviations
    call hpcs_median(dev, n_eff, mad, HPCS_MODE_FAST, st)
    if (st /= HPCS_SUCCESS) then
      mad    = 0.0_c_double
      status = st
      deallocate(dev)
      return
    end if

    status = HPCS_SUCCESS
    deallocate(dev)
  end subroutine hpcs_mad_fast

  !--------------------------------------------------------------------------
  ! hpcs_mad_deterministic - DETERMINISTIC mode (same as SAFE)
  !--------------------------------------------------------------------------
  subroutine hpcs_mad_deterministic(x, n, mad, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double), intent(out) :: mad
    integer(c_int),  intent(out):: status

    integer(c_int)             :: n_eff, st
    real(c_double)             :: med
    real(c_double), allocatable :: dev(:)
    integer(c_int)             :: i
    logical                    :: has_nan

    n_eff = n
    if (n_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Compute the median; this also handles NaN propagation
    call hpcs_median(x, n_eff, med, HPCS_MODE_DETERMINISTIC, st)
    if (st /= HPCS_SUCCESS) then
      mad    = 0.0_c_double
      status = st
      return
    end if

    ! If median is NaN, then MAD is also NaN
    if (med /= med) then
      mad    = med
      status = HPCS_SUCCESS
      return
    end if

    ! Allocate deviation array
    allocate(dev(n_eff))
    has_nan = .false.
    do i = 1_c_int, n_eff
      dev(i) = abs(x(i) - med)
      if (dev(i) /= dev(i)) then
        has_nan = .true.
      end if
    end do
    if (has_nan) then
      mad    = ieee_value(0.0_c_double, ieee_quiet_nan)
      status = HPCS_SUCCESS
      deallocate(dev)
      return
    end if

    ! Median of deviations
    call hpcs_median(dev, n_eff, mad, HPCS_MODE_DETERMINISTIC, st)
    if (st /= HPCS_SUCCESS) then
      mad    = 0.0_c_double
      status = st
      deallocate(dev)
      return
    end if

    status = HPCS_SUCCESS
    deallocate(dev)
  end subroutine hpcs_mad_deterministic

  !--------------------------------------------------------------------------
  ! hpcs_quantile - Dispatcher with execution mode support (v0.8.0 Phase 2)
  !
  ! Compute the q-th quantile (0 <= q <= 1) of a 1-D array using the Type 7
  ! definition (linear interpolation between order statistics).  For q=0 the
  ! minimum is returned; for q=1 the maximum is returned.  NaNs in the input
  ! propagate to the result (value = NaN, status = success).  The function
  ! copies the input into a temporary array and uses Quickselect to find the
  ! required order statistics.  Invalid q (<0 or >1) or n<=0 triggers
  ! HPCS_ERR_INVALID_ARGS.
  !
  ! Arguments (C view):
  !   x     - const double*    input array of length n
  !   n     - int             number of elements
  !   q     - double          quantile in [0,1]
  !   value - double*         output quantile value
  !   mode  - int             execution mode (0=SAFE, 1=FAST, 2=DETERMINISTIC, -1=use global)
  !   st    - int*            output status code
  !
  ! Status codes:
  !   HPCS_SUCCESS          (0) : success
  !   HPCS_ERR_INVALID_ARGS (1) : n <= 0 or q outside [0,1] or invalid mode
  !--------------------------------------------------------------------------
  subroutine hpcs_quantile(x, n, q, value, mode, status) bind(C, name="hpcs_quantile")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double),  value      :: q
    real(c_double), intent(out) :: value
    integer(c_int),  value      :: mode
    integer(c_int),  intent(out):: status
    integer(c_int)              :: effective_mode

    ! Resolve mode
    if (mode == HPCS_MODE_USE_GLOBAL) then
      call hpcs_get_execution_mode_internal(effective_mode)
    else
      effective_mode = mode
    end if

    ! Dispatch to appropriate implementation
    select case (effective_mode)
      case (HPCS_MODE_SAFE)
        call hpcs_quantile_safe(x, n, q, value, status)
      case (HPCS_MODE_FAST)
        call hpcs_quantile_fast(x, n, q, value, status)
      case (HPCS_MODE_DETERMINISTIC)
        call hpcs_quantile_deterministic(x, n, q, value, status)
      case default
        status = HPCS_ERR_INVALID_ARGS
    end select
  end subroutine hpcs_quantile

  !--------------------------------------------------------------------------
  ! hpcs_quantile_safe - SAFE mode implementation
  !--------------------------------------------------------------------------
  subroutine hpcs_quantile_safe(x, n, q, value, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double),  value      :: q
    real(c_double), intent(out) :: value
    integer(c_int),  intent(out):: status

    integer(c_int)             :: n_eff, k, k2, st
    real(c_double)             :: h, delta, x_k, x_k1
    real(c_double), allocatable :: tmp(:)
    integer(c_int)             :: i
    logical                    :: has_nan

    n_eff = n
    if (n_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if
    if (q < 0.0_c_double .or. q > 1.0_c_double) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Copy input data
    allocate(tmp(n_eff))
    do i = 1_c_int, n_eff
      tmp(i) = x(i)
    end do
    ! Check for NaN
    has_nan = .false.
    do i = 1_c_int, n_eff
      if (tmp(i) /= tmp(i)) then
        has_nan = .true.
        exit
      end if
    end do
    if (has_nan) then
      value  = ieee_value(0.0_c_double, ieee_quiet_nan)
      status = HPCS_SUCCESS
      deallocate(tmp)
      return
    end if

    ! Compute the fractional index h (1-based indexing)
    h = (real(n_eff - 1_c_int, c_double) * q) + 1.0_c_double
    k = floor(h)
    delta = h - real(k, c_double)

    ! Boundaries: ensure k is within [1, n]
    if (k < 1_c_int) then
      k2 = 1_c_int
      call quickselect(tmp, 1_c_int, n_eff, k2)
      x_k  = tmp(k2)
      x_k1 = x_k
    else if (k >= n_eff) then
      k2 = n_eff
      call quickselect(tmp, 1_c_int, n_eff, k2)
      x_k  = tmp(k2)
      x_k1 = x_k
    else
      ! get k-th and (k+1)-th order statistics
      call quickselect(tmp, 1_c_int, n_eff, k)
      x_k = tmp(k)
      call quickselect(tmp, 1_c_int, n_eff, k + 1_c_int)
      x_k1 = tmp(k + 1_c_int)
    end if

    ! Interpolate if necessary
    if (delta <= 0.0_c_double) then
      value = x_k
    else
      value = x_k + delta * (x_k1 - x_k)
    end if

    status = HPCS_SUCCESS
    deallocate(tmp)
  end subroutine hpcs_quantile_safe

  !--------------------------------------------------------------------------
  ! hpcs_quantile_fast - FAST mode implementation (no NaN checks)
  !--------------------------------------------------------------------------
  subroutine hpcs_quantile_fast(x, n, q, value, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double),  value      :: q
    real(c_double), intent(out) :: value
    integer(c_int),  intent(out):: status

    integer(c_int)             :: n_eff, k, k2, st
    real(c_double)             :: h, delta, x_k, x_k1
    real(c_double), allocatable :: tmp(:)
    integer(c_int)             :: i

    n_eff = n
    if (n_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if
    if (q < 0.0_c_double .or. q > 1.0_c_double) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Copy input data (no NaN check in FAST mode)
    allocate(tmp(n_eff))
    do i = 1_c_int, n_eff
      tmp(i) = x(i)
    end do

    ! Compute the fractional index h (1-based indexing)
    h = (real(n_eff - 1_c_int, c_double) * q) + 1.0_c_double
    k = floor(h)
    delta = h - real(k, c_double)

    ! Boundaries: ensure k is within [1, n]
    if (k < 1_c_int) then
      k2 = 1_c_int
      call quickselect(tmp, 1_c_int, n_eff, k2)
      x_k  = tmp(k2)
      x_k1 = x_k
    else if (k >= n_eff) then
      k2 = n_eff
      call quickselect(tmp, 1_c_int, n_eff, k2)
      x_k  = tmp(k2)
      x_k1 = x_k
    else
      ! get k-th and (k+1)-th order statistics
      call quickselect(tmp, 1_c_int, n_eff, k)
      x_k = tmp(k)
      call quickselect(tmp, 1_c_int, n_eff, k + 1_c_int)
      x_k1 = tmp(k + 1_c_int)
    end if

    ! Interpolate if necessary
    if (delta <= 0.0_c_double) then
      value = x_k
    else
      value = x_k + delta * (x_k1 - x_k)
    end if

    status = HPCS_SUCCESS
    deallocate(tmp)
  end subroutine hpcs_quantile_fast

  !--------------------------------------------------------------------------
  ! hpcs_quantile_deterministic - DETERMINISTIC mode (NaN checks, bit-exact)
  !--------------------------------------------------------------------------
  subroutine hpcs_quantile_deterministic(x, n, q, value, status)
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    real(c_double),  value      :: q
    real(c_double), intent(out) :: value
    integer(c_int),  intent(out):: status

    integer(c_int)             :: n_eff, k, k2, st
    real(c_double)             :: h, delta, x_k, x_k1
    real(c_double), allocatable :: tmp(:)
    integer(c_int)             :: i
    logical                    :: has_nan

    n_eff = n
    if (n_eff <= 0_c_int) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if
    if (q < 0.0_c_double .or. q > 1.0_c_double) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Copy input data
    allocate(tmp(n_eff))
    do i = 1_c_int, n_eff
      tmp(i) = x(i)
    end do
    ! Check for NaN
    has_nan = .false.
    do i = 1_c_int, n_eff
      if (tmp(i) /= tmp(i)) then
        has_nan = .true.
        exit
      end if
    end do
    if (has_nan) then
      value  = ieee_value(0.0_c_double, ieee_quiet_nan)
      status = HPCS_SUCCESS
      deallocate(tmp)
      return
    end if

    ! Compute the fractional index h (1-based indexing)
    h = (real(n_eff - 1_c_int, c_double) * q) + 1.0_c_double
    k = floor(h)
    delta = h - real(k, c_double)

    ! Boundaries: ensure k is within [1, n]
    if (k < 1_c_int) then
      k2 = 1_c_int
      call quickselect(tmp, 1_c_int, n_eff, k2)
      x_k  = tmp(k2)
      x_k1 = x_k
    else if (k >= n_eff) then
      k2 = n_eff
      call quickselect(tmp, 1_c_int, n_eff, k2)
      x_k  = tmp(k2)
      x_k1 = x_k
    else
      ! get k-th and (k+1)-th order statistics
      call quickselect(tmp, 1_c_int, n_eff, k)
      x_k = tmp(k)
      call quickselect(tmp, 1_c_int, n_eff, k + 1_c_int)
      x_k1 = tmp(k + 1_c_int)
    end if

    ! Interpolate if necessary
    if (delta <= 0.0_c_double) then
      value = x_k
    else
      value = x_k + delta * (x_k1 - x_k)
    end if

    status = HPCS_SUCCESS
    deallocate(tmp)
  end subroutine hpcs_quantile_deterministic

  !--------------------------------------------------------------------------
  ! Internal helper: quickselect
  !
  ! Select the k-th smallest element in arr(left:right) using Hoare's
  ! partitioning algorithm with a median-of-three pivot.  The array is
  ! modified in-place; on return, arr(k) contains the k-th order statistic.
  ! Arguments:
  !   arr  - real(c_double) array (modified)
  !   left - integer(c_int) left index (1-based inclusive)
  !   right- integer(c_int) right index (1-based inclusive)
  !   k    - integer(c_int) target order (1-based)
  !
  ! This routine assumes that left <= k <= right and that arr contains no NaN
  ! values.  It is not exposed via the module's public interface.
  !--------------------------------------------------------------------------
  subroutine quickselect(arr, left, right, k)
    implicit none
    real(c_double), intent(inout) :: arr(*)
    integer(c_int), intent(in)    :: left
    integer(c_int), intent(in)    :: right
    integer(c_int), intent(in)    :: k
    integer(c_int)               :: l, r, i, j, pivot_index
    real(c_double)               :: pivot
    real(c_double)               :: tmp

    l = left
    r = right
    do while (l < r)
      ! Choose a pivot via median-of-three
      pivot_index = median_of_three(arr, l, r)
      pivot = arr(pivot_index)
      i = l
      j = r
      ! Hoare partitioning
      do
        do while (arr(i) < pivot)
          i = i + 1
        end do
        do while (arr(j) > pivot)
          j = j - 1
        end do
        if (i <= j) then
          ! swap arr(i) and arr(j)
          tmp = arr(i)
          arr(i) = arr(j)
          arr(j) = tmp
          i = i + 1
          j = j - 1
        else
          exit
        end if
      end do
      ! Narrow the search range
      if (j < k) then
        l = i
      end if
      if (k < i) then
        r = j
      end if
    end do
  end subroutine quickselect

  !--------------------------------------------------------------------------
  ! Internal helper: median_of_three
  !
  ! Return the index of the median of arr(left), arr(mid), arr(right).
  !--------------------------------------------------------------------------
  integer(c_int) function median_of_three(arr, left, right) result(pivot_index)
    implicit none
    real(c_double), intent(inout) :: arr(*)
    integer(c_int), intent(in)    :: left
    integer(c_int), intent(in)    :: right
    integer(c_int)               :: mid

    mid = left + (right - left) / 2_c_int
    ! Determine median of three values
    if (arr(left) < arr(mid)) then
      if (arr(mid) < arr(right)) then
        pivot_index = mid
      else if (arr(left) < arr(right)) then
        pivot_index = right
      else
        pivot_index = left
      end if
    else
      if (arr(left) < arr(right)) then
        pivot_index = left
      else if (arr(mid) < arr(right)) then
        pivot_index = right
      else
        pivot_index = mid
      end if
    end if
  end function median_of_three

end module hpcs_core_stats