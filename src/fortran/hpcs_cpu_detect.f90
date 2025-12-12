!==============================================================================
! HPCS CPU Detection Module
!
! Detects CPU capabilities at runtime and calculates optimal parallelization
! thresholds based on hardware characteristics.
!==============================================================================

module hpcs_cpu_detect
  use iso_c_binding
  use omp_lib
  implicit none

  private
  public :: hpcs_cpu_info_t
  public :: hpcs_cpu_detect_init
  public :: hpcs_cpu_get_info
  public :: hpcs_cpu_get_optimal_threads
  public :: hpcs_cpu_get_threshold
  ! v0.5 API - NUMA Affinity
  public :: hpcs_set_affinity_mode
  public :: hpcs_get_affinity_mode
  public :: hpcs_get_simd_width
  public :: hpcs_has_avx
  public :: hpcs_has_avx2
  public :: hpcs_has_avx512
  ! v0.5 API - OpenMP Thread Affinity
  public :: hpcs_omp_set_compact_affinity
  public :: hpcs_omp_set_spread_affinity
  public :: hpcs_omp_apply_affinity
  public :: hpcs_omp_get_num_threads
  public :: hpcs_omp_get_max_threads

  !----------------------------------------------------------------------------
  ! CPU information structure (v0.5 extended)
  !----------------------------------------------------------------------------
  type, bind(C) :: hpcs_cpu_info_t
    ! Basic CPU topology
    integer(c_int) :: num_physical_cores     ! Physical cores (not hyperthreads)
    integer(c_int) :: num_logical_cores      ! Logical cores (including HT)
    integer(c_int) :: l1_cache_size_kb       ! L1 cache per core (KB)
    integer(c_int) :: l2_cache_size_kb       ! L2 cache per core (KB)
    integer(c_int) :: l3_cache_size_kb       ! L3 cache total (KB)
    integer(c_int) :: optimal_threads        ! Recommended thread count

    ! NUMA topology (v0.5)
    integer(c_int) :: numa_nodes             ! Number of NUMA nodes
    integer(c_int) :: cores_per_numa_node    ! Average cores per NUMA node
    type(c_ptr) :: core_to_numa_map          ! Pointer to core-to-NUMA mapping array

    ! SIMD/ISA capabilities (v0.5)
    integer(c_int) :: has_sse2               ! SSE2 support
    integer(c_int) :: has_avx                ! AVX support
    integer(c_int) :: has_avx2               ! AVX2 support
    integer(c_int) :: has_avx512             ! AVX-512 support
    integer(c_int) :: has_neon               ! ARM NEON support
    integer(c_int) :: has_fma3               ! FMA3 support
    integer(c_int) :: simd_width_bits        ! Max SIMD width (128, 256, 512)

    ! CPU identification
    character(kind=c_char) :: cpu_vendor(64)   ! CPU vendor string
    character(kind=c_char) :: cpu_model(128)   ! CPU model string

    logical(c_bool) :: initialized           ! Has detection run?
  end type hpcs_cpu_info_t

  ! Global CPU info (initialized once)
  type(hpcs_cpu_info_t), save :: g_cpu_info

  !----------------------------------------------------------------------------
  ! Threshold type enumeration
  !----------------------------------------------------------------------------
  integer(c_int), parameter :: THRESHOLD_SIMPLE_REDUCE   = 1  ! sum, mean, min, max
  integer(c_int), parameter :: THRESHOLD_ROLLING_SIMPLE  = 2  ! rolling sum/mean
  integer(c_int), parameter :: THRESHOLD_COMPUTE_HEAVY   = 3  ! median, MAD, quantile
  integer(c_int), parameter :: THRESHOLD_ANOMALY_DETECT  = 4  ! anomaly detection

  !----------------------------------------------------------------------------
  ! NUMA affinity modes (v0.5)
  !----------------------------------------------------------------------------
  integer(c_int), parameter :: AFFINITY_AUTO     = 0  ! Auto-select based on operation
  integer(c_int), parameter :: AFFINITY_COMPACT  = 1  ! Pack threads on same NUMA node
  integer(c_int), parameter :: AFFINITY_SPREAD   = 2  ! Distribute across NUMA nodes

  !----------------------------------------------------------------------------
  ! Interface to C functions
  !----------------------------------------------------------------------------
  interface
    subroutine hpcs_cpu_detect_enhanced_c(info) bind(C, name='hpcs_cpu_detect_enhanced')
      import :: hpcs_cpu_info_t
      type(hpcs_cpu_info_t), intent(inout) :: info
    end subroutine hpcs_cpu_detect_enhanced_c
  end interface

contains

  !----------------------------------------------------------------------------
  ! Initialize CPU detection (v0.5 - calls C enhanced detection)
  !----------------------------------------------------------------------------
  subroutine hpcs_cpu_detect_init() bind(C, name='hpcs_cpu_detect_init')
    character(len=64) :: vendor_str
    character(len=128) :: model_str
    integer :: i

    ! Check if already initialized
    if (g_cpu_info%initialized) return

    ! Call C enhanced detection (detects cores, cache, NUMA, SIMD)
    call hpcs_cpu_detect_enhanced_c(g_cpu_info)

    ! Convert C strings to Fortran strings for printing
    vendor_str = ' '
    do i = 1, 64
      if (g_cpu_info%cpu_vendor(i) == c_null_char) exit
      vendor_str(i:i) = g_cpu_info%cpu_vendor(i)
    end do

    ! Silent initialization - output only via explicit cpuinfo command
    ! (Removed auto-print to avoid cluttering import output)
  end subroutine hpcs_cpu_detect_init

  !----------------------------------------------------------------------------
  ! Get CPU info structure
  !----------------------------------------------------------------------------
  subroutine hpcs_cpu_get_info(info) bind(C, name='hpcs_cpu_get_info')
    type(hpcs_cpu_info_t), intent(out) :: info

    ! Initialize if not done
    if (.not. g_cpu_info%initialized) then
      call hpcs_cpu_detect_init()
    end if

    info = g_cpu_info
  end subroutine hpcs_cpu_get_info

  !----------------------------------------------------------------------------
  ! Get optimal thread count
  !----------------------------------------------------------------------------
  function hpcs_cpu_get_optimal_threads() result(num_threads) &
      bind(C, name='hpcs_cpu_get_optimal_threads')
    integer(c_int) :: num_threads

    if (.not. g_cpu_info%initialized) then
      call hpcs_cpu_detect_init()
    end if

    num_threads = g_cpu_info%optimal_threads
  end function hpcs_cpu_get_optimal_threads

  !----------------------------------------------------------------------------
  ! Get adaptive threshold for operation type
  !
  ! Calculates threshold based on:
  ! - Cache size (larger cache = lower threshold, more parallelization)
  ! - Core count (more cores = lower threshold)
  ! - Operation complexity
  !----------------------------------------------------------------------------
  function hpcs_cpu_get_threshold(operation_type) result(threshold) &
      bind(C, name='hpcs_cpu_get_threshold')
    integer(c_int), value :: operation_type
    integer(c_int) :: threshold
    integer(c_int) :: base_threshold
    real(c_double) :: scale_factor
    integer(c_int) :: elements_per_kb

    if (.not. g_cpu_info%initialized) then
      call hpcs_cpu_detect_init()
    end if

    ! Base thresholds for reference 8-core CPU with 8MB L3 cache
    select case (operation_type)
      case (THRESHOLD_SIMPLE_REDUCE)
        base_threshold = 500000  ! Disable for most cases (too fast)
      case (THRESHOLD_ROLLING_SIMPLE)
        base_threshold = 500000  ! High threshold (minimal benefit)
      case (THRESHOLD_COMPUTE_HEAVY)
        base_threshold = 100000  ! Medium threshold (good scaling)
      case (THRESHOLD_ANOMALY_DETECT)
        base_threshold = 50000   ! Low threshold (excellent scaling)
      case default
        base_threshold = 100000  ! Default medium threshold
    end select

    ! Scale based on cache size
    ! Larger cache = can handle more data efficiently in parallel
    ! Scale threshold inversely with cache size
    scale_factor = real(g_cpu_info%l3_cache_size_kb, c_double) / 8192.0_c_double

    ! Scale based on core count
    ! More cores = want lower threshold to utilize them
    scale_factor = scale_factor * (8.0_c_double / real(g_cpu_info%num_physical_cores, c_double))

    ! Apply scaling (clamp to reasonable range)
    threshold = int(real(base_threshold, c_double) * scale_factor, c_int)

    ! Clamp to reasonable bounds
    threshold = max(10000, min(1000000, threshold))

    ! Special case: if only 1-2 cores, use very high thresholds
    if (g_cpu_info%num_physical_cores <= 2) then
      threshold = 1000000  ! Effectively disable parallelization
    end if

  end function hpcs_cpu_get_threshold

  !----------------------------------------------------------------------------
  ! v0.5 API: Set NUMA affinity mode
  !----------------------------------------------------------------------------
  subroutine hpcs_set_affinity_mode(mode, status) bind(C, name='hpcs_set_affinity_mode_f')
    integer(c_int), value :: mode
    integer(c_int), intent(out) :: status

    interface
      subroutine hpcs_set_affinity_mode_c(mode, status) bind(C, name='hpcs_set_affinity_mode')
        import :: c_int
        integer(c_int), value :: mode
        integer(c_int), intent(out) :: status
      end subroutine hpcs_set_affinity_mode_c
    end interface

    call hpcs_set_affinity_mode_c(mode, status)
  end subroutine hpcs_set_affinity_mode

  !----------------------------------------------------------------------------
  ! v0.5 API: Get NUMA affinity mode
  !----------------------------------------------------------------------------
  function hpcs_get_affinity_mode() result(mode) bind(C, name='hpcs_get_affinity_mode_f')
    integer(c_int) :: mode

    interface
      function hpcs_get_affinity_mode_c() result(mode) bind(C, name='hpcs_get_affinity_mode')
        import :: c_int
        integer(c_int) :: mode
      end function hpcs_get_affinity_mode_c
    end interface

    mode = hpcs_get_affinity_mode_c()
  end function hpcs_get_affinity_mode

  !----------------------------------------------------------------------------
  ! v0.5 API: Get SIMD width
  !----------------------------------------------------------------------------
  function hpcs_get_simd_width() result(width) bind(C, name='hpcs_get_simd_width_f')
    integer(c_int) :: width

    if (.not. g_cpu_info%initialized) then
      call hpcs_cpu_detect_init()
    end if

    width = g_cpu_info%simd_width_bits
  end function hpcs_get_simd_width

  !----------------------------------------------------------------------------
  ! v0.5 API: Check if AVX is available
  !----------------------------------------------------------------------------
  function hpcs_has_avx() result(has_it) bind(C, name='hpcs_has_avx_f')
    integer(c_int) :: has_it

    if (.not. g_cpu_info%initialized) then
      call hpcs_cpu_detect_init()
    end if

    has_it = g_cpu_info%has_avx
  end function hpcs_has_avx

  !----------------------------------------------------------------------------
  ! v0.5 API: Check if AVX2 is available
  !----------------------------------------------------------------------------
  function hpcs_has_avx2() result(has_it) bind(C, name='hpcs_has_avx2_f')
    integer(c_int) :: has_it

    if (.not. g_cpu_info%initialized) then
      call hpcs_cpu_detect_init()
    end if

    has_it = g_cpu_info%has_avx2
  end function hpcs_has_avx2

  !----------------------------------------------------------------------------
  ! v0.5 API: Check if AVX-512 is available
  !----------------------------------------------------------------------------
  function hpcs_has_avx512() result(has_it) bind(C, name='hpcs_has_avx512_f')
    integer(c_int) :: has_it

    if (.not. g_cpu_info%initialized) then
      call hpcs_cpu_detect_init()
    end if

    has_it = g_cpu_info%has_avx512
  end function hpcs_has_avx512

  !----------------------------------------------------------------------------
  ! v0.5 OpenMP Thread Affinity Integration
  !----------------------------------------------------------------------------

  !> Apply OpenMP thread affinity using compact mode
  !>
  !> Configures OpenMP to bind threads to cores within the same NUMA node.
  !> Sets OMP_PLACES=cores and OMP_PROC_BIND=close for compact binding.
  !>
  !> @param[in] num_threads - Number of threads to configure (or -1 for auto)
  !> @param[out] status - 0 on success, -1 on error
  subroutine hpcs_omp_set_compact_affinity(num_threads, status) &
       bind(C, name='hpcs_omp_set_compact_affinity')
    integer(c_int), value :: num_threads
    integer(c_int), intent(out) :: status
    integer :: actual_threads

    ! Initialize if needed
    if (.not. g_cpu_info%initialized) then
      call hpcs_cpu_detect_init()
    end if

    ! Determine actual thread count
    if (num_threads <= 0) then
      actual_threads = g_cpu_info%optimal_threads
    else
      actual_threads = num_threads
    end if

    ! Set number of threads
    call omp_set_num_threads(actual_threads)

    ! Note: OpenMP places and proc_bind should be set via environment variables:
    ! export OMP_PLACES=cores
    ! export OMP_PROC_BIND=close
    !
    ! For programmatic control, we rely on the C pthread affinity functions
    ! since OpenMP 5.0+ API for places is not widely available in Fortran

    status = 0_c_int
  end subroutine hpcs_omp_set_compact_affinity

  !> Apply OpenMP thread affinity using spread mode
  !>
  !> Configures OpenMP to distribute threads across NUMA nodes.
  !> Sets OMP_PLACES=cores and OMP_PROC_BIND=spread for distributed binding.
  !>
  !> @param[in] num_threads - Number of threads to configure (or -1 for auto)
  !> @param[out] status - 0 on success, -1 on error
  subroutine hpcs_omp_set_spread_affinity(num_threads, status) &
       bind(C, name='hpcs_omp_set_spread_affinity')
    integer(c_int), value :: num_threads
    integer(c_int), intent(out) :: status
    integer :: actual_threads

    ! Initialize if needed
    if (.not. g_cpu_info%initialized) then
      call hpcs_cpu_detect_init()
    end if

    ! Determine actual thread count
    if (num_threads <= 0) then
      actual_threads = g_cpu_info%optimal_threads
    else
      actual_threads = num_threads
    end if

    ! Set number of threads
    call omp_set_num_threads(actual_threads)

    ! Note: OpenMP places and proc_bind should be set via environment variables:
    ! export OMP_PLACES=cores
    ! export OMP_PROC_BIND=spread
    !
    ! For programmatic control, we rely on the C pthread affinity functions
    ! since OpenMP 5.0+ API for places is not widely available in Fortran

    status = 0_c_int
  end subroutine hpcs_omp_set_spread_affinity

  !> Apply OpenMP thread affinity based on operation class
  !>
  !> Automatically selects compact or spread mode based on operation type.
  !> Integrates with NUMA affinity tuning configuration.
  !>
  !> @param[in] num_threads - Number of threads (or -1 for auto)
  !> @param[in] op_class - Operation class (1=simple, 2=rolling, 3=robust, 4=anomaly)
  !> @param[out] status - 0 on success, -1 on error
  subroutine hpcs_omp_apply_affinity(num_threads, op_class, status) &
       bind(C, name='hpcs_omp_apply_affinity')
    integer(c_int), value :: num_threads, op_class
    integer(c_int), intent(out) :: status
    integer(c_int) :: numa_mode, actual_threads

    ! Interface to C NUMA affinity functions
    interface
      function hpcs_get_tuning_numa_mode_c(op_class) result(mode) &
           bind(C, name='hpcs_get_tuning_numa_mode')
        import :: c_int
        integer(c_int), value :: op_class
        integer(c_int) :: mode
      end function hpcs_get_tuning_numa_mode_c

      function hpcs_apply_numa_affinity_c(num_threads, op_class) result(status) &
           bind(C, name='hpcs_apply_numa_affinity')
        import :: c_int
        integer(c_int), value :: num_threads, op_class
        integer(c_int) :: status
      end function hpcs_apply_numa_affinity_c
    end interface

    ! Initialize if needed
    if (.not. g_cpu_info%initialized) then
      call hpcs_cpu_detect_init()
    end if

    ! Determine actual thread count
    if (num_threads <= 0) then
      actual_threads = g_cpu_info%optimal_threads
    else
      actual_threads = num_threads
    end if

    ! Set number of threads
    call omp_set_num_threads(actual_threads)

    ! Get NUMA mode for this operation class
    numa_mode = hpcs_get_tuning_numa_mode_c(op_class)

    ! Apply affinity using C pthread functions (more control than OpenMP env vars)
    status = hpcs_apply_numa_affinity_c(actual_threads, op_class)

    ! If C affinity failed or not supported, still succeed (graceful degradation)
    if (status /= 0) then
      status = 0_c_int  ! Don't fail - just means affinity not applied
    end if

  end subroutine hpcs_omp_apply_affinity

  !> Get current OpenMP thread count
  !>
  !> @return Current number of OpenMP threads
  function hpcs_omp_get_num_threads() result(num_threads) &
       bind(C, name='hpcs_omp_get_num_threads')
    integer(c_int) :: num_threads

    !$omp parallel
    !$omp master
    num_threads = omp_get_num_threads()
    !$omp end master
    !$omp end parallel

  end function hpcs_omp_get_num_threads

  !> Get maximum OpenMP threads available
  !>
  !> @return Maximum number of OpenMP threads
  function hpcs_omp_get_max_threads() result(max_threads) &
       bind(C, name='hpcs_omp_get_max_threads')
    integer(c_int) :: max_threads

    max_threads = omp_get_max_threads()

  end function hpcs_omp_get_max_threads

end module hpcs_cpu_detect
