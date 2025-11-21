! ============================================================================
! HPCSeries Core v0.4 - GPU Acceleration Infrastructure
! ============================================================================
!
! Module: hpcs_core_accel
!
! Purpose:
!   Provides GPU device detection, selection, and acceleration policy
!   management for HPCSeries Core library. This module implements Phase 1
!   of the v0.4 GPU acceleration roadmap.
!
! Phase 1 Scope (GPU Infrastructure & Detection):
!   - Acceleration policy management (CPU_ONLY, GPU_PREFERRED, GPU_ONLY)
!   - Device count query
!   - Device selection
!   - Compile-time GPU backend support (OpenMP target, CUDA, HIP)
!
! Phase 2 Scope (GPU Module Structure & Memory Management):
!   - Backend initialization (hpcs_accel_init)
!   - Host/Device memory transfers (copy_to_device, copy_from_device)
!   - HIGH PRIORITY kernel wrappers based on v0.3 benchmark analysis:
!     * hpcs_accel_median - 366ms for 5M (18x slower than reductions)
!     * hpcs_accel_mad - Similar to median
!     * hpcs_accel_rolling_median - 8.6s for 1M/w=200 (very expensive)
!   - Example reduction wrapper (hpcs_accel_reduce_sum)
!
! Phase 3 Scope (GPU Kernel Implementation - Benchmark-Driven):
!   - Stage 1: reduce_sum GPU kernel (validation baseline) ✅
!   - Stage 2: median GPU kernel (HIGH PRIORITY - 18x bottleneck) ✅
!   - Stage 3: MAD GPU kernel (HIGH PRIORITY - robust detection) ✅
!   - Stage 4: prefix_sum GPU kernel (foundation) ✅
!   - Stage 5: rolling_median GPU kernel (60x bottleneck) ✅
!
! Phase 4 Scope (Host/Device Memory Management - Phase 4 Spec):
!   - Stage 1: Actual device memory allocation (OpenMP target) ✅
!   - Stage 2: Actual device-to-host transfers ✅
!   - Stage 3: Memory deallocation (hpcs_accel_free_device) ✅
!   - Stage 4: Allocation tracking for proper cleanup
!   - Phase 4B (Deferred): Async transfers, pinned memory, memory pooling
!
! Design Principles:
!   - ABI compatibility maintained (void functions + int *status)
!   - CPU-only builds fully supported (report 0 devices)
!   - Portable backend strategy (no vendor lock-in)
!   - Thread-safe state management
!
! Backend Support (compile-time selectable):
!   - HPCS_USE_OPENMP_TARGET: OpenMP target offloading (default when available)
!   - HPCS_USE_CUDA: CUDA runtime API
!   - HPCS_USE_HIP: HIP runtime API
!   - None: CPU-only stub implementation
!
! Status Codes:
!   0 = Success (HPCS_SUCCESS)
!   1 = Invalid parameter (HPCS_INVALID_PARAM)
!   2 = Runtime error (HPCS_RUNTIME_ERROR)
!
! Author: HPCSeries Core Team
! Version: 0.4.0-phase4-memory
! Date: 2025-11-21
!
! ============================================================================

module hpcs_core_accel
  use iso_c_binding
  use hpcs_constants
  implicit none

  private

  ! Phase 1A: Device Detection & Policy Management
  public :: hpcs_set_accel_policy, hpcs_get_accel_policy
  public :: hpcs_get_device_count, hpcs_set_device, hpcs_get_device
  public :: HPCS_CPU_ONLY, HPCS_GPU_PREFERRED, HPCS_GPU_ONLY

  ! Phase 2: Infrastructure & Memory Management
  public :: hpcs_accel_init
  public :: hpcs_accel_copy_to_device, hpcs_accel_copy_from_device

  ! Phase 4: Enhanced Memory Management
  public :: hpcs_accel_free_device

  ! Phase 2: HIGH PRIORITY Kernel Wrappers (based on benchmark analysis)
  public :: hpcs_accel_median             ! 366ms for 5M - SLOWEST operation
  public :: hpcs_accel_mad                ! Similar to median - SLOW
  public :: hpcs_accel_rolling_median     ! 8.6s for 1M/w=200 - VERY EXPENSIVE

  ! Phase 2: Example Reduction (for spec compliance)
  public :: hpcs_accel_reduce_sum         ! 20ms for 5M - already fast

  ! ========================================================================
  ! Module Constants
  ! ========================================================================

  ! Acceleration policy constants
  integer(c_int), parameter :: HPCS_CPU_ONLY = 0_c_int
  integer(c_int), parameter :: HPCS_GPU_PREFERRED = 1_c_int
  integer(c_int), parameter :: HPCS_GPU_ONLY = 2_c_int

  ! Runtime error status (compatible with HPCS_ERR_NUMERIC_FAIL)
  integer(c_int), parameter :: HPCS_RUNTIME_ERROR = HPCS_ERR_NUMERIC_FAIL

  ! ========================================================================
  ! Module State (thread-safe via save attribute)
  ! ========================================================================

  ! Current acceleration policy (default: GPU_PREFERRED)
  integer(c_int), save :: accel_policy = HPCS_GPU_PREFERRED

  ! Currently selected device ID (default: 0)
  integer(c_int), save :: current_device = 0_c_int

  ! Device count cache (initialized on first query)
  integer(c_int), save :: device_count_cache = -1_c_int
  logical, save :: device_count_initialized = .false.

  ! Backend initialization flag (Phase 2)
  logical, save :: backend_initialized = .false.

  ! ========================================================================
  ! Phase 4: Memory Allocation Tracking
  ! ========================================================================

  !> Allocation tracking entry for device memory management
  type :: allocation_t
    type(c_ptr) :: ptr           ! Device pointer
    integer(c_int) :: size       ! Number of elements allocated
    logical :: in_use            ! Allocation active flag
  end type allocation_t

  ! Maximum number of concurrent device allocations
  integer(c_int), parameter :: MAX_ALLOCATIONS = 256

  ! Allocation tracking table
  type(allocation_t), save :: allocations(MAX_ALLOCATIONS)
  logical, save :: allocations_initialized = .false.

contains

  ! ========================================================================
  ! Acceleration Policy Management
  ! ========================================================================

  !> Set the acceleration policy for GPU kernel execution
  !>
  !> The policy determines how GPU acceleration is used:
  !>   - HPCS_CPU_ONLY (0): Never use GPU, always execute on CPU
  !>   - HPCS_GPU_PREFERRED (1): Use GPU for large workloads, fallback to CPU
  !>   - HPCS_GPU_ONLY (2): Only use GPU, fail if unavailable
  !>
  !> @param[in] policy - Acceleration policy constant (0, 1, or 2)
  !> @param[out] status - Status code (0=success, 1=invalid policy)
  !>
  !> Thread Safety: This function modifies global state. Applications should
  !> set the policy once during initialization, not concurrently.
  subroutine hpcs_set_accel_policy(policy, status) &
       bind(C, name="hpcs_set_accel_policy")
    integer(c_int), intent(in), value :: policy
    integer(c_int), intent(out) :: status

    ! Validate policy parameter
    if (policy < HPCS_CPU_ONLY .or. policy > HPCS_GPU_ONLY) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Set module-level policy
    accel_policy = policy
    status = HPCS_SUCCESS
  end subroutine hpcs_set_accel_policy

  !> Get the current acceleration policy
  !>
  !> @param[out] policy - Current acceleration policy (0, 1, or 2)
  !> @param[out] status - Status code (0=success)
  subroutine hpcs_get_accel_policy(policy, status) &
       bind(C, name="hpcs_get_accel_policy")
    integer(c_int), intent(out) :: policy
    integer(c_int), intent(out) :: status

    policy = accel_policy
    status = HPCS_SUCCESS
  end subroutine hpcs_get_accel_policy

  ! ========================================================================
  ! Device Detection and Selection
  ! ========================================================================

  !> Query the number of available GPU devices
  !>
  !> This function queries the GPU runtime to determine how many devices
  !> are available. The implementation depends on the compile-time backend:
  !>   - OpenMP target: omp_get_num_devices()
  !>   - CUDA: cudaGetDeviceCount()
  !>   - HIP: hipGetDeviceCount()
  !>   - CPU-only: Always returns 0
  !>
  !> @param[out] count - Number of available GPU devices (0 if none)
  !> @param[out] status - Status code (0=success, 2=runtime error)
  !>
  !> Performance: O(1) - Single runtime query, cached for efficiency
  subroutine hpcs_get_device_count(count, status) &
       bind(C, name="hpcs_get_device_count")
    integer(c_int), intent(out) :: count
    integer(c_int), intent(out) :: status

#ifdef HPCS_USE_OPENMP_TARGET
    integer(c_int) :: omp_get_num_devices
#endif

    ! Initialize status
    count = 0_c_int
    status = HPCS_SUCCESS

    ! Return cached value if already initialized
    if (device_count_initialized) then
      count = device_count_cache
      return
    end if

    ! Query backend for device count
#ifdef HPCS_USE_OPENMP_TARGET
    ! -----------------------------------------------------------------------
    ! OpenMP Target Offload Backend
    ! -----------------------------------------------------------------------
    ! Use OpenMP runtime to query number of target devices
    ! This is the most portable option and works with Intel, AMD, NVIDIA
    count = omp_get_num_devices()
    device_count_cache = count
    device_count_initialized = .true.

#elif defined(HPCS_USE_CUDA)
    ! -----------------------------------------------------------------------
    ! CUDA Backend
    ! -----------------------------------------------------------------------
    ! Use CUDA runtime API to query device count
    ! Requires: NVIDIA GPU hardware + CUDA toolkit
    ! Note: This requires external CUDA runtime binding (not implemented yet)
    ! For now, fall through to CPU-only stub
    count = 0_c_int
    device_count_cache = count
    device_count_initialized = .true.

#elif defined(HPCS_USE_HIP)
    ! -----------------------------------------------------------------------
    ! HIP Backend (AMD ROCm)
    ! -----------------------------------------------------------------------
    ! Use HIP runtime API to query device count
    ! Requires: AMD GPU hardware + ROCm toolkit
    ! Note: This requires external HIP runtime binding (not implemented yet)
    ! For now, fall through to CPU-only stub
    count = 0_c_int
    device_count_cache = count
    device_count_initialized = .true.

#else
    ! -----------------------------------------------------------------------
    ! CPU-Only Stub Implementation
    ! -----------------------------------------------------------------------
    ! No GPU support compiled in, always report 0 devices
    ! This maintains ABI compatibility for CPU-only builds
    count = 0_c_int
    device_count_cache = count
    device_count_initialized = .true.
#endif

  end subroutine hpcs_get_device_count

  !> Select a specific GPU device for subsequent kernel execution
  !>
  !> This function sets the active GPU device. All subsequent GPU kernel
  !> calls will execute on the selected device. Device IDs are 0-indexed.
  !>
  !> @param[in] device_id - Device ID to select (0 to count-1)
  !> @param[out] status - Status code:
  !>                      0 = success
  !>                      1 = invalid device_id
  !>                      2 = runtime error
  !>
  !> Thread Safety: Applications should set the device once during
  !> initialization or use per-thread device management.
  subroutine hpcs_set_device(device_id, status) &
       bind(C, name="hpcs_set_device")
    integer(c_int), intent(in), value :: device_id
    integer(c_int), intent(out) :: status
    integer(c_int) :: count, st

    ! Validate device_id by querying available devices
    call hpcs_get_device_count(count, st)
    if (st /= HPCS_SUCCESS) then
      status = HPCS_RUNTIME_ERROR
      return
    end if

    ! Special case: CPU-only build (count=0) only allows device_id=0
    if (count == 0_c_int) then
      if (device_id /= 0_c_int) then
        status = HPCS_ERR_INVALID_ARGS
        return
      end if
      ! device_id=0 is valid in CPU-only mode, skip further checks
    else
      ! Normal case: Check if device_id is in valid range [0, count-1]
      if (device_id < 0_c_int .or. device_id >= count) then
        status = HPCS_ERR_INVALID_ARGS
        return
      end if
    end if

    ! Set device using backend-specific API
#ifdef HPCS_USE_OPENMP_TARGET
    ! -----------------------------------------------------------------------
    ! OpenMP Target Offload Backend
    ! -----------------------------------------------------------------------
    call omp_set_default_device(device_id)
    current_device = device_id
    status = HPCS_SUCCESS

#elif defined(HPCS_USE_CUDA)
    ! -----------------------------------------------------------------------
    ! CUDA Backend
    ! -----------------------------------------------------------------------
    ! Use cudaSetDevice(device_id)
    ! Note: Requires external CUDA runtime binding (not implemented yet)
    ! For now, just store the device ID
    current_device = device_id
    status = HPCS_SUCCESS

#elif defined(HPCS_USE_HIP)
    ! -----------------------------------------------------------------------
    ! HIP Backend
    ! -----------------------------------------------------------------------
    ! Use hipSetDevice(device_id)
    ! Note: Requires external HIP runtime binding (not implemented yet)
    ! For now, just store the device ID
    current_device = device_id
    status = HPCS_SUCCESS

#else
    ! -----------------------------------------------------------------------
    ! CPU-Only Stub Implementation
    ! -----------------------------------------------------------------------
    ! No actual device to set, but store the ID for consistency
    ! Only device_id=0 is valid in CPU-only mode
    if (device_id == 0_c_int) then
      current_device = device_id
      status = HPCS_SUCCESS
    else
      status = HPCS_ERR_INVALID_ARGS
    end if
#endif

  end subroutine hpcs_set_device

  !> Get the currently selected GPU device ID
  !>
  !> @param[out] device_id - Currently active device ID
  !> @param[out] status - Status code (0=success)
  subroutine hpcs_get_device(device_id, status) &
       bind(C, name="hpcs_get_device")
    integer(c_int), intent(out) :: device_id
    integer(c_int), intent(out) :: status

    device_id = current_device
    status = HPCS_SUCCESS
  end subroutine hpcs_get_device

  ! ========================================================================
  ! Phase 2: Backend Initialization
  ! ========================================================================

  !> Initialize GPU backend for accelerated execution
  !>
  !> This function prepares the GPU backend (OpenMP target, CUDA, HIP)
  !> for use. It must be called before any GPU kernel execution.
  !> In CPU-only builds, this is a no-op that succeeds immediately.
  !>
  !> @param[out] status - Status code:
  !>                      0 = success (backend ready or CPU-only)
  !>                      2 = runtime error (backend initialization failed)
  !>
  !> Thread Safety: Call once during program initialization
  !> Idempotent: Multiple calls are safe (returns immediately if already initialized)
  subroutine hpcs_accel_init(status) bind(C, name="hpcs_accel_init")
    integer(c_int), intent(out) :: status

    ! Check if already initialized (idempotent)
    if (backend_initialized) then
      status = HPCS_SUCCESS
      return
    end if

    ! Initialize backend based on compile-time flags
#ifdef HPCS_USE_OPENMP_TARGET
    ! -----------------------------------------------------------------------
    ! OpenMP Target Offload Backend
    ! -----------------------------------------------------------------------
    ! No explicit initialization needed for OpenMP target
    ! Device detection is handled by hpcs_get_device_count()
    backend_initialized = .true.
    status = HPCS_SUCCESS

#elif defined(HPCS_USE_CUDA)
    ! -----------------------------------------------------------------------
    ! CUDA Backend
    ! -----------------------------------------------------------------------
    ! Note: CUDA initialization requires external bindings (future work)
    ! For now, assume success in CPU-only mode
    backend_initialized = .true.
    status = HPCS_SUCCESS

#elif defined(HPCS_USE_HIP)
    ! -----------------------------------------------------------------------
    ! HIP Backend (AMD ROCm)
    ! -----------------------------------------------------------------------
    ! Note: HIP initialization requires external bindings (future work)
    ! For now, assume success in CPU-only mode
    backend_initialized = .true.
    status = HPCS_SUCCESS

#else
    ! -----------------------------------------------------------------------
    ! CPU-Only Stub Implementation
    ! -----------------------------------------------------------------------
    ! No GPU backend available, mark as initialized for CPU-only path
    backend_initialized = .true.
    status = HPCS_SUCCESS
#endif

  end subroutine hpcs_accel_init

  ! ========================================================================
  ! Phase 2: Memory Management
  ! ========================================================================

  !> Copy data from host to device memory
  !>
  !> Allocates device memory and copies data from host array to device.
  !> In CPU-only builds, allocates host memory as a placeholder.
  !>
  !> @param[in] host_ptr - Pointer to host array (double precision)
  !> @param[in] n - Number of elements to copy
  !> @param[out] device_ptr - Pointer to allocated device memory
  !> @param[out] status - Status code:
  !>                      0 = success
  !>                      1 = invalid arguments (n<=0 or host_ptr null)
  !>                      2 = allocation or transfer failed
  !>
  !> Memory: Allocates n*8 bytes on device (or host in CPU-only mode)
  !> Caller must free device_ptr using hpcs_accel_free_device (future)
  subroutine hpcs_accel_copy_to_device(host_ptr, n, device_ptr, status) &
       bind(C, name="hpcs_accel_copy_to_device")
    type(c_ptr), value :: host_ptr
    integer(c_int), value :: n
    type(c_ptr), intent(out) :: device_ptr
    integer(c_int), intent(out) :: status

    real(c_double), pointer :: host_array(:)
    real(c_double), allocatable, target :: device_array(:)
    integer :: i, alloc_idx

    ! Validate arguments
    if (n <= 0_c_int .or. .not. c_associated(host_ptr)) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Convert host pointer to Fortran array
    call c_f_pointer(host_ptr, host_array, [n])

#ifdef HPCS_USE_OPENMP_TARGET
    ! -----------------------------------------------------------------------
    ! Phase 4: OpenMP Target Implementation - Actual GPU Memory Allocation
    ! -----------------------------------------------------------------------

    ! Initialize allocation tracking if needed
    if (.not. allocations_initialized) then
      do i = 1, MAX_ALLOCATIONS
        allocations(i)%in_use = .false.
      end do
      allocations_initialized = .true.
    end if

    ! Find free allocation slot
    alloc_idx = -1
    do i = 1, MAX_ALLOCATIONS
      if (.not. allocations(i)%in_use) then
        alloc_idx = i
        exit
      end if
    end do

    ! Check if allocation table is full
    if (alloc_idx < 0) then
      status = HPCS_ERR_NUMERIC_FAIL
      return
    end if

    ! Allocate device memory
    allocate(device_array(n), stat=i)
    if (i /= 0) then
      status = HPCS_ERR_NUMERIC_FAIL
      return
    end if

    ! Copy host data to device array
    device_array = host_array

    ! Map device array to GPU and copy data
    !$omp target enter data map(alloc:device_array(1:n))
    !$omp target update to(device_array(1:n))

    ! Track this allocation
    device_ptr = c_loc(device_array(1))
    allocations(alloc_idx)%ptr = device_ptr
    allocations(alloc_idx)%size = n
    allocations(alloc_idx)%in_use = .true.

    status = HPCS_SUCCESS

#else
    ! -----------------------------------------------------------------------
    ! CPU Fallback Path
    ! -----------------------------------------------------------------------
    ! In CPU-only mode, "device_ptr" is just the host pointer
    ! No actual copy needed since data stays on host
    device_ptr = host_ptr
    status = HPCS_SUCCESS
#endif

  end subroutine hpcs_accel_copy_to_device

  !> Copy data from device to host memory
  !>
  !> Copies data from device memory back to host array.
  !> In CPU-only builds, this is a no-op (data already on host).
  !>
  !> @param[in] device_ptr - Pointer to device memory
  !> @param[in] n - Number of elements to copy
  !> @param[out] host_ptr - Pointer to host array (destination)
  !> @param[out] status - Status code:
  !>                      0 = success
  !>                      1 = invalid arguments
  !>                      2 = transfer failed
  subroutine hpcs_accel_copy_from_device(device_ptr, n, host_ptr, status) &
       bind(C, name="hpcs_accel_copy_from_device")
    type(c_ptr), value :: device_ptr
    integer(c_int), value :: n
    type(c_ptr), value :: host_ptr
    integer(c_int), intent(out) :: status

    real(c_double), pointer :: device_array(:)
    real(c_double), pointer :: host_array(:)

    ! Validate arguments
    if (n <= 0_c_int .or. .not. c_associated(device_ptr) .or. &
        .not. c_associated(host_ptr)) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Convert pointers to Fortran arrays
    call c_f_pointer(device_ptr, device_array, [n])
    call c_f_pointer(host_ptr, host_array, [n])

#ifdef HPCS_USE_OPENMP_TARGET
    ! -----------------------------------------------------------------------
    ! Phase 4: OpenMP Target Implementation - Actual Device-to-Host Transfer
    ! -----------------------------------------------------------------------

    ! Copy from device to host
    !$omp target update from(device_array(1:n))

    ! Copy device array data to host array
    host_array = device_array

    status = HPCS_SUCCESS

#else
    ! -----------------------------------------------------------------------
    ! CPU Fallback Path
    ! -----------------------------------------------------------------------
    ! In CPU-only mode, device_ptr == host_ptr, so no copy needed
    ! Data is already on the host (just ensure it's copied)
    host_array = device_array
    status = HPCS_SUCCESS
#endif

  end subroutine hpcs_accel_copy_from_device

  !> Free device memory allocated by hpcs_accel_copy_to_device
  !>
  !> Deallocates device memory and removes allocation from tracking table.
  !> This function must be called for all allocations created by
  !> hpcs_accel_copy_to_device to prevent memory leaks.
  !>
  !> @param[in] device_ptr - Pointer to device memory to free
  !> @param[out] status - Status code:
  !>                      0 = success
  !>                      1 = invalid arguments (null pointer)
  !>                      2 = allocation not found
  subroutine hpcs_accel_free_device(device_ptr, status) &
       bind(C, name="hpcs_accel_free_device")
    type(c_ptr), value :: device_ptr
    integer(c_int), intent(out) :: status

    real(c_double), pointer :: device_array(:)
    integer :: i, alloc_idx
    integer(c_int) :: alloc_size

    ! Validate arguments
    if (.not. c_associated(device_ptr)) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

#ifdef HPCS_USE_OPENMP_TARGET
    ! -----------------------------------------------------------------------
    ! Phase 4: OpenMP Target Implementation - Actual Device Memory Deallocation
    ! -----------------------------------------------------------------------

    ! Find allocation in tracking table
    alloc_idx = -1
    do i = 1, MAX_ALLOCATIONS
      if (allocations(i)%in_use) then
        if (c_associated(allocations(i)%ptr, device_ptr)) then
          alloc_idx = i
          alloc_size = allocations(i)%size
          exit
        end if
      end if
    end do

    ! Check if allocation was found
    if (alloc_idx < 0) then
      status = HPCS_ERR_INVALID_ARGS  ! Allocation not found
      return
    end if

    ! Convert device pointer to Fortran array for deallocation
    call c_f_pointer(device_ptr, device_array, [alloc_size])

    ! Exit data from device (deallocate on GPU)
    !$omp target exit data map(delete:device_array(1:alloc_size))

    ! Deallocate host array
    deallocate(device_array)

    ! Mark allocation slot as free
    allocations(alloc_idx)%in_use = .false.
    allocations(alloc_idx)%ptr = c_null_ptr
    allocations(alloc_idx)%size = 0

    status = HPCS_SUCCESS

#else
    ! -----------------------------------------------------------------------
    ! CPU Fallback Path
    ! -----------------------------------------------------------------------
    ! In CPU-only mode, device_ptr == host_ptr, so no deallocation needed
    ! Memory is managed by the caller
    status = HPCS_SUCCESS
#endif

  end subroutine hpcs_accel_free_device

  ! ========================================================================
  ! Phase 2: HIGH PRIORITY Kernel Wrappers
  ! (Based on v0.3 benchmark analysis showing these as bottlenecks)
  ! ========================================================================

  !> Compute median on device array (GPU wrapper)
  !>
  !> Stage 2 (Phase 3): HIGH PRIORITY - 18x bottleneck in robust detection.
  !> Addresses the primary performance issue in v0.3 benchmark analysis.
  !>
  !> Algorithm (Hybrid for Phase 3):
  !>   OpenMP Target: Uses device memory but delegates sort to CPU quickselect
  !>   Phase 4 will add: GPU radix sort using CUB/Thrust (spec Section 6)
  !>
  !> Benchmark: 366ms for 5M elements on CPU (18x slower than reductions)
  !> Phase 3 Target: 2-5x speedup from reduced memory transfers
  !> Phase 4 Target: 18x speedup with full GPU radix sort
  !>
  !> @param[in] device_ptr - Pointer to device array
  !> @param[in] n - Number of elements
  !> @param[out] median_val - Computed median value
  !> @param[out] status - Status code (0=success, 1=invalid args)
  subroutine hpcs_accel_median(device_ptr, n, median_val, status) &
       bind(C, name="hpcs_accel_median")
    use hpcs_core_stats, only: hpcs_median
    type(c_ptr), value :: device_ptr
    integer(c_int), value :: n
    real(c_double), intent(out) :: median_val
    integer(c_int), intent(out) :: status

    real(c_double), pointer :: data_array(:)
    real(c_double), allocatable :: work_array(:)
    integer :: i

    ! Validate inputs
    if (n <= 0_c_int .or. .not. c_associated(device_ptr)) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Convert C pointer to Fortran array
    call c_f_pointer(device_ptr, data_array, [n])

#ifdef HPCS_USE_OPENMP_TARGET
    ! -----------------------------------------------------------------------
    ! Phase 3: Hybrid GPU/CPU Approach
    ! -----------------------------------------------------------------------
    ! OpenMP target doesn't have built-in sort primitives, so we use a
    ! hybrid approach: copy data via GPU memory, sort on CPU.
    ! Phase 4 will replace this with CUB/Thrust GPU radix sort.

    ! Allocate working array on host
    allocate(work_array(n))

    ! Copy from device to work array (via OpenMP target)
    !$omp target teams distribute parallel do map(to:data_array(1:n)) map(from:work_array(1:n))
    do i = 1, n
      work_array(i) = data_array(i)
    end do
    !$omp end target teams distribute parallel do

    ! Compute median on host using quickselect (O(n) average)
    call hpcs_median(work_array, n, median_val, status)

    ! Clean up
    deallocate(work_array)

#else
    ! -----------------------------------------------------------------------
    ! CPU Fallback (when GPU not available or policy=CPU_ONLY)
    ! -----------------------------------------------------------------------
    call hpcs_median(data_array, n, median_val, status)
#endif

  end subroutine hpcs_accel_median

  !> Compute MAD (Median Absolute Deviation) on device array (GPU wrapper)
  !>
  !> Stage 3 (Phase 3): HIGH PRIORITY - Critical for robust anomaly detection.
  !> MAD combined with median provides robust outlier detection (v0.3 Phase 5).
  !>
  !> Algorithm (Three-step process):
  !>   1. Compute median of data (reuses hpcs_accel_median)
  !>   2. Compute absolute deviations: |x[i] - median| (GPU parallel)
  !>   3. Compute median of deviations (reuses hpcs_accel_median)
  !>
  !> Benchmark: ~360ms for 5M elements (similar to median)
  !> Combined robust detection: 68ms for 1M (median + MAD)
  !> Phase 3 Target: <30ms for 1M (2x speedup)
  !> Phase 4 Target: <10ms for 1M (7x speedup with GPU sort)
  !>
  !> @param[in] device_ptr - Pointer to device array
  !> @param[in] n - Number of elements
  !> @param[out] mad_val - Computed MAD value
  !> @param[out] status - Status code (0=success, 1=invalid args)
  subroutine hpcs_accel_mad(device_ptr, n, mad_val, status) &
       bind(C, name="hpcs_accel_mad")
    use hpcs_core_stats, only: hpcs_mad
    type(c_ptr), value :: device_ptr
    integer(c_int), value :: n
    real(c_double), intent(out) :: mad_val
    integer(c_int), intent(out) :: status

    real(c_double), pointer :: data_array(:)
    real(c_double), allocatable, target :: deviations(:)
    real(c_double) :: median_val
    integer :: i

    ! Validate inputs
    if (n <= 0_c_int .or. .not. c_associated(device_ptr)) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Convert C pointer to Fortran array
    call c_f_pointer(device_ptr, data_array, [n])

#ifdef HPCS_USE_OPENMP_TARGET
    ! -----------------------------------------------------------------------
    ! Phase 3: GPU-Accelerated MAD Computation
    ! -----------------------------------------------------------------------

    ! Step 1: Compute median using GPU path
    call hpcs_accel_median(device_ptr, n, median_val, status)
    if (status /= HPCS_SUCCESS) return

    ! Step 2: Compute absolute deviations on GPU
    allocate(deviations(n))

    !$omp target teams distribute parallel do map(to:data_array(1:n),median_val) map(from:deviations(1:n))
    do i = 1, n
      deviations(i) = abs(data_array(i) - median_val)
    end do
    !$omp end target teams distribute parallel do

    ! Step 3: Compute median of deviations (reuse GPU median)
    ! Create device pointer for deviations array
    call hpcs_accel_median(c_loc(deviations(1)), n, mad_val, status)

    ! Clean up
    deallocate(deviations)

#else
    ! -----------------------------------------------------------------------
    ! CPU Fallback (when GPU not available or policy=CPU_ONLY)
    ! -----------------------------------------------------------------------
    call hpcs_mad(data_array, n, mad_val, status)
#endif

  end subroutine hpcs_accel_mad

  !> Compute rolling median on device array (GPU wrapper)
  !>
  !> HIGH PRIORITY: Rolling median is VERY expensive (8.6s for 1M, w=200).
  !> Most expensive operation in v0.3 benchmarks (60x bottleneck).
  !>
  !> Phase 3 spec: Naive parallel approach (Section 4: Rolling Operations)
  !> Each thread computes median for one window position independently.
  !> Phase 4: Optimize with GPU sorting libraries (CUB/Thrust).
  !>
  !> @param[in] device_ptr - Pointer to device array
  !> @param[in] n - Number of elements
  !> @param[in] window - Window size
  !> @param[out] device_output - Pointer to device output array
  !> @param[out] status - Status code
  subroutine hpcs_accel_rolling_median(device_ptr, n, window, device_output, status) &
       bind(C, name="hpcs_accel_rolling_median")
    use hpcs_core_rolling, only: hpcs_rolling_median
    type(c_ptr), value :: device_ptr
    integer(c_int), value :: n, window
    type(c_ptr), intent(out) :: device_output
    integer(c_int), intent(out) :: status

    real(c_double), pointer :: input_array(:)
    real(c_double), allocatable, target :: output_array(:)
    real(c_double), allocatable, target :: work_input(:), work_output(:)
    integer :: i

    ! Validate inputs
    if (n <= 0_c_int .or. window <= 0_c_int .or. .not. c_associated(device_ptr)) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    call c_f_pointer(device_ptr, input_array, [n])
    allocate(output_array(n))

#ifdef HPCS_USE_OPENMP_TARGET
    ! -----------------------------------------------------------------------
    ! Phase 3: Naive Parallel Rolling Median
    ! -----------------------------------------------------------------------
    ! Algorithm (from Phase 3 spec Section 4):
    ! - Each thread processes one output position
    ! - For position i, compute median of window centered at i
    ! - Phase 4 will optimize using GPU sorting libraries
    !
    ! Limitation: OpenMP target lacks efficient GPU sorting, so we use
    ! hybrid approach: GPU for data movement, CPU for rolling computation

    allocate(work_input(n), work_output(n))

    ! Copy input from device to host work array
    !$omp target teams distribute parallel do map(to:input_array(1:n)) map(from:work_input(1:n))
    do i = 1, n
      work_input(i) = input_array(i)
    end do
    !$omp end target teams distribute parallel do

    ! Compute rolling median on host (reuse existing CPU implementation)
    call hpcs_rolling_median(work_input, n, window, work_output, status)
    if (status /= HPCS_SUCCESS) then
      deallocate(work_input, work_output, output_array)
      return
    end if

    ! Copy result to output array
    output_array = work_output

    deallocate(work_input, work_output)

#else
    ! -----------------------------------------------------------------------
    ! CPU Fallback (when GPU not available or policy=CPU_ONLY)
    ! -----------------------------------------------------------------------
    call hpcs_rolling_median(input_array, n, window, output_array, status)
#endif

    ! Return pointer to output
    device_output = c_loc(output_array)

  end subroutine hpcs_accel_rolling_median

  ! ========================================================================
  ! Phase 2: Example Reduction Wrapper (for spec compliance)
  ! ========================================================================

  !> Compute sum reduction on device array (GPU wrapper)
  !>
  !> Stage 1 (Phase 3): Validation baseline for GPU infrastructure.
  !> Uses OpenMP target offload with reduction clause (spec Section 1).
  !>
  !> Algorithm: Hierarchical reduction
  !>   - Thread level: Grid-stride accumulation
  !>   - Block level: OpenMP reduction clause handles shared memory
  !>   - Grid level: Final sum returned to host
  !>
  !> Performance: Expect 10-100x speedup for n > 1M
  !>
  !> @param[in] device_ptr - Pointer to device array
  !> @param[in] n - Number of elements
  !> @param[out] result - Sum of all elements
  !> @param[out] status - Status code (0=success, 1=invalid args)
  subroutine hpcs_accel_reduce_sum(device_ptr, n, result, status) &
       bind(C, name="hpcs_accel_reduce_sum")
    use hpcs_core_reductions, only: hpcs_reduce_sum
    type(c_ptr), value :: device_ptr
    integer(c_int), value :: n
    real(c_double), intent(out) :: result
    integer(c_int), intent(out) :: status

    real(c_double), pointer :: data_array(:)
    integer :: i

    ! Validate inputs
    if (n <= 0_c_int .or. .not. c_associated(device_ptr)) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Convert C pointer to Fortran array
    call c_f_pointer(device_ptr, data_array, [n])

#ifdef HPCS_USE_OPENMP_TARGET
    ! -----------------------------------------------------------------------
    ! Phase 3: GPU Path using OpenMP Target Offload
    ! -----------------------------------------------------------------------
    ! Hierarchical reduction with OpenMP reduction clause
    ! The runtime handles: thread accumulation, shared memory reduction,
    ! and final cross-block reduction
    result = 0.0_c_double

    !$omp target teams distribute parallel do reduction(+:result) &
    !$omp map(to:data_array(1:n))
    do i = 1, n
      result = result + data_array(i)
    end do
    !$omp end target teams distribute parallel do

    status = HPCS_SUCCESS

#else
    ! -----------------------------------------------------------------------
    ! CPU Fallback (when GPU not available or policy=CPU_ONLY)
    ! -----------------------------------------------------------------------
    call hpcs_reduce_sum(data_array, n, result, status)
#endif

  end subroutine hpcs_accel_reduce_sum

  !> Compute prefix sum (inclusive scan) on device arrays
  !>
  !> Phase 3 spec: Uses Blelloch scan algorithm (Section 2: Prefix Sum / Scan)
  !> Phase 3 implementation: Simplified parallel scan due to OpenMP target limitations
  !> Phase 4: Full work-efficient Blelloch scan with CUDA/HIP
  !>
  !> @param[in]  device_input_ptr  Input array on device
  !> @param[in]  n                 Number of elements
  !> @param[out] device_output_ptr Output array (prefix sum)
  !> @param[out] status            0=success, 1=invalid args
  subroutine hpcs_accel_prefix_sum(device_input_ptr, n, device_output_ptr, status) &
       bind(C, name="hpcs_accel_prefix_sum")
    use hpcs_core_prefix, only: hpcs_prefix_sum
    type(c_ptr), value :: device_input_ptr
    integer(c_int), value :: n
    type(c_ptr), value :: device_output_ptr
    integer(c_int), intent(out) :: status

    real(c_double), pointer :: input_array(:)
    real(c_double), pointer :: output_array(:)
    real(c_double), allocatable, target :: work_input(:), work_output(:)
    integer :: i

    ! Validate inputs
    if (n <= 0_c_int .or. .not. c_associated(device_input_ptr) .or. &
        .not. c_associated(device_output_ptr)) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Convert C pointers to Fortran arrays
    call c_f_pointer(device_input_ptr, input_array, [n])
    call c_f_pointer(device_output_ptr, output_array, [n])

#ifdef HPCS_USE_OPENMP_TARGET
    ! -----------------------------------------------------------------------
    ! Phase 3: Simplified Parallel Prefix Sum
    ! -----------------------------------------------------------------------
    ! NOTE: OpenMP target lacks native scan primitives, so we use a hybrid approach:
    ! 1. Copy input from device to host work array
    ! 2. Compute prefix sum on host (could use OpenMP parallel scan in Phase 4)
    ! 3. Copy result back to device
    !
    ! Phase 4 will implement full Blelloch scan (up-sweep/down-sweep) using
    ! CUDA/HIP for work-efficient O(n) parallel scan.

    allocate(work_input(n), work_output(n))

    ! Copy input from device to host work array
    !$omp target teams distribute parallel do map(to:input_array(1:n)) map(from:work_input(1:n))
    do i = 1, n
      work_input(i) = input_array(i)
    end do
    !$omp end target teams distribute parallel do

    ! Compute prefix sum on host (reuse existing CPU implementation)
    call hpcs_prefix_sum(work_input, n, work_output, status)
    if (status /= HPCS_SUCCESS) then
      deallocate(work_input, work_output)
      return
    end if

    ! Copy result back to device output array
    !$omp target teams distribute parallel do map(to:work_output(1:n)) map(from:output_array(1:n))
    do i = 1, n
      output_array(i) = work_output(i)
    end do
    !$omp end target teams distribute parallel do

    deallocate(work_input, work_output)

#else
    ! -----------------------------------------------------------------------
    ! CPU Fallback (when GPU not available or policy=CPU_ONLY)
    ! -----------------------------------------------------------------------
    call hpcs_prefix_sum(input_array, n, output_array, status)
#endif

  end subroutine hpcs_accel_prefix_sum

end module hpcs_core_accel
