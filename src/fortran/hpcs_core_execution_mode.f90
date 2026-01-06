! ==============================================================================
! HPC Series Core Library - Execution Mode Management Module (v0.8.0)
! ==============================================================================
! Thread-safe runtime control for execution modes: SAFE vs FAST vs DETERMINISTIC
!
! Key Features:
!   - Global mode setting with thread-local storage (OpenMP threadprivate)
!   - Default mode: SAFE (full NaN checks, validation, IEEE 754 compliance)
!   - Zero-overhead mode query for kernel dispatch
!   - Input validation for mode setting
!
! Usage Pattern:
!   ! Set global mode
!   call hpcs_set_execution_mode(HPCS_MODE_FAST, status)
!
!   ! Kernels query mode and dispatch
!   call hpcs_ewma(x, n, alpha, y, HPCS_MODE_USE_GLOBAL, status)
! ==============================================================================

module hpcs_core_execution_mode
  use iso_c_binding, only: c_int
  use hpcs_constants, only: HPCS_SUCCESS, HPCS_ERR_INVALID_ARGS, &
                            HPCS_MODE_SAFE, HPCS_MODE_FAST, &
                            HPCS_MODE_DETERMINISTIC, HPCS_MODE_USE_GLOBAL
  implicit none
  private

  ! Public API
  public :: hpcs_set_execution_mode
  public :: hpcs_get_execution_mode
  public :: hpcs_get_execution_mode_internal

  ! ----------------------------------------------------------------------------
  ! Thread-local execution mode state
  ! ----------------------------------------------------------------------------

  ! Global execution mode (thread-local via OpenMP threadprivate)
  ! Default: SAFE mode (0)
  integer(c_int), save :: g_execution_mode = HPCS_MODE_SAFE
  !$omp threadprivate(g_execution_mode)

contains

  ! ============================================================================
  ! Public API: Set global execution mode
  ! ============================================================================

  !> Set the global execution mode for the current thread
  !!
  !! This mode is used by all kernel calls that specify HPCS_MODE_USE_GLOBAL (-1)
  !! as their mode parameter.
  !!
  !! @param[in]  mode    Execution mode: 0=SAFE, 1=FAST, 2=DETERMINISTIC
  !! @param[out] status  0=success, 1=invalid mode
  !!
  !! Thread Safety: Thread-local via OpenMP threadprivate
  !! Default: HPCS_MODE_SAFE (0)
  subroutine hpcs_set_execution_mode(mode, status) bind(C, name="hpcs_set_execution_mode")
    integer(c_int), value,  intent(in)  :: mode
    integer(c_int),         intent(out) :: status

    ! Validate mode
    if (mode /= HPCS_MODE_SAFE .and. &
        mode /= HPCS_MODE_FAST .and. &
        mode /= HPCS_MODE_DETERMINISTIC) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if

    ! Set global mode (thread-local)
    g_execution_mode = mode
    status = HPCS_SUCCESS
  end subroutine hpcs_set_execution_mode

  ! ============================================================================
  ! Public API: Get global execution mode
  ! ============================================================================

  !> Query the current thread's global execution mode
  !!
  !! @param[out] mode    Current execution mode (0, 1, or 2)
  !! @param[out] status  0=success
  !!
  !! Thread Safety: Thread-local via OpenMP threadprivate
  subroutine hpcs_get_execution_mode(mode, status) bind(C, name="hpcs_get_execution_mode")
    integer(c_int), intent(out) :: mode
    integer(c_int), intent(out) :: status

    mode = g_execution_mode
    status = HPCS_SUCCESS
  end subroutine hpcs_get_execution_mode

  ! ============================================================================
  ! Internal Helper: Fast mode query for kernel dispatch
  ! ============================================================================

  !> Internal helper to query global mode (inline, zero overhead)
  !!
  !! This function is used by kernel dispatchers to resolve the effective
  !! execution mode when HPCS_MODE_USE_GLOBAL is passed.
  !!
  !! @param[out] mode  Current global execution mode
  !!
  !! Performance: Inlined, zero overhead (direct global variable access)
  !! Thread Safety: Thread-local via OpenMP threadprivate
  pure elemental subroutine hpcs_get_execution_mode_internal(mode)
    integer(c_int), intent(out) :: mode

    mode = g_execution_mode
  end subroutine hpcs_get_execution_mode_internal

end module hpcs_core_execution_mode
