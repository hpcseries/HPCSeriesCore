! ==============================================================================
! HPC Series Core Library - Shared Constants Module
! Error codes and common constants used across all kernel modules
! ==============================================================================

module hpcs_constants
  use iso_c_binding, only: c_int
  implicit none
  public

  ! Error status codes (C-compatible)
  integer(c_int), parameter :: HPCS_SUCCESS          = 0_c_int
  integer(c_int), parameter :: HPCS_ERR_INVALID_ARGS = 1_c_int
  integer(c_int), parameter :: HPCS_ERR_NUMERIC_FAIL = 2_c_int

end module hpcs_constants
