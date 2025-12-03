! ==============================================================================
! HPC Series Core Library - SIMD Interface Module (v0.6)
!
! Fortran interface to C SIMD dispatch system. Provides drop-in replacements
! for scalar reduction kernels with automatic SIMD acceleration.
!
! SIMD Kernels (AVX2/AVX/SSE2 or OpenMP SIMD fallback):
!   - hpcs_reduce_sum_simd
!   - hpcs_reduce_mean_simd
!   - hpcs_reduce_min_simd
!   - hpcs_reduce_max_simd
!   - hpcs_reduce_variance_simd
!   - hpcs_reduce_std_simd
!
! Usage:
!   ! Replace:  call hpcs_reduce_sum(x, n, result, status)
!   ! With:     call hpcs_reduce_sum_simd(x, n, result, status)
!
! Performance:
!   - AVX2: Up to 4x faster (4 doubles per vector)
!   - OpenMP SIMD: Portable vectorization on all platforms
!   - Automatic ISA detection at runtime
!
! All routines:
!   - use ISO_C_BINDING with bind(C)
!   - return status via an explicit integer(c_int) argument
! ==============================================================================

module hpcs_simd_interface
  use iso_c_binding, only: c_int, c_double, c_char
  use hpcs_constants
  implicit none
  private

  ! Public SIMD reduction interfaces
  public :: hpcs_reduce_sum_simd
  public :: hpcs_reduce_mean_simd
  public :: hpcs_reduce_min_simd
  public :: hpcs_reduce_max_simd
  public :: hpcs_reduce_variance_simd
  public :: hpcs_reduce_std_simd
  public :: hpcs_simd_get_info

  ! ============================================================================
  ! C Interface Declarations
  ! ============================================================================

  interface

    !--------------------------------------------------------------------------
    ! Sum reduction - SIMD-accelerated
    !--------------------------------------------------------------------------
    subroutine hpcs_reduce_sum_simd(x, n, out, status) &
         bind(C, name="hpcs_reduce_sum_simd")
      use iso_c_binding, only: c_int, c_double
      implicit none
      real(c_double), intent(in)  :: x(*)
      integer(c_int), value       :: n
      real(c_double), intent(out) :: out
      integer(c_int), intent(out) :: status
    end subroutine hpcs_reduce_sum_simd

    !--------------------------------------------------------------------------
    ! Mean reduction - SIMD-accelerated
    !--------------------------------------------------------------------------
    subroutine hpcs_reduce_mean_simd(x, n, out, status) &
         bind(C, name="hpcs_reduce_mean_simd")
      use iso_c_binding, only: c_int, c_double
      implicit none
      real(c_double), intent(in)  :: x(*)
      integer(c_int), value       :: n
      real(c_double), intent(out) :: out
      integer(c_int), intent(out) :: status
    end subroutine hpcs_reduce_mean_simd

    !--------------------------------------------------------------------------
    ! Min reduction - SIMD-accelerated
    !--------------------------------------------------------------------------
    subroutine hpcs_reduce_min_simd(x, n, out, status) &
         bind(C, name="hpcs_reduce_min_simd")
      use iso_c_binding, only: c_int, c_double
      implicit none
      real(c_double), intent(in)  :: x(*)
      integer(c_int), value       :: n
      real(c_double), intent(out) :: out
      integer(c_int), intent(out) :: status
    end subroutine hpcs_reduce_min_simd

    !--------------------------------------------------------------------------
    ! Max reduction - SIMD-accelerated
    !--------------------------------------------------------------------------
    subroutine hpcs_reduce_max_simd(x, n, out, status) &
         bind(C, name="hpcs_reduce_max_simd")
      use iso_c_binding, only: c_int, c_double
      implicit none
      real(c_double), intent(in)  :: x(*)
      integer(c_int), value       :: n
      real(c_double), intent(out) :: out
      integer(c_int), intent(out) :: status
    end subroutine hpcs_reduce_max_simd

    !--------------------------------------------------------------------------
    ! Variance reduction - SIMD-accelerated
    !--------------------------------------------------------------------------
    subroutine hpcs_reduce_variance_simd(x, n, out, status) &
         bind(C, name="hpcs_reduce_variance_simd")
      use iso_c_binding, only: c_int, c_double
      implicit none
      real(c_double), intent(in)  :: x(*)
      integer(c_int), value       :: n
      real(c_double), intent(out) :: out
      integer(c_int), intent(out) :: status
    end subroutine hpcs_reduce_variance_simd

    !--------------------------------------------------------------------------
    ! Standard deviation reduction - SIMD-accelerated
    !--------------------------------------------------------------------------
    subroutine hpcs_reduce_std_simd(x, n, out, status) &
         bind(C, name="hpcs_reduce_std_simd")
      use iso_c_binding, only: c_int, c_double
      implicit none
      real(c_double), intent(in)  :: x(*)
      integer(c_int), value       :: n
      real(c_double), intent(out) :: out
      integer(c_int), intent(out) :: status
    end subroutine hpcs_reduce_std_simd

    !--------------------------------------------------------------------------
    ! SIMD diagnostic info
    !--------------------------------------------------------------------------
    subroutine hpcs_simd_get_info(isa_name, name_len, simd_width) &
         bind(C, name="hpcs_simd_get_info")
      use iso_c_binding, only: c_int, c_char
      implicit none
      character(kind=c_char), intent(out) :: isa_name(*)
      integer(c_int), value               :: name_len
      integer(c_int), intent(out)         :: simd_width
    end subroutine hpcs_simd_get_info

  end interface

end module hpcs_simd_interface
