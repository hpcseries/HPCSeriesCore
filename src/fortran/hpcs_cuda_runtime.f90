module hpcs_cuda_runtime
  !!
  !! Lightweight Fortran interface to the CUDA runtime for HPCSeries Core.
  !! Phase 1: we only expose device count and device selection.
  !!
  use iso_c_binding
  use hpcs_constants
  implicit none
  private

  ! Public API
  public :: hpcs_cuda_get_device_count
  public :: hpcs_cuda_set_device
  public :: hpcs_cuda_get_last_error

  ! CUDA success code (cudaSuccess = 0)
  integer(c_int), parameter :: CUDA_SUCCESS = 0_c_int

  interface
    !
    ! int cudaGetDeviceCount(int *count);
    !
    integer(c_int) function cudaGetDeviceCount(count) bind(C, name="cudaGetDeviceCount")
      import :: c_int
      integer(c_int) :: count
    end function cudaGetDeviceCount

    !
    ! int cudaSetDevice(int device);
    !
    integer(c_int) function cudaSetDevice(device) bind(C, name="cudaSetDevice")
      import :: c_int
      integer(c_int), value :: device
    end function cudaSetDevice

    !
    ! cudaError_t cudaGetLastError(void);
    ! We expose it as an int error code for diagnostics.
    !
    integer(c_int) function cudaGetLastError() bind(C, name="cudaGetLastError")
      import :: c_int
    end function cudaGetLastError

  end interface

contains

  subroutine hpcs_cuda_get_device_count(count, ierr)
    !! Query the number of CUDA devices via cudaGetDeviceCount.
    !! Returns HPCSeries status codes: HPCS_SUCCESS or HPCS_ERR_NUMERIC_FAIL.
    !! On error, returns count = 0.
    integer(c_int), intent(out) :: count
    integer(c_int), intent(out) :: ierr
    integer(c_int) :: cuda_status

    cuda_status = cudaGetDeviceCount(count)
    if (cuda_status == CUDA_SUCCESS) then
      ierr = HPCS_SUCCESS
    else
      ! CUDA query failed, return error status and zero devices
      count = 0_c_int
      ierr = HPCS_ERR_NUMERIC_FAIL
    end if
  end subroutine hpcs_cuda_get_device_count

  subroutine hpcs_cuda_set_device(device, ierr)
    !! Set the current CUDA device via cudaSetDevice.
    !! Returns HPCSeries status codes: HPCS_SUCCESS or HPCS_ERR_NUMERIC_FAIL.
    integer(c_int), intent(in)  :: device
    integer(c_int), intent(out) :: ierr
    integer(c_int) :: cuda_status

    cuda_status = cudaSetDevice(device)
    if (cuda_status == CUDA_SUCCESS) then
      ierr = HPCS_SUCCESS
    else
      ierr = HPCS_ERR_NUMERIC_FAIL
    end if
  end subroutine hpcs_cuda_set_device

  subroutine hpcs_cuda_get_last_error(ierr)
    !! Retrieve the last CUDA error code via cudaGetLastError.
    !! Returns raw CUDA error code (not translated to HPCSeries status).
    !! Use this for diagnostics and debugging.
    integer(c_int), intent(out) :: ierr

    ierr = cudaGetLastError()
  end subroutine hpcs_cuda_get_last_error

end module hpcs_cuda_runtime
