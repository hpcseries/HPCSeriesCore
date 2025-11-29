! ==============================================================================
! HPC Series Core Library - Structured Anomaly Detection (v0.4 CPU)
!
! This module implements axis‑aware anomaly detection workflows based on
! z‑scores and their robust counterparts.  Functions operate on 2D arrays
! stored in column‑major order, computing per‑row statistics to flag
! anomalous values exceeding a user‑supplied threshold.  A rolling
! anomaly detector over the time axis (axis‑0) is also provided.  All
! routines adhere to the HPCSeries ABI: scalar inputs by value, arrays
! as assumed‑size and an integer status argument at the end.  Internal
! loops avoid deep call chains and are laid out for future OpenMP
! parallelisation.
! ============================================================================

module hpcs_core_anomaly
  use iso_c_binding,  only: c_int, c_double, c_bool
  use hpcs_constants
  use hpcs_core_stats, only: hpcs_median, hpcs_mad
  use hpcs_cpu_detect  ! Hardware-aware adaptive parallelization
  implicit none
  private
  public :: hpcs_detect_anomalies_axis1
  public :: hpcs_detect_anomalies_robust_axis1
  public :: hpcs_rolling_detect_anomalies_2d

contains

  !--------------------------------------------------------------------------
  ! hpcs_detect_anomalies_axis1
  !
  ! Perform anomaly detection along axis‑1 (rows) of a 2D array x(n,m)
  ! using mean and standard deviation.  For each row i, the mean μ_i and
  ! unbiased standard deviation σ_i are computed across all non‑NaN values
  ! x(i,j).  Then z‑scores z(i,j) = (x(i,j) - μ_i) / σ_i are evaluated.
  ! If |z(i,j)| > threshold, the corresponding entry in the output mask
  ! array is set to 1; otherwise 0.  NaN values in x produce mask=0.
  ! Rows with fewer than two valid values or zero variance yield σ_i=0;
  ! all mask entries for that row are set to 0 and status=2.  Invalid
  ! arguments (n<=0, m<=0 or threshold<0) produce status=1.
  ! The mask array is of length n*m and uses integer(c_int) values.
  !--------------------------------------------------------------------------
  subroutine hpcs_detect_anomalies_axis1(x, n, m, threshold, mask, status) &
       bind(C, name="hpcs_detect_anomalies_axis1")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double),  value      :: threshold
    integer(c_int), intent(out) :: mask(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, m_eff
    real(c_double) :: th
    integer(c_int) :: row, col, idx
    real(c_double) :: mean, M2, delta, delta2, var, std, xi
    integer(c_int) :: count
    integer(c_int) :: max_status, par_threshold

    n_eff = n
    m_eff = m
    th = threshold
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int .or. th < 0.0_c_double) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if
    max_status = HPCS_SUCCESS
    par_threshold = hpcs_cpu_get_threshold(4)  ! THRESHOLD_ANOMALY_DETECT

    ! Loop over rows
!$omp parallel do default(none) shared(x,mask,n_eff,m_eff,th,max_status,par_threshold) &
!$omp private(row,col,idx,mean,M2,delta,delta2,var,std,xi,count) if(n_eff*m_eff > par_threshold)
    do row = 1_c_int, n_eff
      mean = 0.0_c_double
      M2   = 0.0_c_double
      count = 0_c_int
      ! First pass: compute mean and M2 via Welford
      do col = 1_c_int, m_eff
        idx = row + (col - 1_c_int) * n_eff
        xi = x(idx)
        if (xi == xi) then
          count = count + 1_c_int
          delta = xi - mean
          mean = mean + delta / real(count, kind=c_double)
          delta2 = xi - mean
          M2 = M2 + delta * delta2
        end if
      end do
      if (count < 2_c_int) then
        ! degenerate: not enough valid values
        do col = 1_c_int, m_eff
          idx = row + (col - 1_c_int) * n_eff
          mask(idx) = 0_c_int
        end do
        if (HPCS_ERR_NUMERIC_FAIL > max_status) max_status = HPCS_ERR_NUMERIC_FAIL
        cycle
      end if
      var = M2 / real(count - 1_c_int, kind=c_double)
      if (var < 0.0_c_double) var = 0.0_c_double
      std = sqrt(var)
      if (std == 0.0_c_double) then
        ! zero variance: no anomalies by definition
        do col = 1_c_int, m_eff
          idx = row + (col - 1_c_int) * n_eff
          mask(idx) = 0_c_int
        end do
        if (HPCS_ERR_NUMERIC_FAIL > max_status) max_status = HPCS_ERR_NUMERIC_FAIL
        cycle
      end if
      ! Second pass: compute z‑scores and flag anomalies
      do col = 1_c_int, m_eff
        idx = row + (col - 1_c_int) * n_eff
        xi = x(idx)
        if (xi /= xi) then
          mask(idx) = 0_c_int
        else
          if (abs((xi - mean) / std) > th) then
            mask(idx) = 1_c_int
          else
            mask(idx) = 0_c_int
          end if
        end if
      end do
    end do
!$omp end parallel do
    status = max_status
  end subroutine hpcs_detect_anomalies_axis1

  !--------------------------------------------------------------------------
  ! hpcs_detect_anomalies_robust_axis1
  !
  ! Perform robust anomaly detection along axis‑1 using the median and
  ! median absolute deviation (MAD).  For each row i, the median m_i and
  ! MAD d_i are computed from the non‑NaN values in x(i,:).  A scaling
  ! constant c defaults to 1.4826 to approximate standard deviation but can
  ! be overridden by an optional argument.  An element x(i,j) is flagged
  ! anomalous if |x(i,j) - m_i| / (d_i*c) > threshold.  If d_i≈0 or
  ! there are no valid values, the entire row is marked non‑anomalous and
  ! status=2.  Invalid arguments (n<=0, m<=0 or threshold<0) produce
  ! status=1.  The mask array must have length n*m and uses integer
  ! values (0 or 1).
  !--------------------------------------------------------------------------
  subroutine hpcs_detect_anomalies_robust_axis1(x, n, m, threshold, mask, status, scale) &
       bind(C, name="hpcs_detect_anomalies_robust_axis1")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    real(c_double),  value      :: threshold
    integer(c_int), intent(out) :: mask(*)
    integer(c_int),  intent(out):: status
    real(c_double), intent(in), optional :: scale

    integer(c_int) :: n_eff, m_eff
    real(c_double) :: th
    real(c_double) :: c
    integer(c_int) :: row, col, idx
    real(c_double), allocatable :: buf(:)
    real(c_double) :: med, mad_val, xi, rz
    integer(c_int) :: st
    integer(c_int) :: max_status, par_threshold

    n_eff = n
    m_eff = m
    th = threshold
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int .or. th < 0.0_c_double) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if
    c = 1.4826_c_double
    if (present(scale)) c = scale
    max_status = HPCS_SUCCESS
    par_threshold = hpcs_cpu_get_threshold(4)  ! THRESHOLD_ANOMALY_DETECT

    ! Loop over rows
!$omp parallel do default(none) shared(x,mask,n_eff,m_eff,th,c,max_status,par_threshold) &
!$omp private(row,col,idx,buf,med,mad_val,xi,rz,st) if(n_eff*m_eff > par_threshold)
    do row = 1_c_int, n_eff
      ! Copy row into buffer
      allocate(buf(m_eff))
      do col = 1_c_int, m_eff
        idx = row + (col - 1_c_int) * n_eff
        buf(col) = x(idx)
      end do
      ! Compute median
      call hpcs_median(buf, m_eff, med, st)
      if (st > max_status) max_status = st
      ! Compute MAD
      call hpcs_mad(buf, m_eff, mad_val, st)
      if (st == HPCS_ERR_NUMERIC_FAIL) then
        ! degeneracy: fill mask with zeros
        do col = 1_c_int, m_eff
          idx = row + (col - 1_c_int) * n_eff
          mask(idx) = 0_c_int
        end do
        if (st > max_status) max_status = st
        deallocate(buf)
        cycle
      elseif (st > max_status) then
        max_status = st
      end if
      ! Mark anomalies per element
      do col = 1_c_int, m_eff
        idx = row + (col - 1_c_int) * n_eff
        xi = buf(col)
        if (xi /= xi) then
          mask(idx) = 0_c_int
        else
          rz = (xi - med) / (mad_val * c)
          if (abs(rz) > th) then
            mask(idx) = 1_c_int
          else
            mask(idx) = 0_c_int
          end if
        end if
      end do
      deallocate(buf)
    end do
!$omp end parallel do
    status = max_status
  end subroutine hpcs_detect_anomalies_robust_axis1

  !--------------------------------------------------------------------------
  ! hpcs_rolling_detect_anomalies_2d
  !
  ! Perform rolling anomaly detection over the time axis (rows) for a 2D
  ! series x(n,m).  For each column j, a rolling mean and standard
  ! deviation with window w are computed on the non‑NaN values.  For
  ! positions i >= w, the z‑score z(i,j) = (x(i,j) - mean_{i,j}) / std_{i,j}
  ! is formed.  If |z(i,j)| > threshold, mask(i,j) = 1; otherwise 0.
  ! Positions i < w or where the rolling window contains fewer than two
  ! valid values yield mask=0.  The status is set to 2 if any window
  ! encountered zero variance.  Invalid arguments (n<=0, m<=0, window<=0
  ! or window>n, threshold<0) result in status=1.  The mask array must
  ! have length n*m and is filled with zeros and ones.
  !--------------------------------------------------------------------------
  subroutine hpcs_rolling_detect_anomalies_2d(x, n, m, window, threshold, mask, status) &
       bind(C, name="hpcs_rolling_detect_anomalies_2d")
    use iso_c_binding, only: c_int, c_double
    implicit none
    real(c_double), intent(in)  :: x(*)
    integer(c_int),  value      :: n
    integer(c_int),  value      :: m
    integer(c_int),  value      :: window
    real(c_double),  value      :: threshold
    integer(c_int), intent(out) :: mask(*)
    integer(c_int),  intent(out):: status

    integer(c_int) :: n_eff, m_eff, w_eff
    real(c_double) :: th
    integer(c_int) :: col, row, idx
    real(c_double), allocatable :: p_sum(:)
    real(c_double), allocatable :: p_sq(:)
    integer(c_int), allocatable :: p_cnt(:)
    integer(c_int) :: i_start, count
    real(c_double) :: sum_val, sum_sq, mean, var, std, xi, z
    integer(c_int) :: max_status, par_threshold

    n_eff = n
    m_eff = m
    w_eff = window
    th = threshold
    if (n_eff <= 0_c_int .or. m_eff <= 0_c_int .or. w_eff <= 0_c_int .or. w_eff > n_eff .or. th < 0.0_c_double) then
      status = HPCS_ERR_INVALID_ARGS
      return
    end if
    max_status = HPCS_SUCCESS
    par_threshold = hpcs_cpu_get_threshold(4)  ! THRESHOLD_ANOMALY_DETECT

    ! Loop over columns
!$omp parallel do default(none) shared(x,mask,n_eff,m_eff,w_eff,th,max_status,par_threshold) &
!$omp private(col,row,idx,i_start,count,sum_val,sum_sq,mean,var,std,xi,z,p_sum,p_sq,p_cnt) if(n_eff*m_eff > par_threshold)
    do col = 1_c_int, m_eff
      ! Allocate prefix arrays per column
      allocate(p_sum(0:n_eff))
      allocate(p_sq(0:n_eff))
      allocate(p_cnt(0:n_eff))
      p_sum(0) = 0.0_c_double
      p_sq(0)  = 0.0_c_double
      p_cnt(0) = 0_c_int
      ! Build prefix sums of values, squared values and valid counts
      do row = 1_c_int, n_eff
        idx = row + (col - 1_c_int) * n_eff
        xi = x(idx)
        if (xi == xi) then
          p_sum(row) = p_sum(row-1_c_int) + xi
          p_sq(row)  = p_sq(row-1_c_int) + xi*xi
          p_cnt(row) = p_cnt(row-1_c_int) + 1_c_int
        else
          p_sum(row) = p_sum(row-1_c_int)
          p_sq(row)  = p_sq(row-1_c_int)
          p_cnt(row) = p_cnt(row-1_c_int)
        end if
      end do
      ! Compute rolling statistics and z‑scores
      do row = 1_c_int, n_eff
        idx = row + (col - 1_c_int) * n_eff
        if (row >= w_eff) then
          i_start = row - w_eff
          sum_val = p_sum(row) - p_sum(i_start)
          sum_sq  = p_sq(row)  - p_sq(i_start)
          count   = p_cnt(row) - p_cnt(i_start)
          if (count < 2_c_int) then
            mask(idx) = 0_c_int
          else
            mean = sum_val / real(count, kind=c_double)
            var = (sum_sq / real(count, kind=c_double)) - mean*mean
            if (var < 0.0_c_double) var = 0.0_c_double
            std = sqrt(var)
            if (std == 0.0_c_double) then
              mask(idx) = 0_c_int
              if (HPCS_ERR_NUMERIC_FAIL > max_status) max_status = HPCS_ERR_NUMERIC_FAIL
            else
              xi = x(idx)
              if (xi == xi) then
                z = (xi - mean) / std
                if (abs(z) > th) then
                  mask(idx) = 1_c_int
                else
                  mask(idx) = 0_c_int
                end if
              else
                mask(idx) = 0_c_int
              end if
            end if
          end if
        else
          ! Not enough history for a full window
          mask(idx) = 0_c_int
        end if
      end do
      deallocate(p_sum)
      deallocate(p_sq)
      deallocate(p_cnt)
    end do
!$omp end parallel do
    status = max_status
  end subroutine hpcs_rolling_detect_anomalies_2d

end module hpcs_core_anomaly