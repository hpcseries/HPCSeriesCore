! ==============================================================================
! HPC Series Core Library - 1D Operations Module
! Rolling window operations and statistical transformations
! ==============================================================================

module hpcs_core_1d
    use iso_c_binding
    implicit none

contains

    ! ==========================================================================
    ! Rolling Mean
    ! ==========================================================================
    subroutine hpcs_rolling_mean(input, n, window, output) bind(C, name="hpcs_rolling_mean")
        integer(c_size_t), intent(in), value :: n, window
        real(c_double), intent(in) :: input(n)
        real(c_double), intent(out) :: output(n)

        integer(c_size_t) :: i
        real(c_double) :: sum_val
        integer(c_size_t) :: start_idx, end_idx, count

        ! Handle initial window (partial)
        do i = 1, min(window, n)
            sum_val = sum(input(1:i))
            output(i) = sum_val / real(i, c_double)
        end do

        ! Full window rolling
        if (n > window) then
            sum_val = sum(input(1:window))
            do i = window + 1, n
                sum_val = sum_val - input(i - window) + input(i)
                output(i) = sum_val / real(window, c_double)
            end do
        end if
    end subroutine hpcs_rolling_mean

    ! ==========================================================================
    ! Rolling Standard Deviation
    ! ==========================================================================
    subroutine hpcs_rolling_std(input, n, window, output) bind(C, name="hpcs_rolling_std")
        integer(c_size_t), intent(in), value :: n, window
        real(c_double), intent(in) :: input(n)
        real(c_double), intent(out) :: output(n)

        integer(c_size_t) :: i, j
        real(c_double) :: mean_val, variance, diff
        integer(c_size_t) :: start_idx, end_idx, count

        do i = 1, n
            start_idx = max(1_c_size_t, i - window + 1)
            end_idx = i
            count = end_idx - start_idx + 1

            ! Calculate mean
            mean_val = sum(input(start_idx:end_idx)) / real(count, c_double)

            ! Calculate variance
            variance = 0.0_c_double
            do j = start_idx, end_idx
                diff = input(j) - mean_val
                variance = variance + diff * diff
            end do
            variance = variance / real(count, c_double)

            output(i) = sqrt(variance)
        end do
    end subroutine hpcs_rolling_std

    ! ==========================================================================
    ! Rolling Minimum
    ! ==========================================================================
    subroutine hpcs_rolling_min(input, n, window, output) bind(C, name="hpcs_rolling_min")
        integer(c_size_t), intent(in), value :: n, window
        real(c_double), intent(in) :: input(n)
        real(c_double), intent(out) :: output(n)

        integer(c_size_t) :: i, start_idx, end_idx

        do i = 1, n
            start_idx = max(1_c_size_t, i - window + 1)
            end_idx = i
            output(i) = minval(input(start_idx:end_idx))
        end do
    end subroutine hpcs_rolling_min

    ! ==========================================================================
    ! Rolling Maximum
    ! ==========================================================================
    subroutine hpcs_rolling_max(input, n, window, output) bind(C, name="hpcs_rolling_max")
        integer(c_size_t), intent(in), value :: n, window
        real(c_double), intent(in) :: input(n)
        real(c_double), intent(out) :: output(n)

        integer(c_size_t) :: i, start_idx, end_idx

        do i = 1, n
            start_idx = max(1_c_size_t, i - window + 1)
            end_idx = i
            output(i) = maxval(input(start_idx:end_idx))
        end do
    end subroutine hpcs_rolling_max

    ! ==========================================================================
    ! Rolling Sum
    ! ==========================================================================
    subroutine hpcs_rolling_sum(input, n, window, output) bind(C, name="hpcs_rolling_sum")
        integer(c_size_t), intent(in), value :: n, window
        real(c_double), intent(in) :: input(n)
        real(c_double), intent(out) :: output(n)

        integer(c_size_t) :: i
        real(c_double) :: sum_val

        ! Initial window
        do i = 1, min(window, n)
            output(i) = sum(input(1:i))
        end do

        ! Rolling sum with sliding window
        if (n > window) then
            sum_val = sum(input(1:window))
            do i = window + 1, n
                sum_val = sum_val - input(i - window) + input(i)
                output(i) = sum_val
            end do
        end if
    end subroutine hpcs_rolling_sum

    ! ==========================================================================
    ! Z-Score Normalization
    ! ==========================================================================
    subroutine hpcs_zscore(input, n, output) bind(C, name="hpcs_zscore")
        integer(c_size_t), intent(in), value :: n
        real(c_double), intent(in) :: input(n)
        real(c_double), intent(out) :: output(n)

        real(c_double) :: mean_val, std_val, variance
        integer(c_size_t) :: i

        ! Calculate mean
        mean_val = sum(input) / real(n, c_double)

        ! Calculate standard deviation
        variance = 0.0_c_double
        do i = 1, n
            variance = variance + (input(i) - mean_val)**2
        end do
        variance = variance / real(n, c_double)
        std_val = sqrt(variance)

        ! Normalize
        if (std_val > 0.0_c_double) then
            do i = 1, n
                output(i) = (input(i) - mean_val) / std_val
            end do
        else
            output = 0.0_c_double
        end if
    end subroutine hpcs_zscore

    ! ==========================================================================
    ! Rank Transformation
    ! ==========================================================================
    subroutine hpcs_rank(input, n, output) bind(C, name="hpcs_rank")
        integer(c_size_t), intent(in), value :: n
        real(c_double), intent(in) :: input(n)
        real(c_double), intent(out) :: output(n)

        integer(c_size_t) :: i, j

        ! Simple ranking (could be optimized with sorting)
        do i = 1, n
            output(i) = 1.0_c_double
            do j = 1, n
                if (input(j) < input(i)) then
                    output(i) = output(i) + 1.0_c_double
                end if
            end do
        end do
    end subroutine hpcs_rank

end module hpcs_core_1d
