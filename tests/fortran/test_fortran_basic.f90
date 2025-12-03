! HPCSeries Core v0.7 - Fortran Basic Tests
! ==========================================
!
! Tests for Fortran module integration and basic operations.

program test_fortran_basic
    use iso_c_binding
    use hpcs_constants
    use hpcs_simd_interface
    implicit none

    integer :: total_tests, passed_tests, failed_tests
    logical :: all_passed

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    print '(A)', '================================'
    print '(A)', 'HPCSeries Fortran Test Suite'
    print '(A)', '================================'
    print *

    ! Run tests
    call test_reduce_sum(passed_tests, failed_tests)
    call test_reduce_mean(passed_tests, failed_tests)
    call test_reduce_min_max(passed_tests, failed_tests)
    call test_simd_interface(passed_tests, failed_tests)

    ! Summary
    total_tests = passed_tests + failed_tests
    print *
    print '(A)', '================================'
    print '(A,I0)', 'Total tests:  ', total_tests
    print '(A,I0)', 'Passed:       ', passed_tests
    print '(A,I0)', 'Failed:       ', failed_tests
    print '(A)', '================================'

    all_passed = (failed_tests == 0)
    if (all_passed) then
        print '(A)', '✓ All tests passed!'
        stop 0
    else
        print '(A)', '✗ Some tests failed'
        stop 1
    end if

contains

    subroutine test_reduce_sum(passed, failed)
        integer, intent(inout) :: passed, failed
        real(c_double) :: x(5), result
        integer(c_int) :: status

        print '(A)', '[TEST] Reduce Sum'

        ! Test data: [1, 2, 3, 4, 5]
        x = [1.0_c_double, 2.0_c_double, 3.0_c_double, 4.0_c_double, 5.0_c_double]

        call hpcs_reduce_sum_simd(x, 5, result, status)

        if (status /= HPCS_SUCCESS) then
            print '(A)', '  ✗ FAILED: Status error'
            failed = failed + 1
            return
        end if

        if (abs(result - 15.0_c_double) < 1.0e-10_c_double) then
            print '(A)', '  ✓ PASSED'
            passed = passed + 1
        else
            print '(A,F10.4)', '  ✗ FAILED: Expected 15.0, got ', result
            failed = failed + 1
        end if
    end subroutine test_reduce_sum

    subroutine test_reduce_mean(passed, failed)
        integer, intent(inout) :: passed, failed
        real(c_double) :: x(5), result
        integer(c_int) :: status

        print '(A)', '[TEST] Reduce Mean'

        x = [1.0_c_double, 2.0_c_double, 3.0_c_double, 4.0_c_double, 5.0_c_double]

        call hpcs_reduce_mean_simd(x, 5, result, status)

        if (status /= HPCS_SUCCESS) then
            print '(A)', '  ✗ FAILED: Status error'
            failed = failed + 1
            return
        end if

        if (abs(result - 3.0_c_double) < 1.0e-10_c_double) then
            print '(A)', '  ✓ PASSED'
            passed = passed + 1
        else
            print '(A,F10.4)', '  ✗ FAILED: Expected 3.0, got ', result
            failed = failed + 1
        end if
    end subroutine test_reduce_mean

    subroutine test_reduce_min_max(passed, failed)
        integer, intent(inout) :: passed, failed
        real(c_double) :: x(5), min_val, max_val
        integer(c_int) :: status

        print '(A)', '[TEST] Reduce Min/Max'

        x = [5.0_c_double, 2.0_c_double, 8.0_c_double, 1.0_c_double, 9.0_c_double]

        call hpcs_reduce_min_simd(x, 5, min_val, status)
        if (status /= HPCS_SUCCESS) then
            print '(A)', '  ✗ FAILED: Min status error'
            failed = failed + 1
            return
        end if

        call hpcs_reduce_max_simd(x, 5, max_val, status)
        if (status /= HPCS_SUCCESS) then
            print '(A)', '  ✗ FAILED: Max status error'
            failed = failed + 1
            return
        end if

        if (abs(min_val - 1.0_c_double) < 1.0e-10_c_double .and. &
            abs(max_val - 9.0_c_double) < 1.0e-10_c_double) then
            print '(A)', '  ✓ PASSED'
            passed = passed + 1
        else
            print '(A,F10.4,A,F10.4)', '  ✗ FAILED: Min=', min_val, ', Max=', max_val
            failed = failed + 1
        end if
    end subroutine test_reduce_min_max

    subroutine test_simd_interface(passed, failed)
        integer, intent(inout) :: passed, failed
        real(c_double) :: x(10), result
        integer(c_int) :: status, i

        print '(A)', '[TEST] SIMD Interface'

        ! Initialize array
        do i = 1, 10
            x(i) = real(i, c_double)
        end do

        ! Test SIMD-accelerated sum
        call hpcs_reduce_sum_simd(x, 10, result, status)

        if (status /= HPCS_SUCCESS) then
            print '(A)', '  ✗ FAILED: Status error'
            failed = failed + 1
            return
        end if

        ! Sum of 1..10 = 55
        if (abs(result - 55.0_c_double) < 1.0e-10_c_double) then
            print '(A)', '  ✓ PASSED'
            passed = passed + 1
        else
            print '(A,F10.4)', '  ✗ FAILED: Expected 55.0, got ', result
            failed = failed + 1
        end if
    end subroutine test_simd_interface

end program test_fortran_basic
