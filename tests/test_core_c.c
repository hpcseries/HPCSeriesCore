/**
 * @file test_core_c.c
 * @brief C test harness for HPC Series Core Library
 *
 * Tests kernel correctness using C ABI
 */

#include <stdio.h>
#include <stdlib.h>
#include "hpcs_core.h"

int main(void) {
    printf("HPC Series Core - C Test Suite\n");
    printf("Version: %s\n", hpcs_version());

    // TODO: Implement test cases for:
    // - Rolling operations
    // - Statistical transforms
    // - Reductions

    printf("Tests not yet implemented\n");
    return 0;
}
