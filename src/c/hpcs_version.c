/*
 * HPCSeries Core - Version Query Functions
 *
 * Provides runtime version information for ABI compatibility checking.
 * Enables downstream consumers (like SignalCore) to verify library versions
 * at compile-time and runtime.
 *
 * Author: HPCSeries Core Team
 * Version: 0.8.0
 */

#include "../../include/hpcs_core.h"

/*
 * Get library version string
 *
 * Returns the semantic version string (MAJOR.MINOR.PATCH).
 * This allows runtime version checks and logging.
 *
 * Returns:
 *   Statically allocated version string (do not free)
 *
 * Example:
 *   const char* version = hpcs_get_version();
 *   printf("Using HPCSeries Core %s\n", version);
 */
const char* hpcs_get_version(void) {
    return HPCS_VERSION_STRING;
}

/*
 * Get ABI version number
 *
 * Returns the ABI compatibility version. Increment this on breaking changes.
 * Downstream consumers can verify ABI compatibility:
 *   - Same ABI version = compatible
 *   - Different ABI version = incompatible, rebuild required
 *
 * Returns:
 *   ABI version number (currently 1)
 *
 * Example:
 *   int abi = hpcs_get_abi_version();
 *   if (abi < MIN_REQUIRED_ABI) {
 *       fprintf(stderr, "Incompatible ABI version\n");
 *       exit(1);
 *   }
 */
int hpcs_get_abi_version(void) {
    return HPCS_ABI_VERSION;
}
