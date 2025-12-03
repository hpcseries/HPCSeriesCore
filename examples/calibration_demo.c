/**
 * HPCS Calibration Demo - v0.5
 *
 * Demonstrates the benchmark-based auto-tuning system:
 * - Run calibration to find optimal thresholds
 * - Save configuration to ~/.hpcseries/config.json
 * - Load configuration on subsequent runs
 * - "Calibrate once → fast forever"
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// HPCS API
extern void hpcs_cpu_detect_init(void);
extern void hpcs_calibrate(int *status);
extern void hpcs_calibrate_quick(int *status);
extern void hpcs_save_config(const char *path, int *status);
extern void hpcs_load_config(const char *path, int *status);

// Tuning API
extern int hpcs_get_tuning_threshold(int op_class);
extern int hpcs_get_tuning_threads(int op_class);
extern int hpcs_get_tuning_numa_mode(int op_class);

// Operation classes
#define OP_SIMPLE    1
#define OP_ROLLING   2
#define OP_ROBUST    3
#define OP_ANOMALY   4

/**
 * Print current tuning configuration
 */
void print_tuning_config(void) {
    printf("=== Current Tuning Configuration ===\n");
    printf("\n");

    printf("Parallel Thresholds:\n");
    printf("  Simple:    %d elements\n", hpcs_get_tuning_threshold(OP_SIMPLE));
    printf("  Rolling:   %d elements\n", hpcs_get_tuning_threshold(OP_ROLLING));
    printf("  Robust:    %d elements\n", hpcs_get_tuning_threshold(OP_ROBUST));
    printf("  Anomaly:   %d elements\n", hpcs_get_tuning_threshold(OP_ANOMALY));
    printf("\n");

    printf("Thread Counts:\n");
    printf("  Simple:    %d threads\n", hpcs_get_tuning_threads(OP_SIMPLE));
    printf("  Rolling:   %d threads\n", hpcs_get_tuning_threads(OP_ROLLING));
    printf("  Robust:    %d threads\n", hpcs_get_tuning_threads(OP_ROBUST));
    printf("  Anomaly:   %d threads\n", hpcs_get_tuning_threads(OP_ANOMALY));
    printf("\n");

    printf("NUMA Modes:\n");
    const char *mode_names[] = {"AUTO", "COMPACT", "SPREAD"};
    printf("  Simple:    %s\n", mode_names[hpcs_get_tuning_numa_mode(OP_SIMPLE)]);
    printf("  Rolling:   %s\n", mode_names[hpcs_get_tuning_numa_mode(OP_ROLLING)]);
    printf("  Robust:    %s\n", mode_names[hpcs_get_tuning_numa_mode(OP_ROBUST)]);
    printf("  Anomaly:   %s\n", mode_names[hpcs_get_tuning_numa_mode(OP_ANOMALY)]);

    printf("====================================\n\n");
}

/**
 * Main demo program
 */
int main(int argc, char *argv[]) {
    int status;
    int mode = 0;  // 0=calibrate, 1=quick, 2=load only

    printf("HPCS Calibration Demo - v0.5\n");
    printf("=====================================\n\n");

    // Parse command line arguments
    if (argc > 1) {
        if (strcmp(argv[1], "--quick") == 0 || strcmp(argv[1], "-q") == 0) {
            mode = 1;
        } else if (strcmp(argv[1], "--load") == 0 || strcmp(argv[1], "-l") == 0) {
            mode = 2;
        } else if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
            printf("Usage: %s [OPTIONS]\n\n", argv[0]);
            printf("Options:\n");
            printf("  (none)      Full calibration (benchmark all kernels)\n");
            printf("  --quick     Quick calibration (hardware heuristics)\n");
            printf("  --load      Load existing config without calibration\n");
            printf("  --help      Show this help message\n\n");
            printf("Examples:\n");
            printf("  %s                # Full calibration\n", argv[0]);
            printf("  %s --quick        # Quick setup\n", argv[0]);
            printf("  %s --load         # Load previous calibration\n\n", argv[0]);
            return 0;
        }
    }

    // Initialize CPU detection
    hpcs_cpu_detect_init();

    // Mode 2: Load existing config
    if (mode == 2) {
        printf("Loading existing configuration...\n\n");
        hpcs_load_config(NULL, &status);  // NULL = use default path

        if (status != 0) {
            printf("ERROR: Failed to load config (status=%d)\n", status);
            printf("Run calibration first: %s\n", argv[0]);
            return 1;
        }

        printf("\nConfiguration loaded successfully!\n\n");
        print_tuning_config();
        return 0;
    }

    // Mode 1: Quick calibration
    if (mode == 1) {
        printf("Running quick calibration...\n\n");
        hpcs_calibrate_quick(&status);

        if (status != 0) {
            printf("ERROR: Quick calibration failed (status=%d)\n", status);
            return 1;
        }
    }
    // Mode 0: Full calibration
    else {
        printf("Running full calibration (this may take 30-60 seconds)...\n\n");
        hpcs_calibrate(&status);

        if (status != 0) {
            printf("ERROR: Calibration failed (status=%d)\n", status);
            return 1;
        }
    }

    // Print resulting configuration
    printf("\n");
    print_tuning_config();

    // Save configuration
    printf("Saving configuration...\n");
    hpcs_save_config(NULL, &status);  // NULL = use default path (~/.hpcseries/config.json)

    if (status != 0) {
        printf("ERROR: Failed to save config (status=%d)\n", status);
        return 1;
    }

    printf("\nCalibration complete!\n\n");
    printf("Configuration saved to: ~/.hpcseries/config.json\n");
    printf("(On Windows: %%APPDATA%%/hpcseries/config.json)\n\n");

    printf("Usage:\n");
    printf("  - Calibration results are now active\n");
    printf("  - Configuration persists across program runs\n");
    printf("  - Re-run calibration if you move to a different machine\n");
    printf("  - Use --load to skip calibration on subsequent runs\n\n");

    printf("Next steps:\n");
    printf("  1. Run your HPCSeries applications - they'll use these settings\n");
    printf("  2. Verify performance improvements with your workloads\n");
    printf("  3. Re-calibrate if you upgrade hardware or change workload\n\n");

    return 0;
}
