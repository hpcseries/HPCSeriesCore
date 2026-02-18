/*
 * HPCSeries Core - Pipeline/Plan API (v0.8.0)
 *
 * Composable kernel execution for multi-stage data processing.
 * Chains multiple kernels with automatic intermediate buffer management.
 *
 * Key Design:
 * - Ping-pong buffers for efficient memory reuse
 * - Sequential stage execution (no cross-stage parallelism)
 * - Optional workspace for memory-intensive stages
 *
 * Author: HPCSeries Core Team
 * Version: 0.8.0
 */

#include "../../include/hpcs_core.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* --------------------------------------------------------------------- */
/* Stage Type Definitions                                                */
/* --------------------------------------------------------------------- */

typedef enum {
    /* Existing stages (v0.8.0) */
    HPCS_OP_DIFF,
    HPCS_OP_EWMA,
    HPCS_OP_EWVAR,
    HPCS_OP_EWSTD,
    HPCS_OP_ROLLING_MEAN,
    HPCS_OP_ROLLING_STD,
    HPCS_OP_ROLLING_MEDIAN,
    HPCS_OP_ROLLING_MAD,
    HPCS_OP_ZSCORE,
    HPCS_OP_ROBUST_ZSCORE,
    HPCS_OP_NORMALIZE_MINMAX,
    HPCS_OP_CLIP,

    /* New stages (v0.8.0) */
    HPCS_OP_CUMULATIVE_MIN,
    HPCS_OP_CUMULATIVE_MAX,
    HPCS_OP_FILL_FORWARD,
    HPCS_OP_PREFIX_SUM,
    HPCS_OP_CONVOLVE,
    HPCS_OP_LAG,
    HPCS_OP_LOG_RETURN,
    HPCS_OP_PCT_CHANGE,
    HPCS_OP_SCALE,
    HPCS_OP_SHIFT,
    HPCS_OP_ABS,
    HPCS_OP_SQRT
} hpcs_op_type_t;

/* Stage descriptor */
typedef struct {
    hpcs_op_type_t type;
    union {
        /* Existing params (v0.8.0) */
        struct { int order; } diff;
        struct { double alpha; } ewma;
        struct { double alpha; } ewvar;
        struct { double alpha; } ewstd;
        struct { int window; } rolling;
        struct { double eps; } robust_zscore;
        struct { double min_val, max_val; } clip;

        /* New params (v0.8.0) */
        struct { double *kernel; int m; } convolve;  /* Owns kernel copy */
        struct { int k; } lag;
        struct { double factor; } scale;
        struct { double offset; } shift;
    } params;
} hpcs_stage_t;

/* Initial capacity for stages array */
#define HPCS_PLAN_INITIAL_CAPACITY 8

/* Summary buffer size */
#define HPCS_PLAN_SUMMARY_SIZE 2048

/* --------------------------------------------------------------------- */
/* Internal Plan Structure                                               */
/* --------------------------------------------------------------------- */

struct pipeline {
    hpcs_stage_t    *stages;      /* Array of stage descriptors */
    int              n_stages;    /* Number of stages */
    int              capacity;    /* Allocated stage slots */
    workspace_t     *ws;          /* Workspace reference (not owned) */
    double          *buf_a;       /* Ping buffer */
    double          *buf_b;       /* Pong buffer */
    size_t           buf_size;    /* Size of ping/pong buffers (in elements) */
    char             summary[HPCS_PLAN_SUMMARY_SIZE];
};

/* --------------------------------------------------------------------- */
/* Internal Helper Functions                                             */
/* --------------------------------------------------------------------- */

/* Get workspace buffer pointer */
extern void* workspace_get_buffer(workspace_t *ws);

/* Ensure plan has space for one more stage */
static int plan_ensure_capacity(pipeline_t *plan, int *status) {
    if (plan->n_stages < plan->capacity) {
        return 1;  /* Already have space */
    }

    int new_capacity = plan->capacity * 2;
    if (new_capacity < HPCS_PLAN_INITIAL_CAPACITY) {
        new_capacity = HPCS_PLAN_INITIAL_CAPACITY;
    }

    hpcs_stage_t *new_stages = (hpcs_stage_t*)realloc(
        plan->stages, new_capacity * sizeof(hpcs_stage_t));

    if (new_stages == NULL) {
        *status = HPCS_ERR_OUT_OF_MEMORY;
        set_last_error("Failed to grow plan stages array");
        return 0;
    }

    plan->stages = new_stages;
    plan->capacity = new_capacity;
    return 1;
}

/* Add a stage to the plan */
static int plan_add_stage(pipeline_t *plan, const hpcs_stage_t *stage, int *status) {
    *status = HPCS_SUCCESS;

    if (plan == NULL) {
        *status = HPCS_ERR_INVALID_ARGS;
        set_last_error("plan_add_stage: NULL plan");
        return -1;
    }

    if (!plan_ensure_capacity(plan, status)) {
        return -1;
    }

    int idx = plan->n_stages;
    plan->stages[idx] = *stage;
    plan->n_stages++;

    clear_last_error();
    return idx;
}

/* Ensure ping/pong buffers are large enough */
static int plan_ensure_buffers(pipeline_t *plan, size_t n, int *status) {
    if (plan->buf_size >= n) {
        return 1;  /* Already large enough */
    }

    /* Free old buffers */
    if (plan->buf_a != NULL) {
        free(plan->buf_a);
        plan->buf_a = NULL;
    }
    if (plan->buf_b != NULL) {
        free(plan->buf_b);
        plan->buf_b = NULL;
    }
    plan->buf_size = 0;

    /* Allocate new buffers */
    plan->buf_a = (double*)malloc(n * sizeof(double));
    if (plan->buf_a == NULL) {
        *status = HPCS_ERR_OUT_OF_MEMORY;
        set_last_error("Failed to allocate plan ping buffer");
        return 0;
    }

    plan->buf_b = (double*)malloc(n * sizeof(double));
    if (plan->buf_b == NULL) {
        free(plan->buf_a);
        plan->buf_a = NULL;
        *status = HPCS_ERR_OUT_OF_MEMORY;
        set_last_error("Failed to allocate plan pong buffer");
        return 0;
    }

    plan->buf_size = n;
    return 1;
}

/* Get operation name string */
static const char* op_name(hpcs_op_type_t type) {
    switch (type) {
        /* Existing stages (v0.8.0) */
        case HPCS_OP_DIFF: return "diff";
        case HPCS_OP_EWMA: return "ewma";
        case HPCS_OP_EWVAR: return "ewvar";
        case HPCS_OP_EWSTD: return "ewstd";
        case HPCS_OP_ROLLING_MEAN: return "rolling_mean";
        case HPCS_OP_ROLLING_STD: return "rolling_std";
        case HPCS_OP_ROLLING_MEDIAN: return "rolling_median";
        case HPCS_OP_ROLLING_MAD: return "rolling_mad";
        case HPCS_OP_ZSCORE: return "zscore";
        case HPCS_OP_ROBUST_ZSCORE: return "robust_zscore";
        case HPCS_OP_NORMALIZE_MINMAX: return "normalize_minmax";
        case HPCS_OP_CLIP: return "clip";

        /* New stages (v0.8.0) */
        case HPCS_OP_CUMULATIVE_MIN: return "cumulative_min";
        case HPCS_OP_CUMULATIVE_MAX: return "cumulative_max";
        case HPCS_OP_FILL_FORWARD: return "fill_forward";
        case HPCS_OP_PREFIX_SUM: return "prefix_sum";
        case HPCS_OP_CONVOLVE: return "convolve";
        case HPCS_OP_LAG: return "lag";
        case HPCS_OP_LOG_RETURN: return "log_return";
        case HPCS_OP_PCT_CHANGE: return "pct_change";
        case HPCS_OP_SCALE: return "scale";
        case HPCS_OP_SHIFT: return "shift";
        case HPCS_OP_ABS: return "abs";
        case HPCS_OP_SQRT: return "sqrt";

        default: return "unknown";
    }
}

/* --------------------------------------------------------------------- */
/* Plan Lifecycle Functions                                              */
/* --------------------------------------------------------------------- */

/*
 * Create execution plan
 *
 * Parameters:
 *   ws     - Optional workspace (can be NULL for simple pipelines)
 *   status - 0=success, 3=out of memory
 *
 * Returns:
 *   New plan handle, or NULL on error
 */
pipeline_t* pipeline_create(workspace_t *ws, int *status) {
    *status = HPCS_SUCCESS;

    pipeline_t *plan = (pipeline_t*)malloc(sizeof(pipeline_t));
    if (plan == NULL) {
        *status = HPCS_ERR_OUT_OF_MEMORY;
        set_last_error("Failed to allocate plan");
        return NULL;
    }

    plan->stages = NULL;
    plan->n_stages = 0;
    plan->capacity = 0;
    plan->ws = ws;
    plan->buf_a = NULL;
    plan->buf_b = NULL;
    plan->buf_size = 0;
    plan->summary[0] = '\0';

    clear_last_error();
    return plan;
}

/*
 * Free plan and all associated resources
 */
void pipeline_free(pipeline_t *plan) {
    if (plan != NULL) {
        if (plan->stages != NULL) {
            free(plan->stages);
        }
        if (plan->buf_a != NULL) {
            free(plan->buf_a);
        }
        if (plan->buf_b != NULL) {
            free(plan->buf_b);
        }
        free(plan);
    }
}

/* --------------------------------------------------------------------- */
/* Plan Stage Addition Functions                                         */
/* --------------------------------------------------------------------- */

int pipeline_add_diff(pipeline_t *plan, int order, int *status) {
    if (order < 1) {
        *status = HPCS_ERR_INVALID_ARGS;
        set_last_error("diff: order must be >= 1");
        return -1;
    }
    hpcs_stage_t stage = { .type = HPCS_OP_DIFF };
    stage.params.diff.order = order;
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_ewma(pipeline_t *plan, double alpha, int *status) {
    if (alpha <= 0.0 || alpha > 1.0) {
        *status = HPCS_ERR_INVALID_ARGS;
        set_last_error("ewma: alpha must be in (0, 1]");
        return -1;
    }
    hpcs_stage_t stage = { .type = HPCS_OP_EWMA };
    stage.params.ewma.alpha = alpha;
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_ewvar(pipeline_t *plan, double alpha, int *status) {
    if (alpha <= 0.0 || alpha > 1.0) {
        *status = HPCS_ERR_INVALID_ARGS;
        set_last_error("ewvar: alpha must be in (0, 1]");
        return -1;
    }
    hpcs_stage_t stage = { .type = HPCS_OP_EWVAR };
    stage.params.ewvar.alpha = alpha;
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_ewstd(pipeline_t *plan, double alpha, int *status) {
    if (alpha <= 0.0 || alpha > 1.0) {
        *status = HPCS_ERR_INVALID_ARGS;
        set_last_error("ewstd: alpha must be in (0, 1]");
        return -1;
    }
    hpcs_stage_t stage = { .type = HPCS_OP_EWSTD };
    stage.params.ewstd.alpha = alpha;
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_rolling_mean(pipeline_t *plan, int window, int *status) {
    if (window < 1) {
        *status = HPCS_ERR_INVALID_ARGS;
        set_last_error("rolling_mean: window must be >= 1");
        return -1;
    }
    hpcs_stage_t stage = { .type = HPCS_OP_ROLLING_MEAN };
    stage.params.rolling.window = window;
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_rolling_std(pipeline_t *plan, int window, int *status) {
    if (window < 1) {
        *status = HPCS_ERR_INVALID_ARGS;
        set_last_error("rolling_std: window must be >= 1");
        return -1;
    }
    hpcs_stage_t stage = { .type = HPCS_OP_ROLLING_STD };
    stage.params.rolling.window = window;
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_rolling_median(pipeline_t *plan, int window, int *status) {
    if (window < 1) {
        *status = HPCS_ERR_INVALID_ARGS;
        set_last_error("rolling_median: window must be >= 1");
        return -1;
    }
    hpcs_stage_t stage = { .type = HPCS_OP_ROLLING_MEDIAN };
    stage.params.rolling.window = window;
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_rolling_mad(pipeline_t *plan, int window, int *status) {
    if (window < 1) {
        *status = HPCS_ERR_INVALID_ARGS;
        set_last_error("rolling_mad: window must be >= 1");
        return -1;
    }
    hpcs_stage_t stage = { .type = HPCS_OP_ROLLING_MAD };
    stage.params.rolling.window = window;
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_zscore(pipeline_t *plan, int *status) {
    hpcs_stage_t stage = { .type = HPCS_OP_ZSCORE };
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_robust_zscore(pipeline_t *plan, double eps, int *status) {
    if (eps < 0.0) {
        *status = HPCS_ERR_INVALID_ARGS;
        set_last_error("robust_zscore: eps must be >= 0");
        return -1;
    }
    hpcs_stage_t stage = { .type = HPCS_OP_ROBUST_ZSCORE };
    stage.params.robust_zscore.eps = eps;
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_normalize_minmax(pipeline_t *plan, int *status) {
    hpcs_stage_t stage = { .type = HPCS_OP_NORMALIZE_MINMAX };
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_clip(pipeline_t *plan, double min_val, double max_val, int *status) {
    if (min_val > max_val) {
        *status = HPCS_ERR_INVALID_ARGS;
        set_last_error("clip: min_val must be <= max_val");
        return -1;
    }
    hpcs_stage_t stage = { .type = HPCS_OP_CLIP };
    stage.params.clip.min_val = min_val;
    stage.params.clip.max_val = max_val;
    return plan_add_stage(plan, &stage, status);
}

/* --------------------------------------------------------------------- */
/* New Pipeline Stage Addition Functions (v0.8.0)                        */
/* --------------------------------------------------------------------- */

int pipeline_add_cumulative_min(pipeline_t *plan, int *status) {
    hpcs_stage_t stage = { .type = HPCS_OP_CUMULATIVE_MIN };
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_cumulative_max(pipeline_t *plan, int *status) {
    hpcs_stage_t stage = { .type = HPCS_OP_CUMULATIVE_MAX };
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_fill_forward(pipeline_t *plan, int *status) {
    hpcs_stage_t stage = { .type = HPCS_OP_FILL_FORWARD };
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_prefix_sum(pipeline_t *plan, int *status) {
    hpcs_stage_t stage = { .type = HPCS_OP_PREFIX_SUM };
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_convolve(pipeline_t *plan, const double *kernel, int m, int *status) {
    *status = HPCS_SUCCESS;
    if (kernel == NULL || m < 1) {
        *status = HPCS_ERR_INVALID_ARGS;
        set_last_error("convolve: kernel must be non-NULL with m >= 1");
        return -1;
    }
    hpcs_stage_t stage = { .type = HPCS_OP_CONVOLVE };
    stage.params.convolve.kernel = (double*)malloc(m * sizeof(double));
    if (!stage.params.convolve.kernel) {
        *status = HPCS_ERR_OUT_OF_MEMORY;
        set_last_error("convolve: failed to allocate kernel copy");
        return -1;
    }
    memcpy(stage.params.convolve.kernel, kernel, m * sizeof(double));
    stage.params.convolve.m = m;
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_lag(pipeline_t *plan, int k, int *status) {
    *status = HPCS_SUCCESS;
    if (k < 1) {
        *status = HPCS_ERR_INVALID_ARGS;
        set_last_error("lag: k must be >= 1");
        return -1;
    }
    hpcs_stage_t stage = { .type = HPCS_OP_LAG };
    stage.params.lag.k = k;
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_log_return(pipeline_t *plan, int *status) {
    hpcs_stage_t stage = { .type = HPCS_OP_LOG_RETURN };
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_pct_change(pipeline_t *plan, int *status) {
    hpcs_stage_t stage = { .type = HPCS_OP_PCT_CHANGE };
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_scale(pipeline_t *plan, double factor, int *status) {
    hpcs_stage_t stage = { .type = HPCS_OP_SCALE };
    stage.params.scale.factor = factor;
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_shift(pipeline_t *plan, double offset, int *status) {
    hpcs_stage_t stage = { .type = HPCS_OP_SHIFT };
    stage.params.shift.offset = offset;
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_abs(pipeline_t *plan, int *status) {
    hpcs_stage_t stage = { .type = HPCS_OP_ABS };
    return plan_add_stage(plan, &stage, status);
}

int pipeline_add_sqrt(pipeline_t *plan, int *status) {
    hpcs_stage_t stage = { .type = HPCS_OP_SQRT };
    return plan_add_stage(plan, &stage, status);
}

/* --------------------------------------------------------------------- */
/* Plan Execution                                                        */
/* --------------------------------------------------------------------- */

/*
 * Execute plan on input array
 *
 * Uses ping-pong buffers for intermediate results.
 * Last stage writes directly to output.
 */
void pipeline_execute(
    const pipeline_t *plan,
    const double *x,
    size_t n,
    double *out,
    size_t *out_n,
    int *status
) {
    *status = HPCS_SUCCESS;
    if (out_n) *out_n = n;  /* Default: output length equals input length */

    /* Validate inputs */
    if (plan == NULL || x == NULL || out == NULL) {
        *status = HPCS_ERR_INVALID_ARGS;
        set_last_error("pipeline_execute: NULL argument");
        return;
    }

    if (n == 0) {
        *status = HPCS_ERR_INVALID_ARGS;
        set_last_error("pipeline_execute: n must be > 0");
        return;
    }

    /* Empty plan - just copy input to output */
    if (plan->n_stages == 0) {
        memcpy(out, x, n * sizeof(double));
        clear_last_error();
        return;
    }

    /* Ensure buffers are large enough (cast away const for buffer management) */
    pipeline_t *mplan = (pipeline_t*)plan;
    if (!plan_ensure_buffers(mplan, n, status)) {
        return;
    }

    /* Execute stages with ping-pong buffering */
    const double *in_ptr = x;
    size_t in_len = n;
    double *out_ptr;
    int use_buf_a = 1;

    for (int i = 0; i < plan->n_stages; i++) {
        const hpcs_stage_t *stage = &plan->stages[i];

        /* Select output buffer */
        if (i == plan->n_stages - 1) {
            out_ptr = out;  /* Last stage writes to output */
        } else {
            out_ptr = use_buf_a ? mplan->buf_a : mplan->buf_b;
            use_buf_a = !use_buf_a;
        }

        /* Execute kernel based on stage type */
        int kernel_status = 0;

        switch (stage->type) {
            case HPCS_OP_DIFF:
                hpcs_diff(in_ptr, (int)in_len, stage->params.diff.order,
                          out_ptr, &kernel_status);
                /* diff reduces length by order (first 'order' elements are NaN) */
                break;

            case HPCS_OP_EWMA:
                hpcs_ewma(in_ptr, (int)in_len, stage->params.ewma.alpha,
                          out_ptr, HPCS_MODE_USE_GLOBAL, &kernel_status);
                break;

            case HPCS_OP_EWVAR:
                hpcs_ewvar(in_ptr, (int)in_len, stage->params.ewvar.alpha,
                           out_ptr, HPCS_MODE_USE_GLOBAL, &kernel_status);
                break;

            case HPCS_OP_EWSTD:
                hpcs_ewstd(in_ptr, (int)in_len, stage->params.ewstd.alpha,
                           out_ptr, HPCS_MODE_USE_GLOBAL, &kernel_status);
                break;

            case HPCS_OP_ROLLING_MEAN:
                hpcs_rolling_mean(in_ptr, (int)in_len, stage->params.rolling.window,
                                  out_ptr, HPCS_MODE_USE_GLOBAL, &kernel_status);
                break;

            case HPCS_OP_ROLLING_STD:
                hpcs_rolling_std(in_ptr, (int)in_len, stage->params.rolling.window,
                                 out_ptr, HPCS_MODE_USE_GLOBAL, &kernel_status);
                break;

            case HPCS_OP_ROLLING_MEDIAN:
                hpcs_rolling_median(in_ptr, (int)in_len, stage->params.rolling.window,
                                    out_ptr, HPCS_MODE_USE_GLOBAL, &kernel_status);
                break;

            case HPCS_OP_ROLLING_MAD:
                hpcs_rolling_mad(in_ptr, (int)in_len, stage->params.rolling.window,
                                 out_ptr, HPCS_MODE_USE_GLOBAL, &kernel_status);
                break;

            case HPCS_OP_ZSCORE:
                hpcs_zscore(in_ptr, (int)in_len, out_ptr, &kernel_status);
                break;

            case HPCS_OP_ROBUST_ZSCORE:
                hpcs_robust_zscore(in_ptr, (int)in_len, out_ptr, &kernel_status);
                break;

            case HPCS_OP_NORMALIZE_MINMAX:
                hpcs_normalize_minmax(in_ptr, (int)in_len, out_ptr, &kernel_status);
                break;

            case HPCS_OP_CLIP:
                /* clip modifies in-place, so copy first */
                memcpy(out_ptr, in_ptr, in_len * sizeof(double));
                hpcs_clip(out_ptr, (int)in_len, stage->params.clip.min_val,
                          stage->params.clip.max_val, &kernel_status);
                break;

            /* New stages (v0.8.0) */
            case HPCS_OP_CUMULATIVE_MIN:
                hpcs_cumulative_min(in_ptr, (int)in_len, out_ptr,
                                    HPCS_MODE_USE_GLOBAL, &kernel_status);
                break;

            case HPCS_OP_CUMULATIVE_MAX:
                hpcs_cumulative_max(in_ptr, (int)in_len, out_ptr,
                                    HPCS_MODE_USE_GLOBAL, &kernel_status);
                break;

            case HPCS_OP_FILL_FORWARD:
                hpcs_fill_forward(in_ptr, (int)in_len, out_ptr, &kernel_status);
                break;

            case HPCS_OP_PREFIX_SUM:
                hpcs_prefix_sum(in_ptr, (int)in_len, out_ptr, &kernel_status);
                break;

            case HPCS_OP_CONVOLVE:
                hpcs_convolve_valid(in_ptr, (int)in_len,
                                    stage->params.convolve.kernel,
                                    stage->params.convolve.m,
                                    out_ptr, HPCS_MODE_USE_GLOBAL, &kernel_status);
                /* Convolve reduces length by (m - 1) */
                if (kernel_status == 0) in_len -= (size_t)(stage->params.convolve.m - 1);
                break;

            case HPCS_OP_LAG:
                /* y[i] = x[i-k] for i >= k, NaN for i < k */
                {
                    int k = stage->params.lag.k;
                    size_t i;
                    for (i = 0; i < (size_t)k && i < in_len; i++) {
                        out_ptr[i] = NAN;
                    }
                    for (i = (size_t)k; i < in_len; i++) {
                        out_ptr[i] = in_ptr[i - k];
                    }
                }
                break;

            case HPCS_OP_LOG_RETURN:
                /* y[0] = NaN, y[t] = log(x[t] / x[t-1]) */
                out_ptr[0] = NAN;
                for (size_t i = 1; i < in_len; i++) {
                    out_ptr[i] = log(in_ptr[i] / in_ptr[i-1]);
                }
                break;

            case HPCS_OP_PCT_CHANGE:
                /* y[0] = NaN, y[t] = (x[t] - x[t-1]) / x[t-1] */
                out_ptr[0] = NAN;
                for (size_t i = 1; i < in_len; i++) {
                    out_ptr[i] = (in_ptr[i] - in_ptr[i-1]) / in_ptr[i-1];
                }
                break;

            case HPCS_OP_SCALE:
                for (size_t i = 0; i < in_len; i++) {
                    out_ptr[i] = in_ptr[i] * stage->params.scale.factor;
                }
                break;

            case HPCS_OP_SHIFT:
                for (size_t i = 0; i < in_len; i++) {
                    out_ptr[i] = in_ptr[i] + stage->params.shift.offset;
                }
                break;

            case HPCS_OP_ABS:
                for (size_t i = 0; i < in_len; i++) {
                    out_ptr[i] = fabs(in_ptr[i]);
                }
                break;

            case HPCS_OP_SQRT:
                for (size_t i = 0; i < in_len; i++) {
                    out_ptr[i] = sqrt(in_ptr[i]);
                }
                break;

            default:
                *status = HPCS_ERR_INTERNAL;
                set_last_error("pipeline_execute: unknown operation type");
                return;
        }

        /* Check kernel status */
        if (kernel_status != HPCS_SUCCESS) {
            *status = kernel_status;
            char err_msg[128];
            snprintf(err_msg, sizeof(err_msg),
                     "pipeline_execute: stage %d (%s) failed with status %d",
                     i, op_name(stage->type), kernel_status);
            set_last_error(err_msg);
            return;
        }

        /* Update input pointer for next stage */
        in_ptr = out_ptr;
    }

    /* Return actual output length (may be smaller due to convolve) */
    if (out_n) *out_n = in_len;

    clear_last_error();
}

/* --------------------------------------------------------------------- */
/* Plan Summary                                                          */
/* --------------------------------------------------------------------- */

/*
 * Get human-readable plan summary
 */
const char* pipeline_summary(const pipeline_t *plan) {
    if (plan == NULL) {
        return "NULL plan";
    }

    pipeline_t *mplan = (pipeline_t*)plan;
    char *buf = mplan->summary;
    size_t remaining = HPCS_PLAN_SUMMARY_SIZE;
    int written;

    written = snprintf(buf, remaining, "Pipeline summary (%d stages):\n", plan->n_stages);
    if (written > 0 && (size_t)written < remaining) {
        buf += written;
        remaining -= written;
    }

    for (int i = 0; i < plan->n_stages && remaining > 0; i++) {
        const hpcs_stage_t *stage = &plan->stages[i];

        switch (stage->type) {
            case HPCS_OP_DIFF:
                written = snprintf(buf, remaining, "  %d) diff(order=%d)\n",
                                   i + 1, stage->params.diff.order);
                break;
            case HPCS_OP_EWMA:
                written = snprintf(buf, remaining, "  %d) ewma(alpha=%.4f)\n",
                                   i + 1, stage->params.ewma.alpha);
                break;
            case HPCS_OP_EWVAR:
                written = snprintf(buf, remaining, "  %d) ewvar(alpha=%.4f)\n",
                                   i + 1, stage->params.ewvar.alpha);
                break;
            case HPCS_OP_EWSTD:
                written = snprintf(buf, remaining, "  %d) ewstd(alpha=%.4f)\n",
                                   i + 1, stage->params.ewstd.alpha);
                break;
            case HPCS_OP_ROLLING_MEAN:
                written = snprintf(buf, remaining, "  %d) rolling_mean(window=%d)\n",
                                   i + 1, stage->params.rolling.window);
                break;
            case HPCS_OP_ROLLING_STD:
                written = snprintf(buf, remaining, "  %d) rolling_std(window=%d)\n",
                                   i + 1, stage->params.rolling.window);
                break;
            case HPCS_OP_ROLLING_MEDIAN:
                written = snprintf(buf, remaining, "  %d) rolling_median(window=%d)\n",
                                   i + 1, stage->params.rolling.window);
                break;
            case HPCS_OP_ROLLING_MAD:
                written = snprintf(buf, remaining, "  %d) rolling_mad(window=%d)\n",
                                   i + 1, stage->params.rolling.window);
                break;
            case HPCS_OP_ZSCORE:
                written = snprintf(buf, remaining, "  %d) zscore()\n", i + 1);
                break;
            case HPCS_OP_ROBUST_ZSCORE:
                written = snprintf(buf, remaining, "  %d) robust_zscore(eps=%.2e)\n",
                                   i + 1, stage->params.robust_zscore.eps);
                break;
            case HPCS_OP_NORMALIZE_MINMAX:
                written = snprintf(buf, remaining, "  %d) normalize_minmax()\n", i + 1);
                break;
            case HPCS_OP_CLIP:
                written = snprintf(buf, remaining, "  %d) clip(min=%.4f, max=%.4f)\n",
                                   i + 1, stage->params.clip.min_val, stage->params.clip.max_val);
                break;

            /* New stages (v0.8.0) */
            case HPCS_OP_CUMULATIVE_MIN:
                written = snprintf(buf, remaining, "  %d) cumulative_min()\n", i + 1);
                break;
            case HPCS_OP_CUMULATIVE_MAX:
                written = snprintf(buf, remaining, "  %d) cumulative_max()\n", i + 1);
                break;
            case HPCS_OP_FILL_FORWARD:
                written = snprintf(buf, remaining, "  %d) fill_forward()\n", i + 1);
                break;
            case HPCS_OP_PREFIX_SUM:
                written = snprintf(buf, remaining, "  %d) prefix_sum()\n", i + 1);
                break;
            case HPCS_OP_CONVOLVE:
                written = snprintf(buf, remaining, "  %d) convolve(m=%d)\n",
                                   i + 1, stage->params.convolve.m);
                break;
            case HPCS_OP_LAG:
                written = snprintf(buf, remaining, "  %d) lag(k=%d)\n",
                                   i + 1, stage->params.lag.k);
                break;
            case HPCS_OP_LOG_RETURN:
                written = snprintf(buf, remaining, "  %d) log_return()\n", i + 1);
                break;
            case HPCS_OP_PCT_CHANGE:
                written = snprintf(buf, remaining, "  %d) pct_change()\n", i + 1);
                break;
            case HPCS_OP_SCALE:
                written = snprintf(buf, remaining, "  %d) scale(factor=%.4f)\n",
                                   i + 1, stage->params.scale.factor);
                break;
            case HPCS_OP_SHIFT:
                written = snprintf(buf, remaining, "  %d) shift(offset=%.4f)\n",
                                   i + 1, stage->params.shift.offset);
                break;
            case HPCS_OP_ABS:
                written = snprintf(buf, remaining, "  %d) abs()\n", i + 1);
                break;
            case HPCS_OP_SQRT:
                written = snprintf(buf, remaining, "  %d) sqrt()\n", i + 1);
                break;

            default:
                written = snprintf(buf, remaining, "  %d) unknown()\n", i + 1);
                break;
        }

        if (written > 0 && (size_t)written < remaining) {
            buf += written;
            remaining -= written;
        }
    }

    return mplan->summary;
}
