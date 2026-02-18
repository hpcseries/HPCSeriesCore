/*
 * HPCSeries Core - Workspace Memory Management (v0.8.0)
 *
 * Pre-allocated memory pool for pipeline execution.
 * Provides 64-byte aligned buffers for SIMD/cache efficiency.
 *
 * Key Design:
 * - NOT thread-safe (caller must use one workspace per thread)
 * - Growable via reserve() without preserving contents
 * - Used by pipelines for intermediate buffers and scratch space
 *
 * Author: HPCSeries Core Team
 * Version: 0.8.0
 */

#include "../../include/hpcs_core.h"
#include <stdlib.h>
#include <stdint.h>

/* Alignment for SIMD and cache line efficiency */
#define HPCS_WORKSPACE_ALIGNMENT 64

/* Internal workspace structure */
struct workspace {
    void   *buffer;      /* Aligned buffer pointer */
    void   *raw_buffer;  /* Original malloc pointer (for free) */
    size_t  capacity;    /* Current capacity in bytes */
};

/*
 * Create workspace with specified capacity
 *
 * Allocates a 64-byte aligned buffer of at least 'bytes' capacity.
 *
 * Parameters:
 *   bytes  - Minimum capacity in bytes
 *   ws     - Output: workspace handle
 *   status - 0=success, 3=out of memory
 */
void workspace_create(size_t bytes, workspace_t **ws, int *status) {
    *ws = NULL;
    *status = HPCS_SUCCESS;

    /* Allocate workspace struct */
    workspace_t *w = (workspace_t*)malloc(sizeof(workspace_t));
    if (w == NULL) {
        *status = HPCS_ERR_OUT_OF_MEMORY;
        set_last_error("Failed to allocate workspace struct");
        return;
    }

    /* Handle zero-size request */
    if (bytes == 0) {
        w->raw_buffer = NULL;
        w->buffer = NULL;
        w->capacity = 0;
        *ws = w;
        clear_last_error();
        return;
    }

    /* Allocate buffer with extra space for alignment */
    size_t alloc_size = bytes + HPCS_WORKSPACE_ALIGNMENT;
    w->raw_buffer = malloc(alloc_size);
    if (w->raw_buffer == NULL) {
        free(w);
        *status = HPCS_ERR_OUT_OF_MEMORY;
        set_last_error("Failed to allocate workspace buffer");
        return;
    }

    /* Align to 64 bytes */
    uintptr_t addr = (uintptr_t)w->raw_buffer;
    uintptr_t aligned = (addr + HPCS_WORKSPACE_ALIGNMENT - 1) & ~((uintptr_t)HPCS_WORKSPACE_ALIGNMENT - 1);
    w->buffer = (void*)aligned;
    w->capacity = bytes;

    *ws = w;
    clear_last_error();
}

/*
 * Free workspace and underlying buffer
 *
 * Safe to call with NULL pointer.
 */
void workspace_free(workspace_t *ws) {
    if (ws != NULL) {
        if (ws->raw_buffer != NULL) {
            free(ws->raw_buffer);
        }
        free(ws);
    }
}

/*
 * Query current workspace capacity in bytes
 *
 * Returns 0 for NULL workspace.
 */
size_t workspace_size(const workspace_t *ws) {
    if (ws == NULL) {
        return 0;
    }
    return ws->capacity;
}

/*
 * Grow workspace if bytes > current capacity
 *
 * Note: Old contents are NOT preserved (per spec).
 * If bytes <= current capacity, this is a no-op.
 *
 * Parameters:
 *   ws     - Workspace to grow
 *   bytes  - New minimum capacity
 *   status - 0=success, 1=invalid args, 3=out of memory
 */
void workspace_reserve(workspace_t *ws, size_t bytes, int *status) {
    *status = HPCS_SUCCESS;

    if (ws == NULL) {
        *status = HPCS_ERR_INVALID_ARGS;
        set_last_error("workspace_reserve: NULL workspace");
        return;
    }

    /* Already large enough */
    if (bytes <= ws->capacity) {
        clear_last_error();
        return;
    }

    /* Free old buffer */
    if (ws->raw_buffer != NULL) {
        free(ws->raw_buffer);
        ws->raw_buffer = NULL;
        ws->buffer = NULL;
        ws->capacity = 0;
    }

    /* Handle zero-size request */
    if (bytes == 0) {
        clear_last_error();
        return;
    }

    /* Allocate new buffer with alignment space */
    size_t alloc_size = bytes + HPCS_WORKSPACE_ALIGNMENT;
    ws->raw_buffer = malloc(alloc_size);
    if (ws->raw_buffer == NULL) {
        *status = HPCS_ERR_OUT_OF_MEMORY;
        set_last_error("Failed to reserve workspace");
        return;
    }

    /* Align to 64 bytes */
    uintptr_t addr = (uintptr_t)ws->raw_buffer;
    uintptr_t aligned = (addr + HPCS_WORKSPACE_ALIGNMENT - 1) & ~((uintptr_t)HPCS_WORKSPACE_ALIGNMENT - 1);
    ws->buffer = (void*)aligned;
    ws->capacity = bytes;

    clear_last_error();
}

/*
 * Get aligned buffer pointer (internal use)
 *
 * Returns the aligned buffer for use by pipelines.
 * Returns NULL if workspace is NULL or has zero capacity.
 */
void* workspace_get_buffer(workspace_t *ws) {
    if (ws == NULL) {
        return NULL;
    }
    return ws->buffer;
}
