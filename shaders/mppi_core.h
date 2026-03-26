#ifndef MPPI_CORE_H
#define MPPI_CORE_H

#include <metal_stdlib>

// ---------------------------------------------------------------------------
// Function type signatures for user-provided [[visible]] device callables.
// ---------------------------------------------------------------------------

/// In-place state propagation (may mutate state).
using MppiDynamicsFn = void(thread uint8_t* state,
                            thread const float* control,
                            device const uint8_t* model_params);

/// Per-timestep running cost (must not mutate state).
using MppiStageCostFn = float(thread const uint8_t* state,
                              thread const float* control,
                              device const uint8_t* model_params,
                              device const uint8_t* cost_params);

/// Terminal cost at end of horizon (must not mutate state).
using MppiTerminalCostFn = float(thread const uint8_t* state,
                                 device const uint8_t* model_params,
                                 device const uint8_t* cost_params);

// ---------------------------------------------------------------------------
// Thread-local array size limits.
// ---------------------------------------------------------------------------

constant uint MPPI_MAX_STATE_BYTES = 128;
constant uint MPPI_MAX_CONTROL_DIM = 16;

// ---------------------------------------------------------------------------
// Visible function table slot indices (fixed by library contract).
// ---------------------------------------------------------------------------

constant uint MPPI_FN_DYNAMICS      = 0;
constant uint MPPI_FN_STAGE_COST    = 1;
constant uint MPPI_FN_TERMINAL_COST = 2;
constant uint MPPI_FN_TABLE_SIZE    = 3;

#endif // MPPI_CORE_H
