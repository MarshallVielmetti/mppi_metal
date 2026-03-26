#include <metal_stdlib>
#include "mppi_core.h"
#include "philox.h"

using namespace metal;

// ---------------------------------------------------------------------------
// Default no-op cost callables (used when user doesn't provide them).
// ---------------------------------------------------------------------------

[[visible]]
float mppi_default_stage_cost(
    thread const float* state,
    thread const float* control,
    device const uint8_t* cost_params
) {
    return 0.0f;
}

[[visible]]
float mppi_default_terminal_cost(
    thread const float* state,
    device const uint8_t* cost_params
) {
    return 0.0f;
}

// ---------------------------------------------------------------------------
// Pass 1: Rollout kernel.
// Each thread rolls out one sample trajectory and writes its total cost.
// Noise is generated on-GPU via Philox and used in-place (NOT stored).
// ---------------------------------------------------------------------------

[[kernel]]
void mppi_rollout(
    device const float*   x0             [[buffer(0)]],
    device const float*   u_nominal      [[buffer(1)]],
    device const uint8_t* model_params   [[buffer(2)]],
    device const uint8_t* cost_params    [[buffer(3)]],
    device float*         costs_out      [[buffer(4)]],
    constant uint&        state_dim      [[buffer(5)]],
    constant uint&        control_dim    [[buffer(6)]],
    constant uint&        horizon        [[buffer(7)]],
    constant uint&        sample_count   [[buffer(8)]],
    device const float*   u_min          [[buffer(9)]],
    device const float*   u_max          [[buffer(10)]],
    visible_function_table<MppiDynamicsFn>      dynamics_table      [[buffer(11)]],
    visible_function_table<MppiStageCostFn>     stage_cost_table    [[buffer(12)]],
    visible_function_table<MppiTerminalCostFn>  terminal_cost_table [[buffer(13)]],
    device const float*   sigma          [[buffer(14)]],
    constant uint&        rng_seed       [[buffer(15)]],
    constant uint&        step_index     [[buffer(16)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= sample_count) return;

    // Thread-local state (copy from x0).
    float state[MPPI_MAX_STATE_DIM];
    for (uint d = 0; d < state_dim && d < MPPI_MAX_STATE_DIM; d++) {
        state[d] = x0[d];
    }

    uint pairs_per_step = (control_dim + 1) / 2;
    float total_cost = 0.0f;

    for (uint t = 0; t < horizon; t++) {
        float control[MPPI_MAX_CONTROL_DIM];

        // Generate noise via Philox + Box-Muller, 2 dims per call.
        for (uint d = 0; d < control_dim; d += 2) {
            uint pair_idx = t * pairs_per_step + d / 2;
            uint2 ctr = uint2(step_index, gid * horizon * pairs_per_step + pair_idx);
            float2 gauss = box_muller(philox2x32_10(ctr, rng_seed));

            float eps0 = gauss.x * sigma[d];
            control[d] = clamp(u_nominal[t * control_dim + d] + eps0,
                               u_min[d], u_max[d]);

            if (d + 1 < control_dim) {
                float eps1 = gauss.y * sigma[d + 1];
                control[d + 1] = clamp(u_nominal[t * control_dim + d + 1] + eps1,
                                       u_min[d + 1], u_max[d + 1]);
            }
        }

        dynamics_table[0](state, control, model_params);
        total_cost += stage_cost_table[0](state, control, cost_params);
    }

    total_cost += terminal_cost_table[0](state, cost_params);
    costs_out[gid] = total_cost;
}

// ---------------------------------------------------------------------------
// Batch variant: each agent gets its own x0, u_nominal, costs slice.
// Grid: (S, N, 1).  gid.y = agent_idx.
// ---------------------------------------------------------------------------

[[kernel]]
void mppi_rollout_batch(
    device const float*   x0_packed      [[buffer(0)]],   // [N * sdim]
    device const float*   u_nom_packed   [[buffer(1)]],   // [N * H * cdim]
    device const uint8_t* model_params   [[buffer(2)]],
    device const uint8_t* cost_params    [[buffer(3)]],
    device float*         costs_out      [[buffer(4)]],   // [N * S]
    constant uint&        state_dim      [[buffer(5)]],
    constant uint&        control_dim    [[buffer(6)]],
    constant uint&        horizon        [[buffer(7)]],
    constant uint&        sample_count   [[buffer(8)]],
    device const float*   u_min          [[buffer(9)]],
    device const float*   u_max          [[buffer(10)]],
    visible_function_table<MppiDynamicsFn>      dynamics_table      [[buffer(11)]],
    visible_function_table<MppiStageCostFn>     stage_cost_table    [[buffer(12)]],
    visible_function_table<MppiTerminalCostFn>  terminal_cost_table [[buffer(13)]],
    device const float*   sigma          [[buffer(14)]],
    constant uint&        rng_seed       [[buffer(15)]],
    constant uint&        step_index     [[buffer(16)]],
    constant uint&        num_agents     [[buffer(17)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint sample_idx = gid.x;
    uint agent_idx  = gid.y;
    if (sample_idx >= sample_count || agent_idx >= num_agents) return;

    uint agent_key = rng_seed + agent_idx;

    // Per-agent offsets.
    device const float* x0       = x0_packed + agent_idx * state_dim;
    device const float* u_nominal = u_nom_packed + agent_idx * horizon * control_dim;

    float state[MPPI_MAX_STATE_DIM];
    for (uint d = 0; d < state_dim && d < MPPI_MAX_STATE_DIM; d++) {
        state[d] = x0[d];
    }

    uint pairs_per_step = (control_dim + 1) / 2;
    float total_cost = 0.0f;

    for (uint t = 0; t < horizon; t++) {
        float control[MPPI_MAX_CONTROL_DIM];

        for (uint d = 0; d < control_dim; d += 2) {
            uint pair_idx = t * pairs_per_step + d / 2;
            uint2 ctr = uint2(step_index, sample_idx * horizon * pairs_per_step + pair_idx);
            float2 gauss = box_muller(philox2x32_10(ctr, agent_key));

            float eps0 = gauss.x * sigma[d];
            control[d] = clamp(u_nominal[t * control_dim + d] + eps0,
                               u_min[d], u_max[d]);

            if (d + 1 < control_dim) {
                float eps1 = gauss.y * sigma[d + 1];
                control[d + 1] = clamp(u_nominal[t * control_dim + d + 1] + eps1,
                                       u_min[d + 1], u_max[d + 1]);
            }
        }

        dynamics_table[0](state, control, model_params);
        total_cost += stage_cost_table[0](state, control, cost_params);
    }

    total_cost += terminal_cost_table[0](state, cost_params);
    costs_out[agent_idx * sample_count + sample_idx] = total_cost;
}
