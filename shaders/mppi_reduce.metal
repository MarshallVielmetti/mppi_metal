#include <metal_stdlib>
#include "mppi_core.h"
#include "philox.h"

using namespace metal;

// ---------------------------------------------------------------------------
// Pass 2: Weight Normalization kernel.
// Finds the minimum cost, calculates the exponential weights, and normalizes them.
// Output diagnostics: [min_cost, mean_cost, effective_sample_size].
// Dispatched as exactly 1 threadgroup.
// ---------------------------------------------------------------------------

[[kernel]]
void mppi_compute_weights(
    device const float* costs        [[buffer(0)]],
    device float*       weights      [[buffer(1)]],
    device float*       diagnostics  [[buffer(2)]],
    constant uint&      sample_count [[buffer(3)]],
    constant float&     lambda       [[buffer(4)]],
    uint lid [[thread_index_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    // 1. Thread-local minimum
    float local_min = 1e38f;
    for (uint i = lid; i < sample_count; i += tgs) {
        local_min = min(local_min, costs[i]);
    }

    // 2. SIMD-group minimum
    float simd_min_val = simd_min(local_min);

    // 3. Threadgroup minimum
    threadgroup float shared_mins[32]; // Max 1024 threads / 32 = 32 SIMD groups
    if (simd_lane_id == 0) {
        shared_mins[simd_group_id] = simd_min_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_min = 1e38f;
    if (simd_group_id == 0) {
        float val = (simd_lane_id < (tgs + 31) / 32) ? shared_mins[simd_lane_id] : 1e38f;
        global_min = simd_min(val);
        if (simd_lane_id == 0) shared_mins[0] = global_min;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_min = shared_mins[0];

    // 4. Thread-local weight sum
    float local_weight_sum = 0.0f;
    float local_cost_sum = 0.0f;
    for (uint i = lid; i < sample_count; i += tgs) {
        float cost = costs[i];
        float w = exp(-(cost - global_min) / lambda);
        weights[i] = w; // Store unnormalized
        local_weight_sum += w;
        local_cost_sum += cost;
    }

    // 5. SIMD-group sums
    float simd_w_sum = simd_sum(local_weight_sum);
    float simd_c_sum = simd_sum(local_cost_sum);

    threadgroup float shared_w_sums[32];
    threadgroup float shared_c_sums[32];

    if (simd_lane_id == 0) {
        shared_w_sums[simd_group_id] = simd_w_sum;
        shared_c_sums[simd_group_id] = simd_c_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_w_sum = 0.0f;
    float global_c_sum = 0.0f;
    if (simd_group_id == 0) {
        float w_val = (simd_lane_id < (tgs + 31) / 32) ? shared_w_sums[simd_lane_id] : 0.0f;
        float c_val = (simd_lane_id < (tgs + 31) / 32) ? shared_c_sums[simd_lane_id] : 0.0f;
        global_w_sum = simd_sum(w_val);
        global_c_sum = simd_sum(c_val);
        if (simd_lane_id == 0) {
            shared_w_sums[0] = global_w_sum;
            shared_c_sums[0] = global_c_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_w_sum = shared_w_sums[0];
    global_c_sum = shared_c_sums[0];

    // 6. Normalize and compute ESS
    float local_w2_sum = 0.0f;
    for (uint i = lid; i < sample_count; i += tgs) {
        float w = weights[i] / global_w_sum;
        weights[i] = w;
        local_w2_sum += w * w;
    }

    float simd_w2_sum = simd_sum(local_w2_sum);
    threadgroup float shared_w2_sums[32];
    if (simd_lane_id == 0) {
        shared_w2_sums[simd_group_id] = simd_w2_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float global_w2_sum = 0.0f;
    if (simd_group_id == 0) {
        float val = (simd_lane_id < (tgs + 31) / 32) ? shared_w2_sums[simd_lane_id] : 0.0f;
        global_w2_sum = simd_sum(val);
        if (simd_lane_id == 0) shared_w2_sums[0] = global_w2_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_w2_sum = shared_w2_sums[0];

    // 7. Write diagnostics
    if (lid == 0) {
        diagnostics[0] = global_min;
        diagnostics[1] = global_c_sum / (float)sample_count;
        diagnostics[2] = 1.0f / global_w2_sum;
    }
}

// ---------------------------------------------------------------------------
// Pass 3: Reduction kernel.
// Each thread handles one output control element (t, d).
// Regenerates noise via Philox for all S samples and computes the
// weighted average: u_out[i] = u_nominal[i] + Σ_k w_k * noise_k(i).
// ---------------------------------------------------------------------------

[[kernel]]
void mppi_reduce(
    device const float*  weights       [[buffer(0)]],  // [S]
    device const float*  u_nominal     [[buffer(1)]],  // [H * cdim]
    device float*        u_out         [[buffer(2)]],  // [H * cdim]
    device const float*  sigma         [[buffer(3)]],  // [cdim]
    device const float*  u_min         [[buffer(4)]],  // [cdim]
    device const float*  u_max         [[buffer(5)]],  // [cdim]
    constant uint&       sample_count  [[buffer(6)]],
    constant uint&       horizon       [[buffer(7)]],
    constant uint&       control_dim   [[buffer(8)]],
    constant uint&       rng_seed      [[buffer(9)]],
    constant uint&       step_index    [[buffer(10)]],
    uint gid [[thread_position_in_grid]]
) {
    // gid indexes a flat (t, d) pair: gid = t * control_dim + d.
    uint seq_len = horizon * control_dim;
    if (gid >= seq_len) return;

    uint t = gid / control_dim;
    uint d = gid % control_dim;
    uint pairs_per_step = (control_dim + 1) / 2;
    uint pair_idx = t * pairs_per_step + d / 2;
    // Which element of the Box-Muller pair: 0 → gauss.x, 1 → gauss.y.
    bool is_second = (d % 2) == 1;

    float weighted_noise = 0.0f;

    for (uint k = 0; k < sample_count; k++) {
        // Regenerate the same Philox counter used in the rollout kernel.
        uint2 ctr = uint2(step_index, k * horizon * pairs_per_step + pair_idx);
        float2 gauss = box_muller(philox2x32_10(ctr, rng_seed));
        float eps = (is_second ? gauss.y : gauss.x) * sigma[d];
        weighted_noise += weights[k] * eps;
    }

    float u = u_nominal[gid] + weighted_noise;
    u_out[gid] = clamp(u, u_min[d], u_max[d]);
}
