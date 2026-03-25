#include <metal_stdlib>
#include "mppi_core.h"
#include "philox.h"

using namespace metal;

// ---------------------------------------------------------------------------
// Pass 2: Reduction kernel.
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
