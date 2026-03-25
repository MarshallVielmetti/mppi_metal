#ifndef PHILOX_H
#define PHILOX_H

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Philox-2x32-10 counter-based RNG.
//
// Deterministic: same (counter, key) always produces the same output,
// regardless of thread dispatch order.
//
// Usage for MPPI:
//   key     = global seed
//   ctr[0]  = step_index
//   ctr[1]  = unique per (sample, timestep, dim_pair)
// ---------------------------------------------------------------------------

constant uint PHILOX_M2x32 = 0xD256D193u;
constant uint PHILOX_W32   = 0x9E3779B9u;

inline uint2 philox_round(uint2 ctr, uint key) {
    ulong product = ulong(ctr.x) * ulong(PHILOX_M2x32);
    uint hi = uint(product >> 32);
    uint lo = uint(product);
    return uint2(hi ^ key ^ ctr.y, lo);
}

/// Philox-2x32-10: produces 2 pseudo-random uint32 values.
inline uint2 philox2x32_10(uint2 counter, uint key) {
    for (int i = 0; i < 10; i++) {
        counter = philox_round(counter, key);
        key += PHILOX_W32;
    }
    return counter;
}

/// Convert uint32 to float in (0, 1).
inline float uint_to_01(uint x) {
    return (float(x) + 0.5f) * (1.0f / 4294967296.0f);
}

/// Box-Muller: convert 2 uniform samples → 2 standard normal samples.
inline float2 box_muller(uint2 u) {
    float u1 = uint_to_01(u.x);
    float u2 = uint_to_01(u.y);
    float r = sqrt(-2.0f * log(u1));
    float theta = 2.0f * M_PI_F * u2;
    return float2(r * cos(theta), r * sin(theta));
}

#endif // PHILOX_H
