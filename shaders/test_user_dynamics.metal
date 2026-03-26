#include <metal_stdlib>
#include "mppi_core.h"

using namespace metal;

// Simple integrator dynamics for testing.
// State: [px, py, vx, vy], Control: [ax, ay], dt = 0.1
[[visible]]
void mppi_dynamics(
    thread float* state,
    thread const float* control,
    device const uint8_t* model_params
) {
    float dt = 0.1f;
    state[0] += state[2] * dt;  // px += vx * dt
    state[1] += state[3] * dt;  // py += vy * dt
    state[2] += control[0] * dt; // vx += ax * dt
    state[3] += control[1] * dt; // vy += ay * dt
}

// Stage cost: penalize distance from origin.
[[visible]]
float mppi_stage_cost(
    thread const float* state,
    thread const float* control,
    device const uint8_t* model_params,
    device const uint8_t* cost_params
) {
    return state[0] * state[0] + state[1] * state[1];
}

// Terminal cost: also penalize distance.
[[visible]]
float mppi_terminal_cost(
    thread const float* state,
    device const uint8_t* model_params,
    device const uint8_t* cost_params
) {
    return state[0] * state[0] + state[1] * state[1];
}
