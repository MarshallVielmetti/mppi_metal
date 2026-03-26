#include <metal_stdlib>
#include "mppi_core.h"

using namespace metal;

struct ModelParams {
    float v_max;
    float dt;
};

struct CostParams {
    float x_goal;
    float y_goal;
    float goal_radius;
    float pos_weight;
    float heading_weight;
    float control_weight;
};

// Dynamics: Kinematic Bicycle Model / Non-holonomic vehicle
// State: [x, y, theta, v]
// Control: [a, omega]
[[visible]]
void mppi_dynamics(
    thread float* state,
    thread const float* control,
    device const uint8_t* model_params_raw
) {
    device const ModelParams* params = (device const ModelParams*)model_params_raw;
    float dt = params->dt;
    
    float theta = state[2];
    float v = state[3];
    
    float a = control[0];
    float omega = control[1];
    
    state[0] += v * cos(theta) * dt;
    state[1] += v * sin(theta) * dt;
    state[2] += omega * dt;
    state[3] += a * dt;
    
    // Wrap theta to [-pi, pi]
    float pi = 3.1415926535f;
    state[2] = fmod(state[2] + pi, 2.0f * pi);
    if (state[2] < 0) state[2] += 2.0f * pi;
    state[2] -= pi;
    
    // Clamp velocity
    if (state[3] > params->v_max) state[3] = params->v_max;
    if (state[3] < -params->v_max) state[3] = -params->v_max;
}

[[visible]]
float mppi_stage_cost(
    thread const float* state,
    thread const float* control,
    device const uint8_t* model_params_raw,
    device const uint8_t* cost_params_raw
) {
    device const CostParams* params = (device const CostParams*)cost_params_raw;
    
    float dx = state[0] - params->x_goal;
    float dy = state[1] - params->y_goal;
    float dist2 = dx*dx + dy*dy;
    
    float cost = params->pos_weight * dist2;
    cost += params->control_weight * (control[0]*control[0] + control[1]*control[1]);
    
    // Optional: add a small penalty on velocity to encourage stopping at goal?
    // This depends on desired behavior. We'll stick to positional cost.
    
    return cost;
}

[[visible]]
float mppi_terminal_cost(
    thread const float* state,
    device const uint8_t* model_params_raw,
    device const uint8_t* cost_params_raw
) {
    device const CostParams* params = (device const CostParams*)cost_params_raw;
    float dx = state[0] - params->x_goal;
    float dy = state[1] - params->y_goal;
    
    // Terminal cost is heavily weighted to encourage completion
    return params->pos_weight * (dx*dx + dy*dy) * 10.0f;
}
