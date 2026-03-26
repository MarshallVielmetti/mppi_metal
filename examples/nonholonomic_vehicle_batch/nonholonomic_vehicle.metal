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

struct AgentState {
    float x;
    float y;
    float theta;
    float v;
};

// Dynamics: Kinematic Bicycle Model / Non-holonomic vehicle
// State: [x, y, theta, v]
// Control: [a, omega]
[[visible]]
void mppi_dynamics(
    thread uint8_t* state_raw,
    thread const float* control,
    device const uint8_t* model_params_raw,
    uint2 rng_counter,
    uint rng_seed
) {
    thread AgentState& state = *(thread AgentState*)state_raw;
    device const ModelParams* params = (device const ModelParams*)model_params_raw;
    float dt = params->dt;
    
    float theta = state.theta;
    float v = state.v;
    
    float a = control[0];
    float omega = control[1];
    
    state.x += v * cos(theta) * dt;
    state.y += v * sin(theta) * dt;
    state.theta += omega * dt;
    state.v += a * dt;
    
    // Wrap theta to [-pi, pi]
    float pi = 3.1415926535f;
    state.theta = fmod(state.theta + pi, 2.0f * pi);
    if (state.theta < 0) state.theta += 2.0f * pi;
    state.theta -= pi;
    
    // Clamp velocity
    if (state.v > params->v_max) state.v = params->v_max;
    if (state.v < -params->v_max) state.v = -params->v_max;
}

[[visible]]
float mppi_stage_cost(
    thread uint8_t* state_raw,
    thread const float* control,
    device const uint8_t* model_params_raw,
    device const uint8_t* cost_params_raw
) {
    thread AgentState& state = *(thread AgentState*)state_raw;
    device const CostParams* params = (device const CostParams*)cost_params_raw;
    
    float dx = state.x - params->x_goal;
    float dy = state.y - params->y_goal;
    float dist2 = dx*dx + dy*dy;
    
    float cost = params->pos_weight * dist2;
    cost += params->control_weight * (control[0]*control[0] + control[1]*control[1]);
    
    // Optional: add a small penalty on velocity to encourage stopping at goal?
    // This depends on desired behavior. We'll stick to positional cost.
    
    return cost;
}

[[visible]]
float mppi_terminal_cost(
    thread uint8_t* state_raw,
    device const uint8_t* model_params_raw,
    device const uint8_t* cost_params_raw
) {
    thread AgentState& state = *(thread AgentState*)state_raw;
    device const CostParams* params = (device const CostParams*)cost_params_raw;
    float dx = state.x - params->x_goal;
    float dy = state.y - params->y_goal;
    
    // Terminal cost is heavily weighted to encourage completion
    return params->pos_weight * (dx*dx + dy*dy) * 10.0f;
}
