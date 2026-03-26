#include <metal_stdlib>
#include "mppi_core.h"

using namespace metal;

struct ModelParams {
    float v_max;
    float dt;

    // Adversary parameters
    float adv_v_max;
    int adv_max_steps;
    int num_adversaries;    // number of adversaries < 32
    int active_adversaries; // bitmask of active adversaries
};

struct CostParams {
    float x_goal;
    float y_goal;
    float goal_radius;
    float pos_weight;
    float heading_weight;
    float control_weight;

    // Adversary parameters
    float collision_radius;
    float adv_collision_weight;
    float adv_dist_weight;
};

// Ego Dynamics: Kinematic Bicycle Model / Non-holonomic vehicle
// Adversary Dynamics: Pure-pursuit (non-holonomic), constant velocity
// State: [x, y, theta, v, x_adv_1, y_adv_1, th_adv_1, v_adv_1, x_adv_2, y_adv_2, th_adv_2, v_adv_2...]
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

    // Update adversaries
    for (int i = 0; i < params->num_adversaries; ++i) {
        if ((params->active_adversaries >> i) & 1) {
            uint base = 4 + i * 4;
            float adv_x = state[base];
            float adv_y = state[base + 1];
            float adv_theta = state[base + 2];
            float adv_v = state[base + 3];

            // Pure pursuit controller: steer towards ego agent
            float dx = state[0] - adv_x;
            float dy = state[1] - adv_y;
            float target_theta = atan2(dy, dx);
            
            // Normalize angle error to [-pi, pi]
            float error = target_theta - adv_theta;
            while (error >  3.1415926535f) error -= 2.0f * 3.1415926535f;
            while (error < -3.1415926535f) error += 2.0f * 3.1415926535f;
            
            float k = 2.0f; // steering gain
            float adv_omega = k * error;
            
            // Non-holonomic dynamics
            state[base]     += adv_v * cos(adv_theta) * dt;
            state[base + 1] += adv_v * sin(adv_theta) * dt;
            state[base + 2] += adv_omega * dt;
            // state[base + 3] (velocity) remains constant
        }
    }

    
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

    device const ModelParams* model_params = (device const ModelParams*)model_params_raw;
    device const CostParams* params = (device const CostParams*)cost_params_raw;
    
    float dx = state[0] - params->x_goal;
    float dy = state[1] - params->y_goal;
    float dist2 = dx*dx + dy*dy;
    
    float cost = params->pos_weight * dist2;
    cost += params->control_weight * (control[0]*control[0] + control[1]*control[1]);
    
    // Optional: add a small penalty on velocity to encourage stopping at goal?
    // This depends on desired behavior. We'll stick to positional cost.

    // Adversary collision cost
    for (int i = 0; i < model_params->num_adversaries; ++i) {
        if ((model_params->active_adversaries >> i) & 1) {
            uint base = 4 + i * 4;
            float adv_x = state[base];
            float adv_y = state[base + 1];
            float dx = state[0] - adv_x;
            float dy = state[1] - adv_y;
            float dist2 = dx*dx + dy*dy;
            cost += params->adv_collision_weight * exp(-dist2 / (params->collision_radius * params->collision_radius));
        }
    }

    
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
