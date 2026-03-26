#include <metal_stdlib>
#include "mppi_core.h"
#include "philox.h"

using namespace metal;

struct ModelParams {
    float v_max;
    float dt;

    // Adversary parameters
    float adv_v_max;
    float adv_fire_prob_max;
    float adv_max_range;
    float adv_min_range;
    float adv_k_gain;
    float adv_capture_radius;
    int adv_max_steps;
    int num_adversaries;    // number of adversaries < 32
};

struct CostParams {
    float x_goal;
    float y_goal;
    float goal_radius;
    float pos_weight;
    float heading_weight;
    float control_weight;

    // Adversary parameters
    float adv_collision_weight;
    float adv_dist_weight;
};

struct Adversary {
    float x;
    float y;
    float theta;
    float v;
    uint16_t steps_since_active;
    bool active;
    bool terminal;
};

struct AgentStateView {
    thread float& x;
    thread float& y;
    thread float& theta;
    thread float& v;
    thread bool& captured;
    thread Adversary* adversaries;

    AgentStateView(thread uint8_t* raw) :
        x(*(thread float*)(raw + 0)),
        y(*(thread float*)(raw + 4)),
        theta(*(thread float*)(raw + 8)),
        v(*(thread float*)(raw + 12)),
        captured(*(thread bool*)(raw + 16)),
        adversaries((thread Adversary*)(raw + 20)) {}
};

struct ConstAgentStateView {
    thread const float& x;
    thread const float& y;
    thread const float& theta;
    thread const float& v;
    thread const Adversary* adversaries;

    ConstAgentStateView(thread const uint8_t* raw) :
        x(*(thread const float*)(raw + 0)),
        y(*(thread const float*)(raw + 4)),
        theta(*(thread const float*)(raw + 8)),
        v(*(thread const float*)(raw + 12)),
        adversaries((thread const Adversary*)(raw + 16)) {}
};

// Validation Logic to ensure everything stays in sync
static_assert(sizeof(Adversary) == 20, "Adversary size unexpected");
static_assert(sizeof(AgentStateView) == 48, "AgentStateView should only be a pointer/reference container");

// Ego Dynamics: Kinematic Bicycle Model / Non-holonomic vehicle
// Adversary Dynamics: Pure-pursuit (non-holonomic), constant velocity
// Control: [a, omega]
[[visible]]
void mppi_dynamics(
    thread uint8_t* state_raw,
    thread const float* control,
    device const uint8_t* model_params_raw,
    uint2 rng_counter,
    uint rng_seed
) {
    device const ModelParams* params = (device const ModelParams*)model_params_raw;
    AgentStateView state(state_raw);

    float dt = params->dt;
    float theta = state.theta;
    float v = state.v;
    
    float a = control[0];
    float omega = control[1];
    
    state.x += v * cos(theta) * dt;
    state.y += v * sin(theta) * dt;
    state.theta += omega * dt;
    state.v += a * dt;

    // Update adversaries
    for (int i = 0; i < params->num_adversaries; ++i) {
        if (state.adversaries[i].active) {
            thread Adversary& adv = state.adversaries[i];

            // Pure pursuit controller: steer towards ego agent
            float dx = state.x - adv.x;
            float dy = state.y - adv.y;
            float target_theta = atan2(dy, dx);

            // Check if captured
            float d2 = dx*dx + dy*dy;
            if (d2 < params->adv_capture_radius * params->adv_capture_radius) {
                state.captured = true;
            }
            
            // Normalize angle error to [-pi, pi]
            float pi = 3.1415926535f;
            float error = fmod(target_theta - adv.theta + pi, 2.0f * pi);
            if (error < 0) error += 2.0f * pi;
            error -= pi;
            
            float k = params->adv_k_gain; // steering gain
            float adv_omega = k * error;
            
            // Non-holonomic dynamics
            adv.x     += adv.v * cos(adv.theta) * dt;
            adv.y     += adv.v * sin(adv.theta) * dt;
            adv.theta += adv_omega * dt;
            // adv.v remains constant

            // Increment steps since active
            state.adversaries[i].steps_since_active++;

            if (state.adversaries[i].steps_since_active > params->adv_max_steps) {
                state.adversaries[i].active = false;
                state.adversaries[i].terminal = true;
            }
        } else if (!state.adversaries[i].terminal) {
            // Probabilistically activate WEZ
            float dx = state.x - state.adversaries[i].x;
            float dy = state.y - state.adversaries[i].y;
            float dist = sqrt(dx*dx + dy*dy);
            
            // Activation threshold logic (example: more likely as ego gets closer)
            // Assumes max range of 5.0 for demonstration, easily parametrized later.
            if (dist < params->adv_max_range) {
                // Linear probability scaling: adv_fire_prob_max when dist <= adv_min_range, 
                // scaling down to 0 at adv_max_range.
                float p = params->adv_fire_prob_max;
                if (dist > params->adv_min_range) {
                    float range_delta = params->adv_max_range - params->adv_min_range;
                    if (range_delta > 1e-6f) { // Avoid division by zero
                        p *= (params->adv_max_range - dist) / range_delta;
                    } else {
                        p = 0.0f; // if max <= min, prob is 0 outside min
                    }
                }

                // Ensure unique stream for each WEZ agent using offset
                uint2 wez_ctr = uint2(rng_counter.x, rng_counter.y + 0x1000 + i); 
                float u = uint_to_01(philox2x32_10(wez_ctr, rng_seed).x);
                
                if (u < p) {
                    state.adversaries[i].active = true;
                    state.adversaries[i].steps_since_active = 0;
                }
            }
        }
    }

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
    thread const uint8_t* state_raw,
    thread const float* control,
    device const uint8_t* model_params_raw,
    device const uint8_t* cost_params_raw
) {
    ConstAgentStateView state(state_raw);
    device const ModelParams* model_params = (device const ModelParams*)model_params_raw;
    device const CostParams* params = (device const CostParams*)cost_params_raw;
    
    float dx = state.x - params->x_goal;
    float dy = state.y - params->y_goal;
    float dist2 = dx*dx + dy*dy;
    
    float cost = params->pos_weight * dist2;
    cost += params->control_weight * (control[0]*control[0] + control[1]*control[1]);
    
    // Optional: add a small penalty on velocity to encourage stopping at goal?
    // This depends on desired behavior. We'll stick to positional cost.

    // Adversary collision cost
    for (int i = 0; i < model_params->num_adversaries; ++i) {
        if (state.adversaries[i].active) {
            thread const Adversary& adv = state.adversaries[i];
            float d_adv_x = state.x - adv.x;
            float d_adv_y = state.y - adv.y;
            float dist_to_adv2 = d_adv_x*d_adv_x + d_adv_y*d_adv_y;

            // Gaussian collision cost
            float r2 = model_params->adv_capture_radius * model_params->adv_capture_radius + 1e-6f;
            cost += params->adv_collision_weight * exp(-dist_to_adv2 / r2);
        }
    }
    
    return cost;
}

[[visible]]
float mppi_terminal_cost(
    thread const uint8_t* state_raw,
    device const uint8_t* model_params_raw,
    device const uint8_t* cost_params_raw
) {
    ConstAgentStateView state(state_raw);
    device const CostParams* params = (device const CostParams*)cost_params_raw;
    float dx = state.x - params->x_goal;
    float dy = state.y - params->y_goal;
    
    // Terminal cost is heavily weighted to encourage completion
    return params->pos_weight * (dx*dx + dy*dy) * 10.0f;
}
