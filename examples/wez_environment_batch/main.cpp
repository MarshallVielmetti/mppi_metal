#include "mppi_metal/mppi_driver.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

#define N_ADVERSARIES 8

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
    float x, y, theta, v;
    uint16_t steps_since_active;
    bool active;
    bool terminal;
};

struct AgentState {
    float x, y, theta, v;
    bool captured;
    Adversary adversaries[N_ADVERSARIES]; // Matches N_ADVERSARIES
};

int main() {

    const uint32_t N_AGENTS = 256;
    const uint32_t N_STEPS = 100;
    const uint32_t EGO_SDIM = 5; // 4 floats + 1 bool, struct packing makes this 5 floats
    const uint32_t CDIM = 2;

    // 4 floats + 1 bool + 1 uint16_t -> Packed into 5 'floats' worth of space by struct rules
    const uint32_t ADVERSARY_SDIM = 5; 

    const uint32_t SDIM = EGO_SDIM + N_ADVERSARIES * ADVERSARY_SDIM;
    static_assert(sizeof(AgentState) == SDIM * sizeof(float), "AgentState layout mismatch");

    // ---- MPPI Configuration ----
    mppi_metal::DriverConfig config;
    config.state_dim = SDIM;
    config.control_dim = CDIM;
    config.horizon = 20;
    config.sample_count = 128;
    config.lambda = 1.0f;

    // Control Noise: [acceleration, omega]
    std::vector<float> sigma = {2.0f, 1.0f};
    std::vector<float> sigma_inv = {1.0f/2.0f, 1.0f/1.0f};

    config.noise.covariance_form = mppi_metal::ControlNoiseConfig::CovarianceForm::kDiagonal;
    config.noise.sigma = sigma.data();
    config.noise.sigma_inv = sigma_inv.data();
    config.noise.control_dim = 2;

    // Control Bounds
    float a_max = 3.0f;
    float omega_max = 1.5f;
    std::vector<float> u_min = {-a_max, -omega_max};
    std::vector<float> u_max = {a_max, omega_max};

    config.bounds.u_min = u_min.data();
    config.bounds.u_max = u_max.data();
    config.bounds.control_dim = 2;

    mppi_metal::MPPIDriver driver(config);

    // ---- Model Plugin ----
    mppi_metal::ModelPluginSpec model_spec;
    model_spec.user_metallib_path = METALLIB_PATH;
    model_spec.callables.dynamics = "mppi_dynamics";
    model_spec.callables.stage_cost = "mppi_stage_cost";
    model_spec.callables.terminal_cost = "mppi_terminal_cost";

    model_spec.param_layout.model_params_size = sizeof(ModelParams);
    model_spec.param_layout.cost_params_size = sizeof(CostParams);

    std::string error;
    if (!driver.initialize(model_spec, &error)) {
        std::cerr << "Initialize failed: " << error << "\n";
        return 1;
    }

    // ---- Dynamic Parameters ----
    ModelParams m_params;
    m_params.v_max = 2.0f;
    m_params.dt = 0.05f;

    // Adversary parameters
    m_params.adv_v_max = 2.0f;
    m_params.adv_max_steps = 100;
    m_params.num_adversaries = N_ADVERSARIES;
    m_params.adv_fire_prob_max = 0.1f;
    m_params.adv_max_range = 8.0f;
    m_params.adv_min_range = 3.0f;
    m_params.adv_k_gain = 3.0f;
    m_params.adv_capture_radius = 0.5f;

    CostParams c_params;
    c_params.x_goal = 5.0f;
    c_params.y_goal = 5.0f;
    c_params.goal_radius = 0.2f;
    c_params.pos_weight = 10.0f;
    c_params.heading_weight = 0.0f;
    c_params.control_weight = 0.1f;

    // Adversary parameters
    c_params.adv_collision_weight = 100.0f;
    c_params.adv_dist_weight = 0.0f;

    mppi_metal::ProblemParams params;
    params.model_params = {reinterpret_cast<uint8_t*>(&m_params), sizeof(ModelParams)};
    params.cost_params = {reinterpret_cast<uint8_t*>(&c_params), sizeof(CostParams)};

    if (!driver.set_problem_params(params, &error)) {
        std::cerr << "set_problem_params failed: " << error << "\n";
        return 1;
    }

    // All agents start at origin: [x=0, y=0, theta=0, v=0]
    // Adversaries start at different locations.
    std::vector<float> x0_packed(N_AGENTS * SDIM, 0.0f);
    for (uint32_t a = 0; a < N_AGENTS; ++a) {
        AgentState* s = reinterpret_cast<AgentState*>(&x0_packed[a * SDIM]);
        // Ego: [x,y,th,v] = 0 (already zeroed)
        
        // Adversary 1
        s->adversaries[0].x = 2.0f;
        s->adversaries[0].y = 0.0f;
        s->adversaries[0].theta = 3.14f;
        s->adversaries[0].v = m_params.adv_v_max;
        s->adversaries[0].active = false;
        s->adversaries[0].terminal = false;
        s->adversaries[0].steps_since_active = 0;
        
        // Adversary 2
        s->adversaries[1].x = 0.0f;
        s->adversaries[1].y = 2.0f;
        s->adversaries[1].theta = -1.57f;
        s->adversaries[1].v = m_params.adv_v_max;
        s->adversaries[1].active = false;
        s->adversaries[1].terminal = false;
        s->adversaries[1].steps_since_active = 50;

        // Adversary 3
        s->adversaries[2].x = 0.0f;
        s->adversaries[2].y = 2.0f;
        s->adversaries[2].theta = -1.57f;
        s->adversaries[2].v = m_params.adv_v_max;
        s->adversaries[2].active = false;
        s->adversaries[2].terminal = false;
        s->adversaries[2].steps_since_active = 50;

        // Adversary 4
        s->adversaries[3].x = 0.0f;
        s->adversaries[3].y = 2.0f;
        s->adversaries[3].theta = -1.57f;
        s->adversaries[3].v = m_params.adv_v_max;
        s->adversaries[3].active = false;
        s->adversaries[3].terminal = false;
        s->adversaries[3].steps_since_active = 50;

        // Adversary 5
        s->adversaries[4].x = 0.0f;
        s->adversaries[4].y = 2.0f;
        s->adversaries[4].theta = -1.57f;
        s->adversaries[4].v = m_params.adv_v_max;
        s->adversaries[4].active = false;
        s->adversaries[4].terminal = false;
        s->adversaries[4].steps_since_active = 50;

        // Adversary 6
        s->adversaries[5].x = 0.0f;
        s->adversaries[5].y = 2.0f;
        s->adversaries[5].theta = -1.57f;
        s->adversaries[5].v = m_params.adv_v_max;
        s->adversaries[5].active = false;
        s->adversaries[5].terminal = false;
        s->adversaries[5].steps_since_active = 50;

        // Adversary 7
        s->adversaries[6].x = 0.0f;
        s->adversaries[6].y = 2.0f;
        s->adversaries[6].theta = -1.57f;
        s->adversaries[6].v = m_params.adv_v_max;
        s->adversaries[6].active = false;
        s->adversaries[6].terminal = false;
        s->adversaries[6].steps_since_active = 50;

        // Adversary 8
        s->adversaries[7].x = 0.0f;
        s->adversaries[7].y = 2.0f;
        s->adversaries[7].theta = -1.57f;
        s->adversaries[7].v = m_params.adv_v_max;
        s->adversaries[7].active = false;
        s->adversaries[7].terminal = false;
        s->adversaries[7].steps_since_active = 50;
    }

    // Allocate contiguous output buffers
    std::vector<float> states_out(N_AGENTS * N_STEPS * SDIM);
    std::vector<float> controls_out(N_AGENTS * N_STEPS * CDIM);
    std::vector<float> costs_out(N_AGENTS * N_STEPS);

    mppi_metal::BatchSimulationResults results;
    results.states_out = states_out.data();
    results.controls_out = controls_out.data();
    results.costs_out = costs_out.data();
    results.num_agents = N_AGENTS;
    results.num_steps = N_STEPS;

    std::cout << "Running " << N_AGENTS << " batched agents for "
              << N_STEPS << " steps toward ("
              << c_params.x_goal << ", " << c_params.y_goal << ")...\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    if (!driver.simulate_batch(N_AGENTS, N_STEPS, x0_packed.data(), results, &error)) {
        std::cerr << "simulate_batch failed: " << error << "\n";
        return 1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Batch simulation completed in " << elapsed_ms << " ms\n";
    std::cout << "Throughput: "
              << static_cast<double>(N_AGENTS * N_STEPS) / (elapsed_ms / 1000.0)
              << " agent-steps/sec\n";

    // ---- Write CSV output ----
    std::ofstream csv("trajectory_batch.csv");
    csv << "sim_id,step,x,y,theta,v,a_cmd,omega_cmd,captured,best_cost";
    for (uint32_t i = 0; i < N_ADVERSARIES; ++i) {
        csv << ",x_adv_" << i+1 << ",y_adv_" << i+1 << ",th_adv_" << i+1 << ",v_adv_" << i+1;
    }
    csv << "\n";

    int reached_goal = 0;
    for (uint32_t a = 0; a < N_AGENTS; ++a) {
        int steps_taken = N_STEPS;
        for (uint32_t step = 0; step < N_STEPS; ++step) {
            size_t s_base = a * N_STEPS * SDIM + step * SDIM;
            size_t c_base = a * N_STEPS * CDIM + step * CDIM;
            size_t cost_idx = a * N_STEPS + step;

            float x = states_out[s_base + 0];
            float y = states_out[s_base + 1];
            float theta = states_out[s_base + 2];
            float v = states_out[s_base + 3];
            float a_cmd = controls_out[c_base + 0];
            float omega_cmd = controls_out[c_base + 1];
            float best_cost = costs_out[cost_idx];
            bool captured = states_out[s_base + 4];

            csv << a << "," << step << "," << x << "," << y << ","
                << theta << "," << v << "," << a_cmd << ","
                << omega_cmd << "," << captured << "," << best_cost;

            for (uint32_t i = 0; i < N_ADVERSARIES; ++i) {
                uint32_t adv_base = s_base + EGO_SDIM + i * ADVERSARY_SDIM;
                csv << "," << states_out[adv_base + 0] 
                    << "," << states_out[adv_base + 1]
                    << "," << states_out[adv_base + 2]
                    << "," << states_out[adv_base + 3];
            }
            csv << "\n";

            float dx = x - c_params.x_goal;
            float dy = y - c_params.y_goal;
            float dist = std::sqrt(dx*dx + dy*dy);
            if (dist < c_params.goal_radius) {
                steps_taken = step + 1;
                break;
            }
        }
        if (steps_taken < static_cast<int>(N_STEPS)) {
            ++reached_goal;
        }
    }

    csv.close();
    std::cout << reached_goal << "/" << N_AGENTS
              << " agents reached the goal\n";
    std::cout << "Trajectory data logged to trajectory_batch.csv\n";
    return 0;
}
