#include "mppi_metal/mppi_driver.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

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

int main() {
    // ---- MPPI Configuration ----
    mppi_metal::DriverConfig config;
    config.state_dim = 4;
    config.control_dim = 2;
    config.horizon = 20;
    config.sample_count = 256;
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

    CostParams c_params;
    c_params.x_goal = 5.0f;
    c_params.y_goal = 5.0f;
    c_params.goal_radius = 0.2f;
    c_params.pos_weight = 10.0f;
    c_params.heading_weight = 0.0f;
    c_params.control_weight = 0.1f;

    mppi_metal::ProblemParams params;
    params.model_params = {reinterpret_cast<uint8_t*>(&m_params), sizeof(ModelParams)};
    params.cost_params = {reinterpret_cast<uint8_t*>(&c_params), sizeof(CostParams)};

    if (!driver.set_problem_params(params, &error)) {
        std::cerr << "set_problem_params failed: " << error << "\n";
        return 1;
    }

    // ---- Batched Simulation ----
    const uint32_t N_AGENTS = 1000;
    const uint32_t N_STEPS = 85;
    const uint32_t SDIM = 4;
    const uint32_t CDIM = 2;

    // All agents start at origin: [x=0, y=0, theta=0, v=0]
    std::vector<float> x0_packed(N_AGENTS * SDIM, 0.0f);

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
    csv << "sim_id,step,x,y,theta,v,a_cmd,omega_cmd,best_cost\n";

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

            csv << a << "," << step << "," << x << "," << y << ","
                << theta << "," << v << "," << a_cmd << ","
                << omega_cmd << "," << best_cost << "\n";

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
