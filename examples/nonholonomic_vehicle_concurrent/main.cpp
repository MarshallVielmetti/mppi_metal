#include "mppi_metal/mppi_driver.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <sstream>

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
    // Basic MPPI Configuration
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

    // Provide paths to the compiled shader
    mppi_metal::ModelPluginSpec model_spec;
    model_spec.user_metallib_path = METALLIB_PATH;
    model_spec.callables.dynamics = "mppi_dynamics";
    model_spec.callables.stage_cost = "mppi_stage_cost";
    model_spec.callables.terminal_cost = "mppi_terminal_cost";
    
    model_spec.param_layout.model_params_size = sizeof(ModelParams);
    model_spec.param_layout.cost_params_size = sizeof(CostParams);

    std::string error;
    // We will initialize drivers inside the threads.

    // Set dynamic parameters for model and cost
    ModelParams m_params;
    m_params.v_max = 2.0f;
    m_params.dt = 0.05f;

    CostParams c_params;
    c_params.x_goal = 5.0f;
    c_params.y_goal = 5.0f;
    c_params.goal_radius = 0.2f;   // Termination radius
    c_params.pos_weight = 10.0f;
    c_params.heading_weight = 0.0f;
    c_params.control_weight = 0.1f;

    mppi_metal::ProblemParams params;
    params.model_params = {reinterpret_cast<uint8_t*>(&m_params), sizeof(ModelParams)};
    params.cost_params = {reinterpret_cast<uint8_t*>(&c_params), sizeof(CostParams)};

    // no local error redefine need

    std::ofstream csv("trajectory.csv");
    csv << "sim_id,step,x,y,theta,v,a_cmd,omega_cmd,best_cost\n";

    std::cout << "Starting 100 simulations to reach (" << c_params.x_goal << ", " << c_params.y_goal << ")...\n";

    std::normal_distribution<float> a_noise(0.0f, 0.2f);
    std::normal_distribution<float> omega_noise(0.0f, 0.1f);

    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::vector<std::thread> threads;
    std::atomic<int> next_sim{0};
    std::mutex csv_mutex;

    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            // Each thread needs its own independent Driver and context
            mppi_metal::MPPIDriver local_driver(config);
            std::string local_error;
            if (!local_driver.initialize(model_spec, &local_error)) {
                std::cerr << "Thread init failed: " << local_error << "\n";
                return;
            }
            if (!local_driver.set_problem_params(params, &local_error)) {
                std::cerr << "Thread param failed: " << local_error << "\n";
                return;
            }

            while (true) {
                int sim = next_sim.fetch_add(1);
                if (sim >= 1000) break;

                std::vector<float> state = {0.0f, 0.0f, 0.0f, 0.0f}; // [x, y, theta, v]
                local_driver.reset(nullptr);
                std::stringstream local_csv;

                std::vector<float> states_out(300 * 4);
                std::vector<float> controls_out(300 * 2);
                std::vector<float> costs_out(300);

                mppi_metal::SimulationResults results;
                results.states_out = states_out.data();
                results.controls_out = controls_out.data();
                results.costs_out = costs_out.data();
                results.num_steps = 300;

                mppi_metal::StepInputs inputs;
                inputs.x0 = {state.data(), static_cast<uint32_t>(state.size())};

                if (!local_driver.simulate(300, inputs, results, &local_error)) {
                    std::cerr << "Sim " << sim << " failed: " << local_error << "\n";
                    break;
                }

                int steps_taken = 300;
                for (int step = 0; step < 300; ++step) {
                    float x = states_out[step * 4 + 0];
                    float y = states_out[step * 4 + 1];
                    float theta = states_out[step * 4 + 2];
                    float v = states_out[step * 4 + 3];
                    float a_cmd = controls_out[step * 2 + 0];
                    float omega_cmd = controls_out[step * 2 + 1];
                    float best_cost = costs_out[step];

                    local_csv << sim << "," << step << "," << x << "," << y << "," << theta << "," << v 
                              << "," << a_cmd << "," << omega_cmd << "," << best_cost << "\n";

                    float dx = x - c_params.x_goal;
                    float dy = y - c_params.y_goal;
                    float dist = std::sqrt(dx*dx + dy*dy);
                    if (dist < c_params.goal_radius) {
                        steps_taken = step + 1;
                        break;
                    }
                }

                // Batch write to the actual CSV file
                std::lock_guard<std::mutex> lock(csv_mutex);
                csv << local_csv.str();
                std::cout << "Finished simulation " << sim << " in " << steps_taken << " steps\n";
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    csv.close();
    std::cout << "Done. Trajectory data logged to trajectory.csv\n";
    return 0;
}
