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

                std::mt19937 rng(42 + sim); // Unique seed per simulation
                std::vector<float> state = {0.0f, 0.0f, 0.0f, 0.0f}; // [x, y, theta, v]
                
                std::vector<float> u_out(config.horizon * config.control_dim, 0.0f);
                mppi_metal::MutableControlSequenceView out_view;
                out_view.data = u_out.data();
                out_view.horizon = config.horizon;
                out_view.control_dim = config.control_dim;
                
                local_driver.reset(nullptr);
                std::stringstream local_csv;

                int max_steps = 300;
                for (int step = 0; step < max_steps; ++step) {
                    mppi_metal::StepInputs inputs;
                    inputs.x0 = {state.data(), static_cast<uint32_t>(state.size())};

                    mppi_metal::StepDiagnostics diag;
                    if (!local_driver.step(inputs, out_view, &diag, &local_error)) {
                        std::cerr << "Sim " << sim << " Step " << step << " failed: " << local_error << "\n";
                        break;
                    }

                    float a_cmd = u_out[0];
                    float omega_cmd = u_out[1];

                    local_csv << sim << "," << step << "," << state[0] << "," << state[1] << "," << state[2] << "," << state[3] 
                              << "," << a_cmd << "," << omega_cmd << "," << diag.best_cost << "\n";

                    float a_real = a_cmd + a_noise(rng);
                    float omega_real = omega_cmd + omega_noise(rng);

                    float dt = m_params.dt;
                    state[0] += state[3] * std::cos(state[2]) * dt;
                    state[1] += state[3] * std::sin(state[2]) * dt;
                    state[2] += omega_real * dt;
                    state[3] += a_real * dt;

                    float pi = std::acos(-1.0f);
                    state[2] = std::fmod(state[2] + pi, 2.0f * pi);
                    if (state[2] < 0) state[2] += 2.0f * pi;
                    state[2] -= pi;

                    if (state[3] > m_params.v_max) state[3] = m_params.v_max;
                    if (state[3] < -m_params.v_max) state[3] = -m_params.v_max;

                    float dx = state[0] - c_params.x_goal;
                    float dy = state[1] - c_params.y_goal;
                    float dist = std::sqrt(dx*dx + dy*dy);
                    if (dist < c_params.goal_radius) {
                        break;
                    }

                    for (size_t t = 0; t < config.horizon - 1; ++t) {
                        u_out[t*2] = u_out[(t+1)*2];
                        u_out[t*2 + 1] = u_out[(t+1)*2 + 1];
                    }
                    u_out[(config.horizon-1)*2] = 0.0f;
                    u_out[(config.horizon-1)*2 + 1] = 0.0f;
                    
                    mppi_metal::ControlSequenceView nominal;
                    nominal.data = u_out.data();
                    nominal.horizon = config.horizon;
                    nominal.control_dim = config.control_dim;
                    local_driver.reset(nominal, nullptr);
                }

                // Batch write to the actual CSV file
                std::lock_guard<std::mutex> lock(csv_mutex);
                csv << local_csv.str();
                std::cout << "Finished simulation " << sim << "\n";
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
