#include "mppi_metal/mppi_driver.hpp"
#include "mppi_metal/mppi_types.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#ifndef TEST_USER_METALLIB_PATH
#define TEST_USER_METALLIB_PATH ""
#endif

// ---------------------------------------------------------------------------
// Minimal test framework
// ---------------------------------------------------------------------------

namespace {

int g_pass = 0;
int g_fail = 0;

bool expect_true(bool condition, const std::string& message) {
	if (!condition) {
		std::cerr << "[FAIL] " << message << '\n';
		++g_fail;
		return false;
	}
	std::cout << "[PASS] " << message << '\n';
	++g_pass;
	return true;
}

bool expect_false(bool condition, const std::string& message) {
	return expect_true(!condition, message);
}

// ---------------------------------------------------------------------------
// Helpers: build a valid DriverConfig for use as a baseline.
// ---------------------------------------------------------------------------

struct ValidConfigFixture {
	std::vector<float> sigma;
	std::vector<float> sigma_inv;
	std::vector<float> u_min;
	std::vector<float> u_max;
	mppi_metal::DriverConfig config;

	ValidConfigFixture() {
		const uint32_t cdim = 2;
		sigma = {1.0f, 1.0f};
		sigma_inv = {1.0f, 1.0f};
		u_min = {-1.0f, -1.0f};
		u_max = {1.0f, 1.0f};

		config.state_dim = 4;
		config.control_dim = cdim;
		config.horizon = 10;
		config.sample_count = 64;
		config.lambda = 1.0f;

		config.noise.covariance_form =
			mppi_metal::ControlNoiseConfig::CovarianceForm::kDiagonal;
		config.noise.sigma = sigma.data();
		config.noise.sigma_inv = sigma_inv.data();
		config.noise.control_dim = cdim;

		config.bounds.u_min = u_min.data();
		config.bounds.u_max = u_max.data();
		config.bounds.control_dim = cdim;
	}
};

// ---------------------------------------------------------------------------
// Tests: dimension validation
// ---------------------------------------------------------------------------

void test_zero_state_dim() {
	ValidConfigFixture f;
	f.config.state_dim = 0;

	mppi_metal::MPPIDriver driver(f.config);
	std::string error;
	mppi_metal::ModelPluginSpec model;
	expect_false(driver.initialize(model, &error),
		"initialize should fail for state_dim == 0");
	expect_true(error.find("state_dim") != std::string::npos,
		"error should mention state_dim");
}

void test_zero_control_dim() {
	ValidConfigFixture f;
	f.config.control_dim = 0;

	mppi_metal::MPPIDriver driver(f.config);
	std::string error;
	mppi_metal::ModelPluginSpec model;
	expect_false(driver.initialize(model, &error),
		"initialize should fail for control_dim == 0");
	expect_true(error.find("control_dim") != std::string::npos,
		"error should mention control_dim");
}

void test_zero_horizon() {
	ValidConfigFixture f;
	f.config.horizon = 0;

	mppi_metal::MPPIDriver driver(f.config);
	std::string error;
	mppi_metal::ModelPluginSpec model;
	expect_false(driver.initialize(model, &error),
		"initialize should fail for horizon == 0");
	expect_true(error.find("horizon") != std::string::npos,
		"error should mention horizon");
}

void test_zero_sample_count() {
	ValidConfigFixture f;
	f.config.sample_count = 0;

	mppi_metal::MPPIDriver driver(f.config);
	std::string error;
	mppi_metal::ModelPluginSpec model;
	expect_false(driver.initialize(model, &error),
		"initialize should fail for sample_count == 0");
	expect_true(error.find("sample_count") != std::string::npos,
		"error should mention sample_count");
}

// ---------------------------------------------------------------------------
// Tests: noise covariance validation
// ---------------------------------------------------------------------------

void test_noise_dim_mismatch() {
	ValidConfigFixture f;
	f.config.noise.control_dim = 99;  // mismatch

	mppi_metal::MPPIDriver driver(f.config);
	std::string error;
	mppi_metal::ModelPluginSpec model;
	expect_false(driver.initialize(model, &error),
		"initialize should fail for noise.control_dim mismatch");
	expect_true(error.find("control_dim") != std::string::npos,
		"error should mention control_dim mismatch");
}

void test_null_sigma() {
	ValidConfigFixture f;
	f.config.noise.sigma = nullptr;

	// Constructor deep-copies only non-null pointers, so sigma stays null.
	mppi_metal::MPPIDriver driver(f.config);
	std::string error;
	mppi_metal::ModelPluginSpec model;
	expect_false(driver.initialize(model, &error),
		"initialize should fail for null sigma");
	expect_true(error.find("sigma") != std::string::npos,
		"error should mention sigma");
}

void test_null_sigma_inv() {
	ValidConfigFixture f;
	f.config.noise.sigma_inv = nullptr;

	mppi_metal::MPPIDriver driver(f.config);
	std::string error;
	mppi_metal::ModelPluginSpec model;
	expect_false(driver.initialize(model, &error),
		"initialize should fail for null sigma_inv");
	expect_true(error.find("sigma_inv") != std::string::npos,
		"error should mention sigma_inv");
}

// ---------------------------------------------------------------------------
// Tests: bounds validation
// ---------------------------------------------------------------------------

void test_bounds_dim_mismatch() {
	ValidConfigFixture f;
	f.config.bounds.control_dim = 99;

	mppi_metal::MPPIDriver driver(f.config);
	std::string error;
	mppi_metal::ModelPluginSpec model;
	expect_false(driver.initialize(model, &error),
		"initialize should fail for bounds.control_dim mismatch");
	expect_true(error.find("control_dim") != std::string::npos,
		"error should mention control_dim mismatch");
}

void test_null_u_min() {
	ValidConfigFixture f;
	f.config.bounds.u_min = nullptr;

	mppi_metal::MPPIDriver driver(f.config);
	std::string error;
	mppi_metal::ModelPluginSpec model;
	expect_false(driver.initialize(model, &error),
		"initialize should fail for null u_min");
	expect_true(error.find("u_min") != std::string::npos,
		"error should mention u_min");
}

void test_null_u_max() {
	ValidConfigFixture f;
	f.config.bounds.u_max = nullptr;

	mppi_metal::MPPIDriver driver(f.config);
	std::string error;
	mppi_metal::ModelPluginSpec model;
	expect_false(driver.initialize(model, &error),
		"initialize should fail for null u_max");
	expect_true(error.find("u_max") != std::string::npos,
		"error should mention u_max");
}

void test_u_min_greater_than_u_max() {
	ValidConfigFixture f;
	// Swap second element so u_min[1] > u_max[1].
	f.u_min[1] = 5.0f;
	f.u_max[1] = -5.0f;
	f.config.bounds.u_min = f.u_min.data();
	f.config.bounds.u_max = f.u_max.data();

	mppi_metal::MPPIDriver driver(f.config);
	std::string error;
	mppi_metal::ModelPluginSpec model;
	expect_false(driver.initialize(model, &error),
		"initialize should fail when u_min > u_max");
	expect_true(error.find("u_min") != std::string::npos,
		"error should mention u_min > u_max");
}

// ---------------------------------------------------------------------------
// Tests: valid config passes validation (reaches MetalBackend stub)
// ---------------------------------------------------------------------------

void test_valid_config_passes_validation() {
	ValidConfigFixture f;

	mppi_metal::MPPIDriver driver(f.config);
	std::string error;
	mppi_metal::ModelPluginSpec model;
	model.user_metallib_path = TEST_USER_METALLIB_PATH;
	bool ok = driver.initialize(model, &error);
	expect_true(ok,
		"initialize should succeed with valid config + valid metallib: " + error);
}

// ---------------------------------------------------------------------------
// Tests: error pointer may be null
// ---------------------------------------------------------------------------

void test_null_error_pointer() {
	ValidConfigFixture f;
	f.config.state_dim = 0;

	mppi_metal::MPPIDriver driver(f.config);
	mppi_metal::ModelPluginSpec model;
	// Should not crash even if error is nullptr.
	bool ok = driver.initialize(model, nullptr);
	expect_false(ok, "initialize should fail with null error pointer without crashing");
}

// ---------------------------------------------------------------------------
// Tests: constructor deep-copies pointer data
// ---------------------------------------------------------------------------

void test_deep_copy_survives_source_destruction() {
	mppi_metal::DriverConfig config;
	config.state_dim = 4;
	config.control_dim = 2;
	config.horizon = 10;
	config.sample_count = 64;
	config.lambda = 1.0f;

	{
		std::vector<float> sigma = {1.0f, 1.0f};
		std::vector<float> sigma_inv = {1.0f, 1.0f};
		std::vector<float> u_min = {-1.0f, -1.0f};
		std::vector<float> u_max = {1.0f, 1.0f};

		config.noise.covariance_form =
			mppi_metal::ControlNoiseConfig::CovarianceForm::kDiagonal;
		config.noise.sigma = sigma.data();
		config.noise.sigma_inv = sigma_inv.data();
		config.noise.control_dim = 2;
		config.bounds.u_min = u_min.data();
		config.bounds.u_max = u_max.data();
		config.bounds.control_dim = 2;
	}
	// Source vectors destroyed. Constructor should have deep-copied.

	mppi_metal::MPPIDriver driver(config);
	std::string error;
	mppi_metal::ModelPluginSpec model;
	model.user_metallib_path = TEST_USER_METALLIB_PATH;
	bool ok = driver.initialize(model, &error);
	expect_true(ok,
		"deep-copied data should pass validation and init: " + error);
}

// ---------------------------------------------------------------------------
// Tests: initialize() Metal backend
// ---------------------------------------------------------------------------

void test_init_nonexistent_metallib() {
	ValidConfigFixture f;
	mppi_metal::MPPIDriver driver(f.config);
	std::string error;
	mppi_metal::ModelPluginSpec model;
	model.user_metallib_path = "/nonexistent/path.metallib";
	bool ok = driver.initialize(model, &error);
	expect_false(ok, "initialize should fail for nonexistent metallib");
}

void test_init_empty_metallib_path() {
	ValidConfigFixture f;
	mppi_metal::MPPIDriver driver(f.config);
	std::string error;
	mppi_metal::ModelPluginSpec model;
	model.user_metallib_path = "";
	bool ok = driver.initialize(model, &error);
	expect_false(ok, "initialize should fail for empty metallib path");
}

void test_init_missing_dynamics_callable() {
	ValidConfigFixture f;
	mppi_metal::MPPIDriver driver(f.config);
	std::string error;
	mppi_metal::ModelPluginSpec model;
	model.user_metallib_path = TEST_USER_METALLIB_PATH;
	model.callables.dynamics = "nonexistent_fn";
	bool ok = driver.initialize(model, &error);
	expect_false(ok, "initialize should fail for missing dynamics callable");
	expect_true(error.find("nonexistent_fn") != std::string::npos,
		"error should mention the missing callable name");
}

// ---------------------------------------------------------------------------
// Tests: reset() (no-arg)
// ---------------------------------------------------------------------------

void test_reset_no_arg_succeeds() {
	ValidConfigFixture f;
	mppi_metal::MPPIDriver driver(f.config);
	std::string error;
	bool ok = driver.reset(&error);
	expect_true(ok, "reset() with no args should succeed");
}

void test_reset_no_arg_null_error_pointer() {
	ValidConfigFixture f;
	mppi_metal::MPPIDriver driver(f.config);
	bool ok = driver.reset(nullptr);
	expect_true(ok, "reset(nullptr) should succeed without crashing");
}

// ---------------------------------------------------------------------------
// Tests: reset(ControlSequenceView)
// ---------------------------------------------------------------------------

void test_reset_with_nominal_succeeds() {
	ValidConfigFixture f;
	mppi_metal::MPPIDriver driver(f.config);

	const size_t n = f.config.horizon * f.config.control_dim;
	std::vector<float> nominal(n, 0.5f);
	mppi_metal::ControlSequenceView view;
	view.data = nominal.data();
	view.horizon = f.config.horizon;
	view.control_dim = f.config.control_dim;

	std::string error;
	bool ok = driver.reset(view, &error);
	expect_true(ok, "reset(valid ControlSequenceView) should succeed");
}

void test_reset_horizon_mismatch() {
	ValidConfigFixture f;
	mppi_metal::MPPIDriver driver(f.config);

	std::vector<float> nominal(100, 0.0f);
	mppi_metal::ControlSequenceView view;
	view.data = nominal.data();
	view.horizon = f.config.horizon + 1;  // wrong
	view.control_dim = f.config.control_dim;

	std::string error;
	bool ok = driver.reset(view, &error);
	expect_false(ok, "reset should fail for horizon mismatch");
	expect_true(error.find("horizon") != std::string::npos,
		"error should mention horizon");
}

void test_reset_control_dim_mismatch() {
	ValidConfigFixture f;
	mppi_metal::MPPIDriver driver(f.config);

	std::vector<float> nominal(100, 0.0f);
	mppi_metal::ControlSequenceView view;
	view.data = nominal.data();
	view.horizon = f.config.horizon;
	view.control_dim = f.config.control_dim + 1;  // wrong

	std::string error;
	bool ok = driver.reset(view, &error);
	expect_false(ok, "reset should fail for control_dim mismatch");
	expect_true(error.find("control_dim") != std::string::npos,
		"error should mention control_dim");
}

void test_reset_null_data_with_nonzero_dims() {
	ValidConfigFixture f;
	mppi_metal::MPPIDriver driver(f.config);

	mppi_metal::ControlSequenceView view;
	view.data = nullptr;  // null!
	view.horizon = f.config.horizon;
	view.control_dim = f.config.control_dim;

	std::string error;
	bool ok = driver.reset(view, &error);
	expect_false(ok, "reset should fail for null data with nonzero dims");
	expect_true(error.find("null") != std::string::npos,
		"error should mention null data");
}

void test_reset_null_error_pointer_with_bad_shape() {
	ValidConfigFixture f;
	mppi_metal::MPPIDriver driver(f.config);

	mppi_metal::ControlSequenceView view;
	view.data = nullptr;
	view.horizon = 999;
	view.control_dim = 999;

	// Should not crash even with nullptr error.
	bool ok = driver.reset(view, nullptr);
	expect_false(ok, "reset with bad shape and null error should fail without crashing");
}

// ---------------------------------------------------------------------------
// Tests: set_problem_params()
// ---------------------------------------------------------------------------

// Helper: create an initialized driver (valid config + metallib).
mppi_metal::MPPIDriver make_initialized_driver(
	const mppi_metal::ParamLayout& layout = {},
	const std::string& stage_cost_name = ""
) {
	ValidConfigFixture f;
	mppi_metal::MPPIDriver driver(f.config);
	mppi_metal::ModelPluginSpec model;
	model.user_metallib_path = TEST_USER_METALLIB_PATH;
	model.param_layout = layout;
	if (!stage_cost_name.empty()) {
		model.callables.stage_cost = stage_cost_name;
	}
	std::string err;
	if (!driver.initialize(model, &err)) {
		std::cerr << "[SETUP] initialize failed: " << err << '\n';
	}
	return driver;
}

void test_set_params_before_init() {
	ValidConfigFixture f;
	mppi_metal::MPPIDriver driver(f.config);  // NOT initialized
	mppi_metal::ProblemParams params;
	std::string error;
	bool ok = driver.set_problem_params(params, &error);
	expect_false(ok, "set_problem_params should fail before initialize");
	expect_true(error.find("not initialized") != std::string::npos,
		"error should mention not initialized");
}

void test_set_params_zero_layout_zero_params() {
	auto driver = make_initialized_driver();  // layout sizes = 0
	mppi_metal::ProblemParams params;  // bytes = 0
	std::string error;
	bool ok = driver.set_problem_params(params, &error);
	expect_true(ok, "set_problem_params should succeed with zero layout + zero params: " + error);
}

void test_set_params_model_size_mismatch() {
	mppi_metal::ParamLayout layout;
	layout.model_params_size = 16;
	auto driver = make_initialized_driver(layout);

	uint8_t buf[8];
	mppi_metal::ProblemParams params;
	params.model_params = {buf, 8};  // wrong size
	std::string error;
	bool ok = driver.set_problem_params(params, &error);
	expect_false(ok, "set_problem_params should fail for model size mismatch");
}

void test_set_params_cost_size_mismatch() {
	mppi_metal::ParamLayout layout;
	layout.cost_params_size = 32;
	auto driver = make_initialized_driver(layout);

	uint8_t buf[16];
	mppi_metal::ProblemParams params;
	params.cost_params = {buf, 16};  // wrong size
	std::string error;
	bool ok = driver.set_problem_params(params, &error);
	expect_false(ok, "set_problem_params should fail for cost size mismatch");
}

void test_set_params_null_data_nonzero_bytes() {
	mppi_metal::ParamLayout layout;
	layout.model_params_size = 16;
	auto driver = make_initialized_driver(layout);

	mppi_metal::ProblemParams params;
	params.model_params = {nullptr, 16};
	std::string error;
	bool ok = driver.set_problem_params(params, &error);
	expect_false(ok, "set_problem_params should fail for null data with nonzero bytes");
}

void test_set_params_matching_layout() {
	mppi_metal::ParamLayout layout;
	layout.model_params_size = 16;
	layout.cost_params_size = 8;
	auto driver = make_initialized_driver(layout);

	std::vector<uint8_t> model_buf(16, 0);
	std::vector<uint8_t> cost_buf(8, 0);
	mppi_metal::ProblemParams params;
	params.model_params = {model_buf.data(), 16};
	params.cost_params = {cost_buf.data(), 8};
	std::string error;
	bool ok = driver.set_problem_params(params, &error);
	expect_true(ok, "set_problem_params should succeed with matching sizes: " + error);
}

void test_set_params_layout_requires_but_zero_provided() {
	mppi_metal::ParamLayout layout;
	layout.model_params_size = 16;
	auto driver = make_initialized_driver(layout);

	mppi_metal::ProblemParams params;  // bytes = 0
	std::string error;
	bool ok = driver.set_problem_params(params, &error);
	expect_false(ok, "set_problem_params should fail when layout requires params but none provided");
}

// ---------------------------------------------------------------------------
// Tests: step()
// ---------------------------------------------------------------------------

void test_step_before_init() {
	ValidConfigFixture f;
	mppi_metal::MPPIDriver driver(f.config);

	std::vector<float> x0(f.config.state_dim, 0.0f);
	std::vector<float> u_out(f.config.horizon * f.config.control_dim, 0.0f);
	mppi_metal::StepInputs inputs;
	inputs.x0 = {x0.data(), f.config.state_dim};
	mppi_metal::MutableControlSequenceView out = {u_out.data(), f.config.horizon, f.config.control_dim};
	mppi_metal::StepDiagnostics diag;
	std::string error;
	bool ok = driver.step(inputs, out, &diag, &error);
	expect_false(ok, "step should fail before initialize");
}

void test_step_null_x0() {
	auto driver = make_initialized_driver();
	std::vector<float> u_out(10 * 2, 0.0f);
	mppi_metal::StepInputs inputs;
	inputs.x0 = {nullptr, 4};
	mppi_metal::MutableControlSequenceView out = {u_out.data(), 10, 2};
	std::string error;
	bool ok = driver.step(inputs, out, nullptr, &error);
	expect_false(ok, "step should fail for null x0");
}

void test_step_x0_dim_mismatch() {
	auto driver = make_initialized_driver();
	std::vector<float> x0(4, 0.0f);
	std::vector<float> u_out(10 * 2, 0.0f);
	mppi_metal::StepInputs inputs;
	inputs.x0 = {x0.data(), 99};  // wrong dim
	mppi_metal::MutableControlSequenceView out = {u_out.data(), 10, 2};
	std::string error;
	bool ok = driver.step(inputs, out, nullptr, &error);
	expect_false(ok, "step should fail for x0 state_dim mismatch");
}

void test_step_null_output() {
	auto driver = make_initialized_driver();
	std::vector<float> x0(4, 0.0f);
	mppi_metal::StepInputs inputs;
	inputs.x0 = {x0.data(), 4};
	mppi_metal::MutableControlSequenceView out = {nullptr, 10, 2};
	std::string error;
	bool ok = driver.step(inputs, out, nullptr, &error);
	expect_false(ok, "step should fail for null output data");
}

void test_step_succeeds() {
	auto driver = make_initialized_driver();
	std::vector<float> x0(4, 1.0f);
	const uint32_t H = 10, C = 2;
	std::vector<float> u_out(H * C, -999.0f);
	mppi_metal::StepInputs inputs;
	inputs.x0 = {x0.data(), 4};
	mppi_metal::MutableControlSequenceView out = {u_out.data(), H, C};
	mppi_metal::StepDiagnostics diag;
	std::string error;
	bool ok = driver.step(inputs, out, &diag, &error);
	expect_true(ok, "step should succeed with valid inputs: " + error);
}

void test_step_override_shape_mismatch() {
	auto driver = make_initialized_driver();
	std::vector<float> x0(4, 0.0f);
	std::vector<float> u_out(10 * 2, 0.0f);
	std::vector<float> override_nom(100, 0.0f);
	mppi_metal::StepInputs inputs;
	inputs.x0 = {x0.data(), 4};
	inputs.u_nominal_override = {override_nom.data(), 99, 99};  // wrong shape
	mppi_metal::MutableControlSequenceView out = {u_out.data(), 10, 2};
	std::string error;
	bool ok = driver.step(inputs, out, nullptr, &error);
	expect_false(ok, "step should fail for u_nominal_override shape mismatch");
}

void test_step_mppi_moves_toward_origin() {
	// With stage_cost = ||position||^2 and dynamics = simple integrator,
	// starting at (1,1,0,0), MPPI should produce negative accelerations
	// to move toward the origin.
	auto driver = make_initialized_driver({}, "mppi_stage_cost");

	// x0 = [px=1, py=1, vx=0, vy=0]
	std::vector<float> x0 = {1.0f, 1.0f, 0.0f, 0.0f};
	const uint32_t H = 10, C = 2;
	std::vector<float> u_out(H * C, 0.0f);

	mppi_metal::StepInputs inputs;
	inputs.x0 = {x0.data(), 4};
	mppi_metal::MutableControlSequenceView out = {u_out.data(), H, C};
	mppi_metal::StepDiagnostics diag;
	std::string error;

	bool ok = driver.step(inputs, out, &diag, &error);
	expect_true(ok, "step with stage cost should succeed: " + error);

	// First control should be negative (accelerate toward origin).
	expect_true(u_out[0] < 0.0f,
		"ax should be negative to move toward origin (got " +
		std::to_string(u_out[0]) + ")");
	expect_true(u_out[1] < 0.0f,
		"ay should be negative to move toward origin (got " +
		std::to_string(u_out[1]) + ")");

	// Diagnostics should have meaningful values.
	expect_true(diag.best_cost > 0.0f,
		"best_cost should be > 0 (got " + std::to_string(diag.best_cost) + ")");
	expect_true(diag.mean_cost >= diag.best_cost,
		"mean_cost should be >= best_cost");
	expect_true(diag.effective_sample_size > 0.0f,
		"ESS should be > 0");
}

void test_step_rng_determinism() {
	// Two drivers with same seed should produce identical results.
	auto driver1 = make_initialized_driver({}, "mppi_stage_cost");
	auto driver2 = make_initialized_driver({}, "mppi_stage_cost");

	std::vector<float> x0 = {1.0f, 1.0f, 0.0f, 0.0f};
	const uint32_t H = 10, C = 2;
	std::vector<float> u1(H * C, 0.0f), u2(H * C, 0.0f);

	mppi_metal::StepInputs inputs;
	inputs.x0 = {x0.data(), 4};
	mppi_metal::MutableControlSequenceView out1 = {u1.data(), H, C};
	mppi_metal::MutableControlSequenceView out2 = {u2.data(), H, C};

	driver1.step(inputs, out1, nullptr, nullptr);
	driver2.step(inputs, out2, nullptr, nullptr);

	bool identical = true;
	for (size_t i = 0; i < H * C; ++i) {
		if (u1[i] != u2[i]) { identical = false; break; }
	}
	expect_true(identical,
		"Same seed should produce identical outputs (Philox determinism)");
}

void test_simulate_multi_step_parity() {
	auto driver1 = make_initialized_driver({}, "mppi_stage_cost");
	auto driver2 = make_initialized_driver({}, "mppi_stage_cost");

	const uint32_t N = 5;
	const uint32_t H = 10, C = 2, Sdim = 4;
	
	std::vector<float> x0_loop = {1.0f, 1.0f, 0.0f, 0.0f};
	std::vector<float> x0_sim = {1.0f, 1.0f, 0.0f, 0.0f};
	
	std::vector<float> history_states_loop(N * Sdim);
	std::vector<float> history_controls_loop(N * C);
	std::vector<float> history_costs_loop(N);

	std::vector<float> u_out(H * C, 0.0f);
	mppi_metal::MutableControlSequenceView out_view = {u_out.data(), H, C};

	for (uint32_t i = 0; i < N; ++i) {
		mppi_metal::StepInputs inputs;
		inputs.x0 = {x0_loop.data(), Sdim};
		mppi_metal::StepDiagnostics diag;
		
		driver1.step(inputs, out_view, &diag, nullptr);

		for (uint32_t d=0; d<Sdim; ++d) history_states_loop[i * Sdim + d] = x0_loop[d];
		for (uint32_t d=0; d<C; ++d) history_controls_loop[i * C + d] = u_out[d];
		history_costs_loop[i] = diag.best_cost;

		float dt = 0.1f;
		x0_loop[0] += x0_loop[2] * dt;
		x0_loop[1] += x0_loop[3] * dt;
		x0_loop[2] += u_out[0] * dt;
		x0_loop[3] += u_out[1] * dt;
	}

	std::vector<float> history_states_sim(N * Sdim);
	std::vector<float> history_controls_sim(N * C);
	std::vector<float> history_costs_sim(N);

	mppi_metal::SimulationResults results;
	results.states_out = history_states_sim.data();
	results.controls_out = history_controls_sim.data();
	results.costs_out = history_costs_sim.data();
	results.num_steps = N;

	mppi_metal::StepInputs sim_inputs;
	sim_inputs.x0 = {x0_sim.data(), Sdim};
	std::string error;
	bool ok = driver2.simulate(N, sim_inputs, results, &error);
	expect_true(ok, "simulate() should succeed: " + error);

	bool match = true;
	for (uint32_t i = 0; i < N; ++i) {
		for (uint32_t d=0; d<Sdim; ++d) {
			if (std::abs(history_states_loop[i*Sdim + d] - history_states_sim[i*Sdim + d]) > 1e-4f) match = false;
		}
		for (uint32_t d=0; d<C; ++d) {
			if (std::abs(history_controls_loop[i*C + d] - history_controls_sim[i*C + d]) > 1e-4f) match = false;
		}
		if (std::abs(history_costs_loop[i] - history_costs_sim[i]) > 1e-4f) match = false;
	}
	
	expect_true(match, "Multi-step simulate() should exactly match N iterations of step()");
}

void test_simulate_batch_parity() {
	// Verify that simulate_batch with N=1 produces identical results
	// to the single-agent simulate() path.
	auto driver1 = make_initialized_driver({}, "mppi_stage_cost");
	auto driver2 = make_initialized_driver({}, "mppi_stage_cost");

	const uint32_t T = 5;
	const uint32_t H = 10, C = 2, Sdim = 4;

	std::vector<float> x0 = {1.0f, 1.0f, 0.0f, 0.0f};

	// Single-agent simulate()
	std::vector<float> states1(T * Sdim);
	std::vector<float> controls1(T * C);
	std::vector<float> costs1(T);
	mppi_metal::SimulationResults results1;
	results1.states_out = states1.data();
	results1.controls_out = controls1.data();
	results1.costs_out = costs1.data();
	results1.num_steps = T;

	mppi_metal::StepInputs inputs1;
	inputs1.x0 = {x0.data(), Sdim};
	std::string err;
	bool ok1 = driver1.simulate(T, inputs1, results1, &err);
	expect_true(ok1, "simulate() should succeed: " + err);

	// Batch simulate with N=1
	std::vector<float> states2(1 * T * Sdim);
	std::vector<float> controls2(1 * T * C);
	std::vector<float> costs2(1 * T);
	mppi_metal::BatchSimulationResults results2;
	results2.states_out = states2.data();
	results2.controls_out = controls2.data();
	results2.costs_out = costs2.data();
	results2.num_agents = 1;
	results2.num_steps = T;

	ok1 = driver2.simulate_batch(1, T, x0.data(), results2, &err);
	expect_true(ok1, "simulate_batch(N=1) should succeed: " + err);

	bool match = true;
	for (uint32_t i = 0; i < T; ++i) {
		for (uint32_t d = 0; d < Sdim; ++d) {
			if (std::abs(states1[i*Sdim + d] - states2[i*Sdim + d]) > 1e-4f) match = false;
		}
		for (uint32_t d = 0; d < C; ++d) {
			if (std::abs(controls1[i*C + d] - controls2[i*C + d]) > 1e-4f) match = false;
		}
		if (std::abs(costs1[i] - costs2[i]) > 1e-4f) match = false;
	}
	expect_true(match, "simulate_batch(N=1) should match simulate()");
}

void test_simulate_batch_multi_agent() {
	// Verify N=4 agents starting at different positions produce
	// distinct trajectories with valid results.
	auto driver = make_initialized_driver({}, "mppi_stage_cost");

	const uint32_t N = 4;
	const uint32_t T = 5;
	const uint32_t H = 10, C = 2, Sdim = 4;

	// 4 agents at different positions.
	std::vector<float> x0_packed = {
		1.0f,  1.0f, 0.0f, 0.0f,   // agent 0
		-1.0f, 1.0f, 0.0f, 0.0f,   // agent 1
		1.0f, -1.0f, 0.0f, 0.0f,   // agent 2
		-1.0f,-1.0f, 0.0f, 0.0f    // agent 3
	};

	std::vector<float> states(N * T * Sdim);
	std::vector<float> controls(N * T * C);
	std::vector<float> costs(N * T);
	mppi_metal::BatchSimulationResults results;
	results.states_out = states.data();
	results.controls_out = controls.data();
	results.costs_out = costs.data();
	results.num_agents = N;
	results.num_steps = T;

	std::string err;
	bool ok = driver.simulate_batch(N, T, x0_packed.data(), results, &err);
	expect_true(ok, "simulate_batch(N=4) should succeed: " + err);

	// Verify agents have distinct initial states in history.
	bool distinct = true;
	for (uint32_t a = 1; a < N; ++a) {
		bool same = true;
		for (uint32_t d = 0; d < Sdim; ++d) {
			if (states[a * T * Sdim + d] != states[0 * T * Sdim + d]) {
				same = false;
				break;
			}
		}
		if (same) { distinct = false; break; }
	}
	expect_true(distinct, "Agents should have distinct initial states");

	// Verify all costs are positive.
	bool valid_costs = true;
	for (uint32_t i = 0; i < N * T; ++i) {
		if (costs[i] <= 0.0f) { valid_costs = false; break; }
	}
	expect_true(valid_costs, "All batch costs should be > 0");
}

}  // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
	test_zero_state_dim();
	test_zero_control_dim();
	test_zero_horizon();
	test_zero_sample_count();

	test_noise_dim_mismatch();
	test_null_sigma();
	test_null_sigma_inv();

	test_bounds_dim_mismatch();
	test_null_u_min();
	test_null_u_max();
	test_u_min_greater_than_u_max();

	test_valid_config_passes_validation();
	test_null_error_pointer();
	test_deep_copy_survives_source_destruction();

	test_init_nonexistent_metallib();
	test_init_empty_metallib_path();
	test_init_missing_dynamics_callable();

	test_reset_no_arg_succeeds();
	test_reset_no_arg_null_error_pointer();
	test_reset_with_nominal_succeeds();
	test_reset_horizon_mismatch();
	test_reset_control_dim_mismatch();
	test_reset_null_data_with_nonzero_dims();
	test_reset_null_error_pointer_with_bad_shape();

	test_set_params_before_init();
	test_set_params_zero_layout_zero_params();
	test_set_params_model_size_mismatch();
	test_set_params_cost_size_mismatch();
	test_set_params_null_data_nonzero_bytes();
	test_set_params_matching_layout();
	test_set_params_layout_requires_but_zero_provided();

	test_step_before_init();
	test_step_null_x0();
	test_step_x0_dim_mismatch();
	test_step_null_output();
	test_step_succeeds();
	test_step_override_shape_mismatch();
	test_step_mppi_moves_toward_origin();
	test_step_rng_determinism();
	test_simulate_multi_step_parity();
	test_simulate_batch_parity();
	test_simulate_batch_multi_agent();

	std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.\n";
	return g_fail > 0 ? 1 : 0;
}
