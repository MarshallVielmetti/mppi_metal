#include "mppi_metal/mppi_driver.hpp"

#include "metal_backend.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace mppi_metal {

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

class MPPIDriver::Impl {
public:
	explicit Impl(const DriverConfig& config);

	DriverConfig config;

	// Deep-copied covariance / bounds arrays.
	std::vector<float> sigma_storage;
	std::vector<float> sigma_inv_storage;
	std::vector<float> u_min_storage;
	std::vector<float> u_max_storage;

	// Internal nominal control warm-start (horizon * control_dim).
	std::vector<float> u_nominal;

	// Bound problem params (set via set_problem_params).
	ProblemParams bound_params{};
	ParamLayout param_layout{};

	MetalBackend backend;
	bool initialized = false;
	uint64_t step_index = 0;
};

MPPIDriver::Impl::Impl(const DriverConfig& cfg) : config(cfg) {
	const uint32_t cdim = cfg.control_dim;

	// Deep-copy sigma.
	if (cfg.noise.sigma != nullptr && cdim > 0) {
		const uint32_t n = (cfg.noise.covariance_form == ControlNoiseConfig::CovarianceForm::kDiagonal)
			? cdim
			: cdim * cdim;
		sigma_storage.assign(cfg.noise.sigma, cfg.noise.sigma + n);
		config.noise.sigma = sigma_storage.data();
	}

	// Deep-copy sigma_inv.
	if (cfg.noise.sigma_inv != nullptr && cdim > 0) {
		const uint32_t n = (cfg.noise.covariance_form == ControlNoiseConfig::CovarianceForm::kDiagonal)
			? cdim
			: cdim * cdim;
		sigma_inv_storage.assign(cfg.noise.sigma_inv, cfg.noise.sigma_inv + n);
		config.noise.sigma_inv = sigma_inv_storage.data();
	}

	// Deep-copy bounds.
	if (cfg.bounds.u_min != nullptr && cdim > 0) {
		u_min_storage.assign(cfg.bounds.u_min, cfg.bounds.u_min + cdim);
		config.bounds.u_min = u_min_storage.data();
	}
	if (cfg.bounds.u_max != nullptr && cdim > 0) {
		u_max_storage.assign(cfg.bounds.u_max, cfg.bounds.u_max + cdim);
		config.bounds.u_max = u_max_storage.data();
	}

	// Initialize nominal control to zeros.
	u_nominal.resize(
		static_cast<size_t>(cfg.horizon) * static_cast<size_t>(cfg.control_dim),
		0.0f
	);
}

// ---------------------------------------------------------------------------
// MPPIDriver public interface
// ---------------------------------------------------------------------------

MPPIDriver::MPPIDriver(const DriverConfig& config)
	: impl_(std::make_unique<Impl>(config)) {}

MPPIDriver::~MPPIDriver() = default;

MPPIDriver::MPPIDriver(MPPIDriver&&) noexcept = default;

MPPIDriver& MPPIDriver::operator=(MPPIDriver&&) noexcept = default;

bool MPPIDriver::initialize(const ModelPluginSpec& model, std::string* error) {
	auto set_err = [&](const std::string& msg) {
		if (error != nullptr) {
			*error = msg;
		}
	};

	const auto& cfg = impl_->config;

	// ---- Validate DriverConfig dimensions ----
	if (cfg.state_dim == 0) {
		set_err("DriverConfig::state_dim must be > 0.");
		return false;
	}
	if (cfg.control_dim == 0) {
		set_err("DriverConfig::control_dim must be > 0.");
		return false;
	}
	if (cfg.horizon == 0) {
		set_err("DriverConfig::horizon must be > 0.");
		return false;
	}
	if (cfg.sample_count == 0) {
		set_err("DriverConfig::sample_count must be > 0.");
		return false;
	}

	// ---- Validate noise covariance ----
	if (cfg.noise.control_dim != cfg.control_dim) {
		set_err("ControlNoiseConfig::control_dim (" +
			std::to_string(cfg.noise.control_dim) +
			") must match DriverConfig::control_dim (" +
			std::to_string(cfg.control_dim) + ").");
		return false;
	}
	if (cfg.noise.sigma == nullptr) {
		set_err("ControlNoiseConfig::sigma must not be null.");
		return false;
	}
	if (cfg.noise.sigma_inv == nullptr) {
		set_err("ControlNoiseConfig::sigma_inv must not be null.");
		return false;
	}

	// ---- Validate control bounds ----
	if (cfg.bounds.control_dim != cfg.control_dim) {
		set_err("ControlBounds::control_dim (" +
			std::to_string(cfg.bounds.control_dim) +
			") must match DriverConfig::control_dim (" +
			std::to_string(cfg.control_dim) + ").");
		return false;
	}
	if (cfg.bounds.u_min == nullptr) {
		set_err("ControlBounds::u_min must not be null when control_dim > 0.");
		return false;
	}
	if (cfg.bounds.u_max == nullptr) {
		set_err("ControlBounds::u_max must not be null when control_dim > 0.");
		return false;
	}
	for (uint32_t i = 0; i < cfg.bounds.control_dim; ++i) {
		if (cfg.bounds.u_min[i] > cfg.bounds.u_max[i]) {
			set_err("ControlBounds::u_min[" + std::to_string(i) +
				"] (" + std::to_string(cfg.bounds.u_min[i]) +
				") > u_max[" + std::to_string(i) +
				"] (" + std::to_string(cfg.bounds.u_max[i]) + ").");
			return false;
		}
	}

	// ---- Delegate to MetalBackend ----
	if (!impl_->backend.initialize(model, cfg, error)) {
		return false;
	}

	// Store param layout for set_problem_params validation.
	impl_->param_layout = model.param_layout;
	impl_->initialized = true;
	return true;
}

bool MPPIDriver::set_problem_params(const ProblemParams& params, std::string* error) {
	auto set_err = [&](const std::string& msg) {
		if (error != nullptr) {
			*error = msg;
		}
	};

	if (!impl_->initialized) {
		set_err("Driver is not initialized. Call initialize() first.");
		return false;
	}

	const auto& layout = impl_->param_layout;

	// Validate model params size.
	if (layout.model_params_size > 0 && params.model_params.bytes == 0) {
		set_err("ParamLayout declares model_params_size " +
			std::to_string(layout.model_params_size) +
			" but provided model_params.bytes is 0.");
		return false;
	}
	if (params.model_params.bytes > 0 &&
		params.model_params.bytes != layout.model_params_size) {
		set_err("model_params.bytes (" +
			std::to_string(params.model_params.bytes) +
			") does not match ParamLayout::model_params_size (" +
			std::to_string(layout.model_params_size) + ").");
		return false;
	}
	if (params.model_params.bytes > 0 && params.model_params.data == nullptr) {
		set_err("model_params.data is null but bytes > 0.");
		return false;
	}

	// Validate cost params size.
	if (layout.cost_params_size > 0 && params.cost_params.bytes == 0) {
		set_err("ParamLayout declares cost_params_size " +
			std::to_string(layout.cost_params_size) +
			" but provided cost_params.bytes is 0.");
		return false;
	}
	if (params.cost_params.bytes > 0 &&
		params.cost_params.bytes != layout.cost_params_size) {
		set_err("cost_params.bytes (" +
			std::to_string(params.cost_params.bytes) +
			") does not match ParamLayout::cost_params_size (" +
			std::to_string(layout.cost_params_size) + ").");
		return false;
	}
	if (params.cost_params.bytes > 0 && params.cost_params.data == nullptr) {
		set_err("cost_params.data is null but bytes > 0.");
		return false;
	}

	// Delegate to backend.
	if (!impl_->backend.bind_params(params.model_params, params.cost_params, error)) {
		return false;
	}

	impl_->bound_params = params;
	return true;
}

bool MPPIDriver::step(
	const StepInputs& inputs,
	const MutableControlSequenceView& u_next_out,
	StepDiagnostics* diagnostics,
	std::string* error
) {
	auto set_err = [&](const std::string& msg) {
		if (error != nullptr) {
			*error = msg;
		}
	};

	// ---- Pre-conditions ----
	if (!impl_->initialized) {
		set_err("Driver is not initialized. Call initialize() first.");
		return false;
	}

	const auto& cfg = impl_->config;

	// Validate x0.
	if (inputs.x0.data == nullptr) {
		set_err("StepInputs::x0.data must not be null.");
		return false;
	}
	if (inputs.x0.state_dim != cfg.state_dim) {
		set_err("StepInputs::x0.state_dim (" +
			std::to_string(inputs.x0.state_dim) +
			") must match DriverConfig::state_dim (" +
			std::to_string(cfg.state_dim) + ").");
		return false;
	}

	// Validate output buffer.
	if (u_next_out.data == nullptr) {
		set_err("u_next_out.data must not be null.");
		return false;
	}
	if (u_next_out.horizon != cfg.horizon ||
		u_next_out.control_dim != cfg.control_dim) {
		set_err("u_next_out shape mismatch.");
		return false;
	}

	// ---- Resolve active nominal controls ----
	const float* active_nominal = impl_->u_nominal.data();
	if (inputs.u_nominal_override.data != nullptr) {
		// Validate override shape.
		if (inputs.u_nominal_override.horizon != cfg.horizon ||
			inputs.u_nominal_override.control_dim != cfg.control_dim) {
			set_err("u_nominal_override shape mismatch.");
			return false;
		}
		active_nominal = inputs.u_nominal_override.data;
	}

	// ---- Resolve active params (overrides vs bound) ----
	ByteView active_model_params = impl_->bound_params.model_params;
	if (inputs.model_params_override.bytes > 0) {
		active_model_params = inputs.model_params_override;
	}

	ByteView active_cost_params = impl_->bound_params.cost_params;
	if (inputs.cost_params_override.bytes > 0) {
		active_cost_params = inputs.cost_params_override;
	}

	// ---- Dispatch GPU rollout ----
	const size_t seq_len =
		static_cast<size_t>(cfg.horizon) * static_cast<size_t>(cfg.control_dim);

	// Temporary buffer for the GPU-computed updated controls.
	std::vector<float> u_updated(seq_len, 0.0f);

	if (!impl_->backend.dispatch_rollout(
			cfg,
			impl_->step_index,
			inputs.x0.data,
			active_nominal,
			cfg.noise,
			active_model_params,
			active_cost_params,
			u_updated.data(),
			diagnostics,
			error)) {
		return false;
	}

	// ---- Write output ----
	std::copy(u_updated.begin(), u_updated.end(), u_next_out.data);

	// ---- Warm-start: shift nominal left by one step, hold-last ----
	const uint32_t cdim = cfg.control_dim;
	for (uint32_t t = 0; t < cfg.horizon - 1; ++t) {
		for (uint32_t d = 0; d < cdim; ++d) {
			impl_->u_nominal[t * cdim + d] = u_updated[(t + 1) * cdim + d];
		}
	}
	// Fill last step with previous terminal control (hold-last).
	for (uint32_t d = 0; d < cdim; ++d) {
		impl_->u_nominal[(cfg.horizon - 1) * cdim + d] =
			u_updated[(cfg.horizon - 1) * cdim + d];
	}

	++impl_->step_index;
	return true;
}

bool MPPIDriver::simulate(
	uint32_t num_steps,
	const StepInputs& inputs,
	const SimulationResults& results,
	std::string* error
) {
	auto set_err = [&](const std::string& msg) {
		if (error != nullptr) {
			*error = msg;
		}
	};

	if (!impl_->initialized) {
		set_err("Driver is not initialized. Call initialize() first.");
		return false;
	}

	if (num_steps == 0) {
		set_err("num_steps must be > 0.");
		return false;
	}

	const auto& cfg = impl_->config;

	// Validate x0.
	if (inputs.x0.data == nullptr) {
		set_err("StepInputs::x0.data must not be null.");
		return false;
	}
	if (inputs.x0.state_dim != cfg.state_dim) {
		set_err("StepInputs::x0.state_dim (" +
			std::to_string(inputs.x0.state_dim) +
			") must match DriverConfig::state_dim (" +
			std::to_string(cfg.state_dim) + ").");
		return false;
	}

	if (results.num_steps != num_steps) {
		set_err("SimulationResults::num_steps mismatch.");
		return false;
	}

	const float* active_nominal = impl_->u_nominal.data();
	if (inputs.u_nominal_override.data != nullptr) {
		if (inputs.u_nominal_override.horizon != cfg.horizon ||
			inputs.u_nominal_override.control_dim != cfg.control_dim) {
			set_err("u_nominal_override shape mismatch.");
			return false;
		}
		active_nominal = inputs.u_nominal_override.data;
	}

	ByteView active_model_params = impl_->bound_params.model_params;
	if (inputs.model_params_override.bytes > 0) {
		active_model_params = inputs.model_params_override;
	}

	ByteView active_cost_params = impl_->bound_params.cost_params;
	if (inputs.cost_params_override.bytes > 0) {
		active_cost_params = inputs.cost_params_override;
	}

	if (!impl_->backend.dispatch_simulation(
			num_steps,
			cfg,
			impl_->step_index,
			inputs.x0.data,
			active_nominal,
			cfg.noise,
			active_model_params,
			active_cost_params,
			results,
			error)) {
		return false;
	}

	impl_->step_index += num_steps;
	return true;
}

bool MPPIDriver::reset(std::string* error) {
	// Zero out internal nominal controls and reset step index.
	// Bound ProblemParams are kept unchanged.
	// RNG state deterministically continues from current step index.
	std::fill(impl_->u_nominal.begin(), impl_->u_nominal.end(), 0.0f);
	impl_->step_index = 0;
	return true;
}

bool MPPIDriver::reset(const ControlSequenceView& u_nominal, std::string* error) {
	auto set_err = [&](const std::string& msg) {
		if (error != nullptr) {
			*error = msg;
		}
	};

	const auto& cfg = impl_->config;

	if (u_nominal.horizon != cfg.horizon) {
		set_err("reset: u_nominal.horizon (" +
			std::to_string(u_nominal.horizon) +
			") must match DriverConfig::horizon (" +
			std::to_string(cfg.horizon) + ").");
		return false;
	}
	if (u_nominal.control_dim != cfg.control_dim) {
		set_err("reset: u_nominal.control_dim (" +
			std::to_string(u_nominal.control_dim) +
			") must match DriverConfig::control_dim (" +
			std::to_string(cfg.control_dim) + ").");
		return false;
	}
	const size_t n = static_cast<size_t>(cfg.horizon) * static_cast<size_t>(cfg.control_dim);
	if (n > 0 && u_nominal.data == nullptr) {
		set_err("reset: u_nominal.data must not be null when horizon * control_dim > 0.");
		return false;
	}

	// Copy provided nominal into internal storage and reset step index.
	if (n > 0) {
		std::copy(u_nominal.data, u_nominal.data + n, impl_->u_nominal.begin());
	}
	impl_->step_index = 0;
	return true;
}

}  // namespace mppi_metal
