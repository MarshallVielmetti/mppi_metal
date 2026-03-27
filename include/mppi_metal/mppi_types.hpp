#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace mppi_metal {

// ---------------------------------------------------------------------------
// Control noise configuration
// ---------------------------------------------------------------------------

struct ControlNoiseConfig {
	enum class CovarianceForm {
		kDiagonal,
		kFull
	};

	// Required by MPPI update law.
	CovarianceForm covariance_form = CovarianceForm::kDiagonal;

	// For kDiagonal: control_dim elements.
	// For kFull: control_dim * control_dim row-major matrix.
	const float* sigma = nullptr;
	const float* sigma_inv = nullptr;
	uint32_t control_dim = 0;
};

// ---------------------------------------------------------------------------
// Control bounds
// ---------------------------------------------------------------------------

struct ControlBounds {
	// Per-control-dimension actuator limits.
	const float* u_min = nullptr;   // control_dim elements
	const float* u_max = nullptr;   // control_dim elements
	uint32_t control_dim = 0;
};

// ---------------------------------------------------------------------------
// Driver configuration
// ---------------------------------------------------------------------------

struct DriverConfig {
	uint32_t state_dim = 0;
	uint32_t control_dim = 0;
	uint32_t horizon = 0;
	uint32_t sample_count = 0;
	float lambda = 1.0f;
	ControlNoiseConfig noise;
	ControlBounds bounds;

	enum class NumericMode {
		kFp32,
		kFp16,
		kMixed
	};

	NumericMode numeric_mode = NumericMode::kFp32;
};

// ---------------------------------------------------------------------------
// RNG configuration
// ---------------------------------------------------------------------------

struct RngConfig {
	// v1 locks Philox for counter-based deterministic RNG.
	// Philox state is derived from step index and trajectory ID, ensuring
	// determinism regardless of threadgroup dispatch order.
	uint64_t seed = 0;
};

// ---------------------------------------------------------------------------
// Model-plugin specification
// ---------------------------------------------------------------------------

struct CallableNames {
	// Required device callable.  Must be defined with [[visible]].
	std::string_view dynamics = "mppi_dynamics";

	// Optional device callables.  If omitted, library uses default / no-op.
	std::string_view stage_cost = "";
	std::string_view terminal_cost = "";
};

struct ParamLayout {
	size_t model_params_size = 0;
	size_t model_params_alignment = 1;
	size_t cost_params_size = 0;
	size_t cost_params_alignment = 1;
	std::string schema_hash;
};

struct ModelPluginSpec {
	std::filesystem::path user_metallib_path;
	CallableNames callables;
	ParamLayout param_layout;
	RngConfig rng;
};

// ---------------------------------------------------------------------------
// Typed view wrappers
// ---------------------------------------------------------------------------

struct StateView {
	const float* data = nullptr;
	uint32_t state_dim = 0;
};

struct ControlSequenceView {
	const float* data = nullptr;
	uint32_t horizon = 0;
	uint32_t control_dim = 0;
};

struct MutableControlSequenceView {
	float* data = nullptr;
	uint32_t horizon = 0;
	uint32_t control_dim = 0;
};

struct ByteView {
	const void* data = nullptr;
	size_t bytes = 0;
};

// ---------------------------------------------------------------------------
// Parameters and per-step inputs
// ---------------------------------------------------------------------------

struct ProblemParams {
	// Bound once (or infrequently) and reused across step() calls.
	ByteView model_params;
	ByteView cost_params;
};

struct StepInputs {
	StateView x0;

	// Optional per-step nominal control override.
	// If data is null, the driver uses its internal warm-start sequence.
	ControlSequenceView u_nominal_override;

	// Optional per-step overrides.
	// If bytes == 0, the driver uses bound ProblemParams.
	ByteView model_params_override;
	ByteView cost_params_override;
};

// ---------------------------------------------------------------------------
// Step diagnostics
// ---------------------------------------------------------------------------

struct StepDiagnostics {
	float best_cost = 0.0f;
	float mean_cost = 0.0f;
	float effective_sample_size = 0.0f;
	bool converged = false;
};

// ---------------------------------------------------------------------------
// Multi-step simulation outputs
// ---------------------------------------------------------------------------

struct SimulationResults {
	float* states_out = nullptr;     // (num_steps) * state_dim
	float* controls_out = nullptr;   // (num_steps) * control_dim (the optimal control chosen at each step)
	float* costs_out = nullptr;      // (num_steps) elements (best_cost at each step)
	uint32_t num_steps = 0;
};

// ---------------------------------------------------------------------------
// Batched multi-agent simulation outputs
// ---------------------------------------------------------------------------

/// Results buffer for simulate_batch().
/// All output arrays are contiguous and indexed as:
///   [step * num_agents * dim + agent_idx * dim + element]
struct BatchSimulationResults {
	float* states_out = nullptr;     // [num_agents * num_steps * state_dim]
	float* controls_out = nullptr;   // [num_agents * num_steps * control_dim]
	float* costs_out = nullptr;      // [num_agents * num_steps]
	uint32_t num_agents = 0;
	uint32_t num_steps = 0;
};

}  // namespace mppi_metal
