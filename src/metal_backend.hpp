#pragma once

#include "mppi_metal/mppi_types.hpp"

#include <cstddef>
#include <memory>
#include <string>

namespace mppi_metal {

/// Internal Metal backend responsible for GPU pipeline lifecycle.
///
/// Responsibilities (from contract):
///   1. Load the library metallib containing the master mppi_rollout kernel.
///   2. Load the user metallib containing [[visible]] callable functions.
///   3. Build MTLLinkedFunctions from user callable names.
///   4. Create MTLComputePipelineState for mppi_rollout with linked functions.
///   5. Create and populate MTLVisibleFunctionTable with callable handles.
///
/// This class is only used from mppi_driver.cpp and is not part of the public
/// API.
class MetalBackend {
public:
	MetalBackend();
	~MetalBackend();

	MetalBackend(const MetalBackend&) = delete;
	MetalBackend& operator=(const MetalBackend&) = delete;

	MetalBackend(MetalBackend&&) noexcept;
	MetalBackend& operator=(MetalBackend&&) noexcept;

	/// Build the linked pipeline from the library and user metallibs.
	/// Performs all hard-fail validation checks.
	bool initialize(
		const ModelPluginSpec& model,
		const DriverConfig& config,
		std::string* error = nullptr
	);

	/// Upload / bind parameter buffers for subsequent dispatches.
	bool bind_params(
		const ByteView& model_params,
		const ByteView& cost_params,
		std::string* error = nullptr
	);

	/// Dispatch the master mppi_rollout kernel with linked user callables.
	///
	/// @param x0              Current state vector (state_dim floats).
	/// @param u_nominal       Nominal control sequence (horizon * control_dim).
	/// @param noise_sigma     Noise covariance data.
	/// @param model_params    Active model param blob (may be override).
	/// @param cost_params     Active cost param blob (may be override).
	/// @param u_out           Output buffer for updated control sequence.
	/// @param diagnostics     Output diagnostics (best/mean cost, ESS, etc.).
	bool dispatch_rollout(
		const DriverConfig& config,
		uint64_t step_index,
		const float* x0,
		const float* u_nominal,
		const ControlNoiseConfig& noise,
		const ByteView& model_params,
		const ByteView& cost_params,
		float* u_out,
		StepDiagnostics* diagnostics,
		std::string* error = nullptr
	);

	/// Dispatch a multi-step continuous sequence on the GPU.
	/// Automatically propagates state and warm-starts across iterations.
	bool dispatch_simulation(
		uint32_t num_steps,
		const DriverConfig& config,
		uint64_t start_step_index,
		const float* x0,
		const float* u_nominal,
		const ControlNoiseConfig& noise,
		const ByteView& model_params,
		const ByteView& cost_params,
		const SimulationResults& results,
		std::string* error = nullptr
	);

	/// Dispatch a batched multi-agent simulation on the GPU.
	/// All N agents share the same pipeline/params but have independent
	/// initial states and RNG seeds.
	bool dispatch_batch_simulation(
		uint32_t num_agents,
		uint32_t num_steps,
		const DriverConfig& config,
		uint64_t start_step_index,
		const float* x0_packed,
		const float* u_nominal_packed,
		const ControlNoiseConfig& noise,
		const ByteView& model_params,
		const ByteView& cost_params,
		const BatchSimulationResults& results,
		std::string* error = nullptr
	);

	/// Release GPU resources and reset internal state.
	bool reset(std::string* error = nullptr);

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

}  // namespace mppi_metal
