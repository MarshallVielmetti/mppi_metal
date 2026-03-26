#pragma once

#include "mppi_metal/mppi_types.hpp"

#include <memory>
#include <string>

namespace mppi_metal {

/// MPPI controller driver backed by Metal GPU compute.
///
/// Lifecycle:
///   1. Construct with a DriverConfig (deep-copies all pointer data).
///   2. Call initialize() with a ModelPluginSpec to link the pipeline.
///   3. Optionally call set_problem_params() to bind model/cost params.
///   4. Call step() in a loop.
///   5. Call reset() to re-initialise warm-start state.
class MPPIDriver {
public:
	/// Constructor deep-copies all raw pointers in DriverConfig:
	///   - ControlNoiseConfig.sigma / sigma_inv
	///   - ControlBounds.u_min / u_max
	/// so all externally owned arrays may safely go out of scope.
	explicit MPPIDriver(const DriverConfig& config);
	~MPPIDriver();

	MPPIDriver(const MPPIDriver&) = delete;
	MPPIDriver& operator=(const MPPIDriver&) = delete;

	MPPIDriver(MPPIDriver&&) noexcept;
	MPPIDriver& operator=(MPPIDriver&&) noexcept;

	/// Load user metallib, link pipeline, build visible function table.
	/// Copies all externally owned strings/paths into driver-owned storage.
	///
	/// Hard-fails if any validation rule is violated (see README §Hard-fail
	/// validation rules).
	bool initialize(const ModelPluginSpec& model, std::string* error = nullptr);

	/// Bind model/cost parameter blobs for reuse across step() calls.
	bool set_problem_params(const ProblemParams& params, std::string* error = nullptr);

	/// Blocking call: dispatches rollout/cost/update on the GPU and returns
	/// the updated control sequence via @p u_next_out.
	///
	/// After a successful step the internal nominal sequence is warm-started:
	///   1. Shift controls left by one horizon step.
	///   2. Fill last control with previous terminal control (hold-last).
	///   3. u_nominal_override (if provided) replaces the warm-start for this
	///      step only.
	bool step(
		const StepInputs& inputs,
		const MutableControlSequenceView& u_next_out,
		StepDiagnostics* diagnostics,
		std::string* error = nullptr
	);

	/// Reset MPPI state.  Bound ProblemParams are kept unchanged.
	/// RNG state deterministically continues from the current step index.
	bool reset(std::string* error = nullptr);

	/// Reset with an explicit nominal control warm-start.
	/// Fails if the shape does not match DriverConfig exactly.
	bool reset(const ControlSequenceView& u_nominal, std::string* error = nullptr);

	/// Run a continuous multi-step MPPI simulation on the GPU.
	/// Encodes the entire sequence of steps into a single command buffer,
	/// automatically propagating the "true" state using the optimal control
	/// and automatically handling the warm-start shift.
	bool simulate(
		uint32_t num_steps,
		const StepInputs& initial_inputs,
		const SimulationResults& results,
		std::string* error = nullptr
	);

private:
	class Impl;
	std::unique_ptr<Impl> impl_;
};

}  // namespace mppi_metal
