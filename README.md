# mppi_metal

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run

```bash
./build/minimal_example
```

## Unit tests

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Contract Draft (v1)

This section defines the intended public contract for an extensible MPPI library.

### Design decisions locked

- One model plugin per driver instance.
- Mostly C++ class API.
- Typed view wrappers instead of raw pointer-only interfaces.
- Hybrid parameter strategy: typed MPPI config + opaque user model/cost parameter blobs.
- Library owns the master rollout kernel: `mppi_rollout`.
- User provides device callables (Metal `[[visible]]` functions), not standalone compute kernels.
- Required user callable: dynamics.
- Optional user callables: stage cost and terminal cost.
- Numeric mode is configurable (FP32, FP16, or mixed).
- RNG policy is configurable (algorithm and state strategy).
- Validation uses a hard-fail layout contract (symbol presence + byte/alignment/index checks).
- `step(...)` is synchronous (blocking) in v1.

### Public API (proposed)

```cpp
namespace mppi_metal {

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

struct ControlBounds {
	// Per-control-dimension actuator limits.
	const float* u_min = nullptr;  // control_dim elements
	const float* u_max = nullptr;  // control_dim elements
	uint32_t control_dim = 0;
};

struct DriverConfig {
	uint32_t state_dim;
	uint32_t control_dim;
	uint32_t horizon;
	uint32_t sample_count;
	float lambda;
	ControlNoiseConfig noise;
	ControlBounds bounds;

	enum class NumericMode {
		kFp32,
		kFp16,
		kMixed
	};

	NumericMode numeric_mode = NumericMode::kFp32;
};

struct RngConfig {
	// v1 locks Philox for counter-based deterministic RNG.
	// Philox state is derived from step index and trajectory ID, ensuring determinism
	// regardless of threadgroup dispatch order.
	uint64_t seed = 0;
};

struct CallableNames {
	// Required device callable. Must be defined with [[visible]].
	std::string_view dynamics = "mppi_dynamics";

	// Optional device callables. If omitted, library uses default/no-op behavior.
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

	// Optional per-step overrides. If bytes == 0, the driver uses bound ProblemParams.
	ByteView model_params_override;
	ByteView cost_params_override;
};

struct StepDiagnostics {
	float best_cost;
	float mean_cost;
	float effective_sample_size;
	bool converged;
};

class MPPIDriver {
public:
	explicit MPPIDriver(const DriverConfig& config);

	bool initialize(const ModelPluginSpec& model, std::string* error = nullptr);
	bool set_problem_params(const ProblemParams& params, std::string* error = nullptr);

	// Blocking call: returns only after GPU rollout/cost/update work is complete.
	bool step(
		const StepInputs& inputs,
		const MutableControlSequenceView& u_next_out,
		StepDiagnostics* diagnostics,
		std::string* error = nullptr
	);

	// Keep error-string overloads for diagnostics.
	bool reset(std::string* error = nullptr);

	// Replace the internal nominal control warm start and reset MPPI state.
	// RNG state deterministically continues from the current step index.
	bool reset(const ControlSequenceView& u_nominal, std::string* error = nullptr);
};

}  // namespace mppi_metal
```

Notes:

- `initialize(...)` copies all externally owned strings/paths into driver-owned storage.
- `std::filesystem::path` is used for metallib ownership and platform-safe path semantics.
- `MPPIDriver()` constructor performs deep copies of all raw pointers in `DriverConfig`:
  - `ControlNoiseConfig.sigma` and `sigma_inv` are copied to internally managed host buffers.
  - `ControlBounds.u_min` and `u_max` are copied to internally managed host buffers.
  - This ensures no pointers dangle after construction, even if caller-provided arrays go out of scope.

### Kernel responsibilities

Library provides and dispatches the master `mppi_rollout` compute kernel.

End users implement Metal device callables marked with `[[visible]]`:

- `mppi_dynamics` (required): in-place state propagation logic.
- `mppi_stage_cost` (optional): stage cost callable.
- `mppi_terminal_cost` (optional): terminal cost callable.

The rollout kernel invokes user callables through a visible function table.

Library handles scheduling, RNG orchestration, weighting, control update, pipeline lifecycle, and validation.

### Device-side contract (Metal)

- User functions are `[[visible]]` callables, not `[[kernel]]` entrypoints.
- Function signatures must match library-defined typedefs in a shared Metal header (for example, `mppi_core.metal`).
- User parameter memory is passed through `ByteView` and must obey the declared `ParamLayout` size/alignment contract.
- Host-backed parameter buffers must be read through `device` or `constant` address space pointers.
- The contract does not use `void*` in MSL callable signatures.
- `mppi_dynamics` may mutate state in-place (takes `thread float* state`).
- `mppi_stage_cost` and `mppi_terminal_cost` must not mutate state (take `thread const float* state`) to prevent side-effects during the reduction pass.
- The RNG counter derives from step index and trajectory thread ID: `counter = seed_xor_index || step_index || trajectory_id`, ensuring deterministic noise generation across identical runs regardless of threadgroup execution order.

Example shape:

```cpp
[[visible]]
void mppi_dynamics(
	thread float* state,
	thread const float* control,
	device const uint8_t* model_params_bytes
);
```

### Host-side contract (C++ / Objective-C++)

`initialize(...)` creates a linked pipeline state:

1. Load the library metallib containing the master `mppi_rollout` kernel.
2. Load the user metallib containing `[[visible]]` callable functions.
3. Build `MTLLinkedFunctions` from user callable names.
4. Create `MTLComputePipelineState` for `mppi_rollout` with linked functions.
5. Create and populate `MTLVisibleFunctionTable` with callable handles at fixed indices.

`step(...)` binds the visible function table along with standard buffers before dispatch.

### Parameter update policy

You do not need to pass model/cost params to every `step(...)` call.

- Typical path: call `set_problem_params(...)` once, then call `step(...)` repeatedly.
- Dynamic path: provide `*_override` in `StepInputs` for values that change this step.
- If an override has `bytes == 0`, the driver uses the bound params.

This keeps the API efficient for mostly-static models while still supporting time-varying parameters.

### Hard-fail validation rules

`initialize(...)` fails if any of the following are true:

- Required library symbol `mppi_rollout` is missing.
- `user_metallib_path` cannot be loaded.
- Required callable `mppi_dynamics` is missing.
- Declared parameter size/alignment in `ParamLayout` is invalid.
- Declared layout does not match driver-side buffer/index binding contract.
- Problem dimensions in `DriverConfig` are invalid.
- `noise.sigma` / `noise.sigma_inv` are null or invalid for the selected covariance form.
- `bounds` are invalid (`u_min > u_max`, null pointer with nonzero `control_dim`, or dimension mismatch).

### Architectural constraints and mitigations

- Register pressure after linking can reduce occupancy.
- Initialization should query pipeline limits (for example `maxTotalThreadsPerThreadgroup`) and enforce policy thresholds.
- `ByteView` is intentionally opaque; structural correctness relies on `ParamLayout` and shared host/device struct definitions.
- RNG uses Philox in v1 (counter-based generator). Philox derives its state from step index and trajectory ID, guaranteeing determinism even if Metal threadgroups are dispatched out of order. `RngConfig` supplies only the base seed. Stateful generators (e.g., PCG32) would lose determinism under uncontrolled dispatch order and are reserved for future non-deterministic modes.

### Synchronous semantics

`step(...)` does all work before returning:

1. Upload/validate input buffers.
2. Resolve active params (overrides if present, otherwise bound params).
3. Dispatch master `mppi_rollout` with linked user callables and visible function table bindings.
4. Write updated control sequence into `u_next_out` and fill `diagnostics`.
5. Return `true`/`false` with populated outputs or `error`.

No background handle/future/poll API is exposed in v1.

### Warm-start semantics (locked)

After a successful `step(...)`, the internal nominal sequence is advanced as:

1. Shift controls left by one horizon step (`u[t] <- u[t+1]`).
2. Fill the last control with the previous terminal control (hold-last policy for v1).
3. If `u_nominal_override` is provided in the next step, it replaces this internal sequence for that step.

### Reset semantics (locked)

- `reset(...)` must fail and set an error string on invalid inputs (for example, nominal control shape mismatch).
- `reset(...)` keeps bound `ProblemParams` unchanged.
- `reset(...)` defaults to deterministic continuation of RNG state (does not reseed unless an explicit reseed API is added later).

### Error and validation policy (locked)

- For `reset(const ControlSequenceView&, ...)`, shape must match exactly:
  - `u_nominal.horizon == DriverConfig::horizon`
  - `u_nominal.control_dim == DriverConfig::control_dim`
  - `u_nominal.data != nullptr` when `horizon * control_dim > 0`
- Any mismatch is a hard fail and returns `false`.
- `std::string* error` may be `nullptr`. In that case, methods still return `false` on failure but do not write an error message.
- v1 keeps `bool + std::string*` for C++17 compatibility; a future C++23 profile may provide `std::expected` overloads.

Expected output shape (values may match exactly for this example):

```text
Input:  0 1 2 3
Output: 0.25 1.25 2.25 3.25
```

### Batched multi-agent simulation

`simulate_batch(...)` runs N independent MPPI agents in a single GPU dispatch.
All agents share the same dynamics/cost functions and parameters but start from
different initial states and use independent RNG seeds.

```cpp
struct BatchSimulationResults {
	float* states_out = nullptr;     // [num_agents * num_steps * state_dim]
	float* controls_out = nullptr;   // [num_agents * num_steps * control_dim]
	float* costs_out = nullptr;      // [num_agents * num_steps]
	float* terminal_states_out = nullptr;  // [num_agents * state_dim], final step only
	uint32_t num_agents = 0;
	uint32_t num_steps = 0;
};

bool simulate_batch(
	uint32_t num_agents,
	uint32_t num_steps,
	const float* x0_packed,        // [num_agents * state_dim]
	const BatchSimulationResults& results,
	std::string* error = nullptr
);
```

**Memory layout**: All buffers are contiguous and indexed as
`[step * num_agents * dim + agent_idx * dim + element]`.

For `terminal_states_out`, the library copies only the final simulated state
for each agent (step index `num_steps - 1`) into a packed
`[num_agents * state_dim]` buffer using `[agent_idx * state_dim + element]`
indexing. If `num_steps == 0`, no terminal-state copy is performed.

**RNG independence**: Each agent derives its Philox key as `base_seed + agent_idx`.
Agent 0 produces identical noise to the single-agent `simulate()` path, preserving
backward compatibility for N=1.

**GPU dispatch**: The batch kernels use 2D grid dispatches:

- Rollout: `(sample_count, num_agents)` threads.
- Weights: `num_agents` threadgroups (one per agent).
- Reduce / Propagate: `(horizon * control_dim, num_agents)` threads.

## Next for MPPI

1. Implement this contract in headers/source.
2. Add layout-contract conformance checks in `initialize(...)`.
3. Add deterministic CPU reference checks for GPU validation.
4. Add richer diagnostics (timing and effective sample size history).
