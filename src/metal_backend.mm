#include "metal_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#ifndef MPPI_ROLLOUT_METALLIB_PATH
#error "MPPI_ROLLOUT_METALLIB_PATH must be defined at build time."
#endif

namespace mppi_metal {

namespace {

void set_error(std::string *error, const std::string &message) {
  if (error != nullptr) {
    *error = message;
  }
}

std::string to_std_string(NSString *str) {
  if (str == nil)
    return {};
  return std::string([str UTF8String]);
}

} // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct MetalBackend::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> command_queue = nil;

  id<MTLComputePipelineState> rollout_pipeline = nil;
  id<MTLVisibleFunctionTable> dynamics_table = nil;
  id<MTLVisibleFunctionTable> stage_cost_table = nil;
  id<MTLVisibleFunctionTable> terminal_cost_table = nil;

  id<MTLComputePipelineState> weights_pipeline = nil;
  id<MTLComputePipelineState> reduce_pipeline = nil;
  id<MTLComputePipelineState> propagate_pipeline = nil;
  id<MTLVisibleFunctionTable> propagate_dynamics_table = nil;

  // Batch pipeline states
  id<MTLComputePipelineState> rollout_batch_pipeline = nil;
  id<MTLVisibleFunctionTable> batch_dynamics_table = nil;
  id<MTLVisibleFunctionTable> batch_stage_cost_table = nil;
  id<MTLVisibleFunctionTable> batch_terminal_cost_table = nil;
  id<MTLComputePipelineState> weights_batch_pipeline = nil;
  id<MTLComputePipelineState> reduce_batch_pipeline = nil;
  id<MTLComputePipelineState> propagate_batch_pipeline = nil;
  id<MTLVisibleFunctionTable> batch_propagate_dynamics_table = nil;

  id<MTLBuffer> diagnostics_buf = nil;

  id<MTLLibrary> library_metallib = nil;
  id<MTLLibrary> user_metallib = nil;
  id<MTLBuffer> model_params_buffer = nil;
  id<MTLBuffer> cost_params_buffer = nil;

  // Persistent shared scratchpad buffers
  id<MTLBuffer> u_nom_buf = nil;
  id<MTLBuffer> sigma_buf = nil;
  id<MTLBuffer> umin_buf = nil;
  id<MTLBuffer> umax_buf = nil;
  id<MTLBuffer> costs_buf = nil;
  id<MTLBuffer> weights_buf = nil;
  id<MTLBuffer> u_out_buf = nil;
  id<MTLBuffer> x0_buf = nil;

  id<MTLBuffer> his_state_buf = nil;
  id<MTLBuffer> his_ctrl_buf = nil;
  id<MTLBuffer> his_cost_buf = nil;

  uint32_t allocated_S = 0;
  uint32_t allocated_H = 0;
  uint32_t allocated_cdim = 0;
  uint32_t allocated_sdim = 0;
  uint32_t allocated_num_steps = 0;

  // Batch scratchpad buffers (separate from single-agent)
  id<MTLBuffer> batch_x0_buf = nil;
  id<MTLBuffer> batch_u_nom_buf = nil;
  id<MTLBuffer> batch_costs_buf = nil;
  id<MTLBuffer> batch_weights_buf = nil;
  id<MTLBuffer> batch_u_out_buf = nil;
  id<MTLBuffer> batch_diag_buf = nil;
  id<MTLBuffer> batch_his_state_buf = nil;
  id<MTLBuffer> batch_his_ctrl_buf = nil;
  id<MTLBuffer> batch_his_cost_buf = nil;
  id<MTLBuffer> batch_terminal_state_buf = nil;
  uint32_t batch_alloc_N = 0;
  uint32_t batch_alloc_S = 0;
  uint32_t batch_alloc_H = 0;
  uint32_t batch_alloc_cdim = 0;
  uint32_t batch_alloc_sdim = 0;
  uint32_t batch_alloc_num_steps = 0;

  NSUInteger rollout_max_tpg = 0;
  NSUInteger reduce_max_tpg = 0;
  uint32_t rng_seed = 0;
  bool initialized = false;

  void ensure_batch_scratchpad(const DriverConfig &config, uint32_t num_agents,
                               uint32_t num_steps) {
    const uint32_t S = config.sample_count;
    const uint32_t H = config.horizon;
    const uint32_t cdim = config.control_dim;
    const uint32_t sdim = config.state_dim;
    const uint32_t N = num_agents;

    // Avoid deallocating and reallocating if num_steps changed/decreased
    // (common case).
    bool changed = (batch_alloc_N != N) || (batch_alloc_S != S) ||
                   (batch_alloc_H != H) || (batch_alloc_cdim != cdim) ||
                   (batch_alloc_sdim != sdim) ||
                   (num_steps > batch_alloc_num_steps);
    if (!changed)
      return;

    id<MTLDevice> dev = device;
    const size_t seq_len = static_cast<size_t>(H) * cdim;

    batch_x0_buf = [dev newBufferWithLength:N * sdim * sizeof(float)
                                    options:MTLResourceStorageModeShared];
    batch_u_nom_buf = [dev newBufferWithLength:N * seq_len * sizeof(float)
                                       options:MTLResourceStorageModeShared];
    batch_costs_buf = [dev newBufferWithLength:N * S * sizeof(float)
                                       options:MTLResourceStorageModeShared];
    batch_weights_buf = [dev newBufferWithLength:N * S * sizeof(float)
                                         options:MTLResourceStorageModeShared];
    batch_u_out_buf = [dev newBufferWithLength:N * seq_len * sizeof(float)
                                       options:MTLResourceStorageModeShared];
    batch_diag_buf = [dev newBufferWithLength:N * 3 * sizeof(float)
                                      options:MTLResourceStorageModeShared];
    batch_his_state_buf =
        [dev newBufferWithLength:N * num_steps * sdim * sizeof(float)
                         options:MTLResourceStorageModeShared];
    batch_his_ctrl_buf =
        [dev newBufferWithLength:N * num_steps * cdim * sizeof(float)
                         options:MTLResourceStorageModeShared];
    batch_his_cost_buf = [dev newBufferWithLength:N * num_steps * sizeof(float)
                                          options:MTLResourceStorageModeShared];
    batch_terminal_state_buf =
        [dev newBufferWithLength:N * sdim * sizeof(float)
                         options:MTLResourceStorageModeShared];

    batch_alloc_N = N;
    batch_alloc_S = S;
    batch_alloc_H = H;
    batch_alloc_cdim = cdim;
    batch_alloc_sdim = sdim;
    batch_alloc_num_steps = num_steps;
  }

  void ensure_scratchpad(const DriverConfig &config, uint32_t num_steps) {
    const uint32_t S = config.sample_count;
    const uint32_t H = config.horizon;
    const uint32_t cdim = config.control_dim;
    const uint32_t sdim = config.state_dim;

    bool size_changed = (allocated_S != S) || (allocated_H != H) ||
                        (allocated_cdim != cdim) || (allocated_sdim != sdim);

    id<MTLDevice> dev = device;
    if (size_changed) {
      const size_t seq_len = static_cast<size_t>(H) * cdim;
      u_nom_buf = [dev newBufferWithLength:seq_len * sizeof(float)
                                   options:MTLResourceStorageModeShared];
      sigma_buf = [dev newBufferWithLength:cdim * sizeof(float)
                                   options:MTLResourceStorageModeShared];
      umin_buf = [dev newBufferWithLength:cdim * sizeof(float)
                                  options:MTLResourceStorageModeShared];
      umax_buf = [dev newBufferWithLength:cdim * sizeof(float)
                                  options:MTLResourceStorageModeShared];
      costs_buf = [dev newBufferWithLength:S * sizeof(float)
                                   options:MTLResourceStorageModeShared];
      weights_buf = [dev newBufferWithLength:S * sizeof(float)
                                     options:MTLResourceStorageModeShared];
      u_out_buf = [dev newBufferWithLength:seq_len * sizeof(float)
                                   options:MTLResourceStorageModeShared];
      x0_buf = [dev newBufferWithLength:sdim * sizeof(float)
                                options:MTLResourceStorageModeShared];

      allocated_S = S;
      allocated_H = H;
      allocated_cdim = cdim;
      allocated_sdim = sdim;
    }

    if (num_steps > 0 && allocated_num_steps != num_steps) {
      his_state_buf = [dev newBufferWithLength:num_steps * sdim * sizeof(float)
                                       options:MTLResourceStorageModeShared];
      his_ctrl_buf = [dev newBufferWithLength:num_steps * cdim * sizeof(float)
                                      options:MTLResourceStorageModeShared];
      his_cost_buf = [dev newBufferWithLength:num_steps * sizeof(float)
                                      options:MTLResourceStorageModeShared];
      allocated_num_steps = num_steps;
    }
  }
};

// ---------------------------------------------------------------------------
// Construction / destruction / move
// ---------------------------------------------------------------------------

MetalBackend::MetalBackend() : impl_(std::make_unique<Impl>()) {}
MetalBackend::~MetalBackend() = default;
MetalBackend::MetalBackend(MetalBackend &&) noexcept = default;
MetalBackend &MetalBackend::operator=(MetalBackend &&) noexcept = default;

// ---------------------------------------------------------------------------
// initialize
// ---------------------------------------------------------------------------

bool MetalBackend::initialize(const ModelPluginSpec &model,
                              const DriverConfig &config, std::string *error) {
  impl_->device = MTLCreateSystemDefaultDevice();
  if (impl_->device == nil) {
    set_error(error, "Failed to get default Metal device.");
    return false;
  }

  NSString *lib_path =
      [NSString stringWithUTF8String:MPPI_ROLLOUT_METALLIB_PATH];
  NSError *ns_err = nil;
  impl_->library_metallib =
      [impl_->device newLibraryWithURL:[NSURL fileURLWithPath:lib_path]
                                 error:&ns_err];
  if (impl_->library_metallib == nil) {
    set_error(error, "Failed to load library metallib: " +
                         to_std_string(ns_err.localizedDescription));
    return false;
  }

  // ---- Rollout kernel (Pass 1) ----
  id<MTLFunction> rollout_fn =
      [impl_->library_metallib newFunctionWithName:@"mppi_rollout"];
  if (rollout_fn == nil) {
    set_error(error, "Required library symbol 'mppi_rollout' missing.");
    return false;
  }

  // ---- Reduce kernel (Pass 3) ----
  id<MTLFunction> reduce_fn =
      [impl_->library_metallib newFunctionWithName:@"mppi_reduce"];
  if (reduce_fn == nil) {
    set_error(error, "Required library symbol 'mppi_reduce' missing.");
    return false;
  }

  // ---- Weights kernel (Pass 2) ----
  id<MTLFunction> weights_fn =
      [impl_->library_metallib newFunctionWithName:@"mppi_compute_weights"];
  if (weights_fn == nil) {
    set_error(error, "Required library symbol 'mppi_compute_weights' missing.");
    return false;
  }

  // ---- Propagation kernel (Pass 4) ----
  id<MTLFunction> propagate_fn =
      [impl_->library_metallib newFunctionWithName:@"mppi_propagate_and_shift"];
  if (propagate_fn == nil) {
    set_error(error,
              "Required library symbol 'mppi_propagate_and_shift' missing.");
    return false;
  }

  // ---- Load user metallib ----
  std::string user_path_str = model.user_metallib_path.string();
  if (user_path_str.empty()) {
    set_error(error, "user_metallib_path is empty.");
    return false;
  }
  impl_->user_metallib = [impl_->device
      newLibraryWithURL:[NSURL fileURLWithPath:[NSString
                                                   stringWithUTF8String:
                                                       user_path_str.c_str()]]
                  error:&ns_err];
  if (impl_->user_metallib == nil) {
    set_error(error, "Failed to load user metallib '" + user_path_str +
                         "': " + to_std_string(ns_err.localizedDescription));
    return false;
  }

  // Dynamics (required).
  std::string dynamics_name(model.callables.dynamics);
  id<MTLFunction> dynamics_fn = [impl_->user_metallib
      newFunctionWithName:[NSString
                              stringWithUTF8String:dynamics_name.c_str()]];
  if (dynamics_fn == nil) {
    set_error(error, "Required callable '" + dynamics_name +
                         "' not found in user metallib.");
    return false;
  }

  // Stage cost (optional → default).
  id<MTLFunction> stage_cost_fn = nil;
  if (!model.callables.stage_cost.empty()) {
    std::string sc(model.callables.stage_cost);
    stage_cost_fn = [impl_->user_metallib
        newFunctionWithName:[NSString stringWithUTF8String:sc.c_str()]];
    if (stage_cost_fn == nil) {
      set_error(error, "Callable '" + sc + "' not found in user metallib.");
      return false;
    }
  } else {
    stage_cost_fn = [impl_->library_metallib
        newFunctionWithName:@"mppi_default_stage_cost"];
  }

  // Terminal cost (optional → default).
  id<MTLFunction> terminal_cost_fn = nil;
  if (!model.callables.terminal_cost.empty()) {
    std::string tc(model.callables.terminal_cost);
    terminal_cost_fn = [impl_->user_metallib
        newFunctionWithName:[NSString stringWithUTF8String:tc.c_str()]];
    if (terminal_cost_fn == nil) {
      set_error(error, "Callable '" + tc + "' not found in user metallib.");
      return false;
    }
  } else {
    terminal_cost_fn = [impl_->library_metallib
        newFunctionWithName:@"mppi_default_terminal_cost"];
  }

  // ---- Build rollout pipeline (linked with user callables) ----
  MTLLinkedFunctions *linked = [[MTLLinkedFunctions alloc] init];
  linked.functions = @[ dynamics_fn, stage_cost_fn, terminal_cost_fn ];

  MTLComputePipelineDescriptor *rollout_desc =
      [[MTLComputePipelineDescriptor alloc] init];
  rollout_desc.computeFunction = rollout_fn;
  rollout_desc.linkedFunctions = linked;

  impl_->rollout_pipeline =
      [impl_->device newComputePipelineStateWithDescriptor:rollout_desc
                                                   options:0
                                                reflection:nil
                                                     error:&ns_err];
  if (impl_->rollout_pipeline == nil) {
    set_error(error, "Failed to create rollout pipeline: " +
                         to_std_string(ns_err.localizedDescription));
    return false;
  }

  // ---- Build weights pipeline (no linked functions) ----
  impl_->weights_pipeline =
      [impl_->device newComputePipelineStateWithFunction:weights_fn
                                                   error:&ns_err];
  if (impl_->weights_pipeline == nil) {
    set_error(error, "Failed to create weights pipeline: " +
                         to_std_string(ns_err.localizedDescription));
    return false;
  }

  // ---- Build reduce pipeline (no linked functions) ----
  impl_->reduce_pipeline =
      [impl_->device newComputePipelineStateWithFunction:reduce_fn
                                                   error:&ns_err];
  if (impl_->reduce_pipeline == nil) {
    set_error(error, "Failed to create reduce pipeline: " +
                         to_std_string(ns_err.localizedDescription));
    return false;
  }

  // ---- Build propagate pipeline (linked with user callables) ----
  MTLComputePipelineDescriptor *propagate_desc =
      [[MTLComputePipelineDescriptor alloc] init];
  propagate_desc.computeFunction = propagate_fn;
  propagate_desc.linkedFunctions = linked;

  impl_->propagate_pipeline =
      [impl_->device newComputePipelineStateWithDescriptor:propagate_desc
                                                   options:0
                                                reflection:nil
                                                     error:&ns_err];
  if (impl_->propagate_pipeline == nil) {
    set_error(error, "Failed to create propagate pipeline: " +
                         to_std_string(ns_err.localizedDescription));
    return false;
  }

  impl_->diagnostics_buf =
      [impl_->device newBufferWithLength:3 * sizeof(float)
                                 options:MTLResourceStorageModeShared];

  // ---- Visible function tables (1 entry each) ----
  auto make_table = [&](id<MTLFunction> fn) -> id<MTLVisibleFunctionTable> {
    MTLVisibleFunctionTableDescriptor *td =
        [[MTLVisibleFunctionTableDescriptor alloc] init];
    td.functionCount = 1;
    id<MTLVisibleFunctionTable> table =
        [impl_->rollout_pipeline newVisibleFunctionTableWithDescriptor:td];
    if (table != nil) {
      id<MTLFunctionHandle> handle =
          [impl_->rollout_pipeline functionHandleWithFunction:fn];
      if (handle != nil)
        [table setFunction:handle atIndex:0];
    }
    return table;
  };

  impl_->dynamics_table = make_table(dynamics_fn);
  impl_->stage_cost_table = make_table(stage_cost_fn);
  impl_->terminal_cost_table = make_table(terminal_cost_fn);

  if (!impl_->dynamics_table || !impl_->stage_cost_table ||
      !impl_->terminal_cost_table) {
    set_error(error, "Failed to create visible function tables.");
    return false;
  }

  // ---- Visible function table for propagate ----
  MTLVisibleFunctionTableDescriptor *td_prop =
      [[MTLVisibleFunctionTableDescriptor alloc] init];
  td_prop.functionCount = 1;
  impl_->propagate_dynamics_table =
      [impl_->propagate_pipeline newVisibleFunctionTableWithDescriptor:td_prop];
  if (impl_->propagate_dynamics_table != nil) {
    id<MTLFunctionHandle> handle =
        [impl_->propagate_pipeline functionHandleWithFunction:dynamics_fn];
    if (handle != nil)
      [impl_->propagate_dynamics_table setFunction:handle atIndex:0];
  }

  if (!impl_->propagate_dynamics_table) {
    set_error(error, "Failed to create propagate dynamics table.");
    return false;
  }

  impl_->command_queue = [impl_->device newCommandQueue];
  if (impl_->command_queue == nil) {
    set_error(error, "Failed to create command queue.");
    return false;
  }

  // ---- Batch kernels ----
  id<MTLFunction> rollout_batch_fn =
      [impl_->library_metallib newFunctionWithName:@"mppi_rollout_batch"];
  id<MTLFunction> weights_batch_fn = [impl_->library_metallib
      newFunctionWithName:@"mppi_compute_weights_batch"];
  id<MTLFunction> reduce_batch_fn =
      [impl_->library_metallib newFunctionWithName:@"mppi_reduce_batch"];
  id<MTLFunction> propagate_batch_fn = [impl_->library_metallib
      newFunctionWithName:@"mppi_propagate_and_shift_batch"];

  if (!rollout_batch_fn || !weights_batch_fn || !reduce_batch_fn ||
      !propagate_batch_fn) {
    set_error(error, "Missing batch kernel symbols in metallib.");
    return false;
  }

  // Batch rollout pipeline (linked)
  MTLComputePipelineDescriptor *rollout_batch_desc =
      [[MTLComputePipelineDescriptor alloc] init];
  rollout_batch_desc.computeFunction = rollout_batch_fn;
  rollout_batch_desc.linkedFunctions = linked;

  impl_->rollout_batch_pipeline =
      [impl_->device newComputePipelineStateWithDescriptor:rollout_batch_desc
                                                   options:0
                                                reflection:nil
                                                     error:&ns_err];
  if (impl_->rollout_batch_pipeline == nil) {
    set_error(error, "Failed to create batch rollout pipeline: " +
                         to_std_string(ns_err.localizedDescription));
    return false;
  }

  // Batch weights pipeline (no linked functions)
  impl_->weights_batch_pipeline =
      [impl_->device newComputePipelineStateWithFunction:weights_batch_fn
                                                   error:&ns_err];
  if (impl_->weights_batch_pipeline == nil) {
    set_error(error, "Failed to create batch weights pipeline: " +
                         to_std_string(ns_err.localizedDescription));
    return false;
  }

  // Batch reduce pipeline (no linked functions)
  impl_->reduce_batch_pipeline =
      [impl_->device newComputePipelineStateWithFunction:reduce_batch_fn
                                                   error:&ns_err];
  if (impl_->reduce_batch_pipeline == nil) {
    set_error(error, "Failed to create batch reduce pipeline: " +
                         to_std_string(ns_err.localizedDescription));
    return false;
  }

  // Batch propagate pipeline (linked)
  MTLComputePipelineDescriptor *propagate_batch_desc =
      [[MTLComputePipelineDescriptor alloc] init];
  propagate_batch_desc.computeFunction = propagate_batch_fn;
  propagate_batch_desc.linkedFunctions = linked;

  impl_->propagate_batch_pipeline =
      [impl_->device newComputePipelineStateWithDescriptor:propagate_batch_desc
                                                   options:0
                                                reflection:nil
                                                     error:&ns_err];
  if (impl_->propagate_batch_pipeline == nil) {
    set_error(error, "Failed to create batch propagate pipeline: " +
                         to_std_string(ns_err.localizedDescription));
    return false;
  }

  // Batch visible function tables for rollout
  auto make_batch_rollout_table =
      [&](id<MTLFunction> fn) -> id<MTLVisibleFunctionTable> {
    MTLVisibleFunctionTableDescriptor *td =
        [[MTLVisibleFunctionTableDescriptor alloc] init];
    td.functionCount = 1;
    id<MTLVisibleFunctionTable> table = [impl_->rollout_batch_pipeline
        newVisibleFunctionTableWithDescriptor:td];
    if (table != nil) {
      id<MTLFunctionHandle> handle =
          [impl_->rollout_batch_pipeline functionHandleWithFunction:fn];
      if (handle != nil)
        [table setFunction:handle atIndex:0];
    }
    return table;
  };

  impl_->batch_dynamics_table = make_batch_rollout_table(dynamics_fn);
  impl_->batch_stage_cost_table = make_batch_rollout_table(stage_cost_fn);
  impl_->batch_terminal_cost_table = make_batch_rollout_table(terminal_cost_fn);

  if (!impl_->batch_dynamics_table || !impl_->batch_stage_cost_table ||
      !impl_->batch_terminal_cost_table) {
    set_error(error, "Failed to create batch visible function tables.");
    return false;
  }

  // Batch visible function table for propagate
  MTLVisibleFunctionTableDescriptor *td_batch_prop =
      [[MTLVisibleFunctionTableDescriptor alloc] init];
  td_batch_prop.functionCount = 1;
  impl_->batch_propagate_dynamics_table = [impl_->propagate_batch_pipeline
      newVisibleFunctionTableWithDescriptor:td_batch_prop];
  if (impl_->batch_propagate_dynamics_table != nil) {
    id<MTLFunctionHandle> handle = [impl_->propagate_batch_pipeline
        functionHandleWithFunction:dynamics_fn];
    if (handle != nil)
      [impl_->batch_propagate_dynamics_table setFunction:handle atIndex:0];
  }

  if (!impl_->batch_propagate_dynamics_table) {
    set_error(error, "Failed to create batch propagate dynamics table.");
    return false;
  }

  impl_->rollout_max_tpg =
      impl_->rollout_pipeline.maxTotalThreadsPerThreadgroup;
  impl_->reduce_max_tpg = impl_->reduce_pipeline.maxTotalThreadsPerThreadgroup;
  impl_->rng_seed = static_cast<uint32_t>(model.rng.seed);
  impl_->initialized = true;
  return true;
}

// ---------------------------------------------------------------------------
// bind_params
// ---------------------------------------------------------------------------

bool MetalBackend::bind_params(const ByteView &model_params,
                               const ByteView &cost_params,
                               std::string *error) {
  if (!impl_->initialized) {
    set_error(error, "MetalBackend is not initialized.");
    return false;
  }
  if (model_params.bytes > 0 && model_params.data != nullptr) {
    impl_->model_params_buffer =
        [impl_->device newBufferWithBytes:model_params.data
                                   length:model_params.bytes
                                  options:MTLResourceStorageModeShared];
  } else {
    impl_->model_params_buffer = nil;
  }
  if (cost_params.bytes > 0 && cost_params.data != nullptr) {
    impl_->cost_params_buffer =
        [impl_->device newBufferWithBytes:cost_params.data
                                   length:cost_params.bytes
                                  options:MTLResourceStorageModeShared];
  } else {
    impl_->cost_params_buffer = nil;
  }
  return true;
}

// ---------------------------------------------------------------------------
// dispatch_rollout  (two-pass: rollout → reduce)
// ---------------------------------------------------------------------------

bool MetalBackend::dispatch_rollout(
    const DriverConfig &config, uint64_t step_index, const float *x0,
    const float *u_nominal, const ControlNoiseConfig &noise,
    const ByteView &model_params, const ByteView &cost_params, float *u_out,
    StepDiagnostics *diagnostics, std::string *error) {
  if (!impl_->initialized) {
    set_error(error, "MetalBackend is not initialized.");
    return false;
  }

  const uint32_t S = config.sample_count;
  const uint32_t H = config.horizon;
  const uint32_t sdim = config.state_dim;
  const uint32_t cdim = config.control_dim;
  const size_t seq_len = static_cast<size_t>(H) * cdim;

  impl_->ensure_scratchpad(config, 0);

  std::memcpy(impl_->u_nom_buf.contents, u_nominal, seq_len * sizeof(float));
  std::memcpy(impl_->sigma_buf.contents, noise.sigma, cdim * sizeof(float));
  std::memcpy(impl_->umin_buf.contents, config.bounds.u_min,
              cdim * sizeof(float));
  std::memcpy(impl_->umax_buf.contents, config.bounds.u_max,
              cdim * sizeof(float));
  std::memcpy(impl_->x0_buf.contents, x0, sdim * sizeof(float));

  uint32_t seed = impl_->rng_seed;
  uint32_t step_idx32 = static_cast<uint32_t>(step_index);
  float lambda = config.lambda;

  auto resolve_param_buf = [&](const ByteView &bv,
                               id<MTLBuffer> bound) -> id<MTLBuffer> {
    if (bv.bytes > 0 && bv.data != nullptr)
      return [impl_->device newBufferWithBytes:bv.data
                                        length:bv.bytes
                                       options:MTLResourceStorageModeShared];
    if (bound != nil)
      return bound;
    uint8_t z = 0;
    return [impl_->device newBufferWithBytes:&z
                                      length:1
                                     options:MTLResourceStorageModeShared];
  };
  id<MTLBuffer> mp_buf =
      resolve_param_buf(model_params, impl_->model_params_buffer);
  id<MTLBuffer> cp_buf =
      resolve_param_buf(cost_params, impl_->cost_params_buffer);

  id<MTLCommandBuffer> cmd_buf = [impl_->command_queue commandBuffer];

  // Encoder Fusion (Level 1): Use one encoder for all passes.
  id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

  // ---- Pass 1: Rollout ----
  [enc setComputePipelineState:impl_->rollout_pipeline];
  [enc setBuffer:impl_->x0_buf offset:0 atIndex:0];
  [enc setBuffer:impl_->u_nom_buf offset:0 atIndex:1];
  [enc setBuffer:mp_buf offset:0 atIndex:2];
  [enc setBuffer:cp_buf offset:0 atIndex:3];
  [enc setBuffer:impl_->costs_buf offset:0 atIndex:4];
  [enc setBytes:&sdim length:sizeof(uint32_t) atIndex:5];
  [enc setBytes:&cdim length:sizeof(uint32_t) atIndex:6];
  [enc setBytes:&H length:sizeof(uint32_t) atIndex:7];
  [enc setBytes:&S length:sizeof(uint32_t) atIndex:8];
  [enc setBuffer:impl_->umin_buf offset:0 atIndex:9];
  [enc setBuffer:impl_->umax_buf offset:0 atIndex:10];
  [enc setVisibleFunctionTable:impl_->dynamics_table atBufferIndex:11];
  [enc setVisibleFunctionTable:impl_->stage_cost_table atBufferIndex:12];
  [enc setVisibleFunctionTable:impl_->terminal_cost_table atBufferIndex:13];
  [enc setBuffer:impl_->sigma_buf offset:0 atIndex:14];
  [enc setBytes:&seed length:sizeof(uint32_t) atIndex:15];
  [enc setBytes:&step_idx32 length:sizeof(uint32_t) atIndex:16];

  NSUInteger tpg = std::min<NSUInteger>(impl_->rollout_max_tpg, 256);
  [enc dispatchThreads:MTLSizeMake(S, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

  [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

  // ---- Pass 2: Weights ----
  [enc setComputePipelineState:impl_->weights_pipeline];
  [enc setBuffer:impl_->costs_buf offset:0 atIndex:0];
  [enc setBuffer:impl_->weights_buf offset:0 atIndex:1];
  [enc setBuffer:impl_->diagnostics_buf offset:0 atIndex:2];
  [enc setBytes:&S length:sizeof(uint32_t) atIndex:3];
  [enc setBytes:&lambda length:sizeof(float) atIndex:4];

  NSUInteger tpg_w = std::min<NSUInteger>(
      impl_->weights_pipeline.maxTotalThreadsPerThreadgroup, 1024);
  [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tpg_w, 1, 1)];

  [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

  // ---- Pass 3: Reduce ----
  [enc setComputePipelineState:impl_->reduce_pipeline];
  [enc setBuffer:impl_->weights_buf offset:0 atIndex:0];
  [enc setBuffer:impl_->u_nom_buf offset:0 atIndex:1];
  [enc setBuffer:impl_->u_out_buf offset:0 atIndex:2];
  [enc setBuffer:impl_->sigma_buf offset:0 atIndex:3];
  [enc setBuffer:impl_->umin_buf offset:0 atIndex:4];
  [enc setBuffer:impl_->umax_buf offset:0 atIndex:5];
  [enc setBytes:&S length:sizeof(uint32_t) atIndex:6];
  [enc setBytes:&H length:sizeof(uint32_t) atIndex:7];
  [enc setBytes:&cdim length:sizeof(uint32_t) atIndex:8];
  [enc setBytes:&seed length:sizeof(uint32_t) atIndex:9];
  [enc setBytes:&step_idx32 length:sizeof(uint32_t) atIndex:10];

  NSUInteger tpg2 = std::min<NSUInteger>(impl_->reduce_max_tpg, 256);
  [enc dispatchThreads:MTLSizeMake(seq_len, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tpg2, 1, 1)];

  [enc endEncoding];

  // Submit all 3 passes and wait
  [cmd_buf commit];
  [cmd_buf waitUntilCompleted];

  if (cmd_buf.status == MTLCommandBufferStatusError) {
    set_error(error, "Command buffer failed: " +
                         to_std_string(cmd_buf.error.localizedDescription));
    return false;
  }

  if (diagnostics != nullptr) {
    const float *d =
        static_cast<const float *>(impl_->diagnostics_buf.contents);
    diagnostics->best_cost = d[0];
    diagnostics->mean_cost = d[1];
    diagnostics->effective_sample_size = d[2];
    diagnostics->converged = false;
  }

  // ---- Read back updated controls ----
  std::memcpy(u_out, impl_->u_out_buf.contents, seq_len * sizeof(float));

  return true;
}

// ---------------------------------------------------------------------------
// dispatch_simulation
// ---------------------------------------------------------------------------

bool MetalBackend::dispatch_simulation(
    uint32_t num_steps, const DriverConfig &config, uint64_t start_step_index,
    const float *x0, const float *u_nominal, const ControlNoiseConfig &noise,
    const ByteView &model_params, const ByteView &cost_params,
    const SimulationResults &results, std::string *error) {
  if (!impl_->initialized) {
    set_error(error, "MetalBackend is not initialized.");
    return false;
  }

  const uint32_t S = config.sample_count;
  const uint32_t H = config.horizon;
  const uint32_t sdim = config.state_dim;
  const uint32_t cdim = config.control_dim;
  const size_t seq_len = static_cast<size_t>(H) * cdim;

  impl_->ensure_scratchpad(config, num_steps);

  std::memcpy(impl_->u_nom_buf.contents, u_nominal, seq_len * sizeof(float));
  std::memcpy(impl_->sigma_buf.contents, noise.sigma, cdim * sizeof(float));
  std::memcpy(impl_->umin_buf.contents, config.bounds.u_min,
              cdim * sizeof(float));
  std::memcpy(impl_->umax_buf.contents, config.bounds.u_max,
              cdim * sizeof(float));
  std::memcpy(impl_->x0_buf.contents, x0, sdim * sizeof(float));

  uint32_t seed = impl_->rng_seed;
  float lambda = config.lambda;

  auto resolve_param_buf = [&](const ByteView &bv,
                               id<MTLBuffer> bound) -> id<MTLBuffer> {
    if (bv.bytes > 0 && bv.data != nullptr)
      return [impl_->device newBufferWithBytes:bv.data
                                        length:bv.bytes
                                       options:MTLResourceStorageModeShared];
    if (bound != nil)
      return bound;
    uint8_t z = 0;
    return [impl_->device newBufferWithBytes:&z
                                      length:1
                                     options:MTLResourceStorageModeShared];
  };
  id<MTLBuffer> mp_buf =
      resolve_param_buf(model_params, impl_->model_params_buffer);
  id<MTLBuffer> cp_buf =
      resolve_param_buf(cost_params, impl_->cost_params_buffer);

  id<MTLBuffer> dummy =
      [impl_->device newBufferWithLength:1
                                 options:MTLResourceStorageModeShared];
  id<MTLBuffer> his_state = impl_->his_state_buf ? impl_->his_state_buf : dummy;
  id<MTLBuffer> his_ctrl = impl_->his_ctrl_buf ? impl_->his_ctrl_buf : dummy;
  id<MTLBuffer> his_cost = impl_->his_cost_buf ? impl_->his_cost_buf : dummy;

  id<MTLCommandBuffer> cmd_buf = [impl_->command_queue commandBuffer];

  id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
  for (uint32_t step = 0; step < num_steps; step++) {
    uint32_t step_idx32 = static_cast<uint32_t>(start_step_index + step);

    // Pass 1: Rollout
    // id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
    [enc setComputePipelineState:impl_->rollout_pipeline];
    [enc setBuffer:impl_->x0_buf offset:0 atIndex:0];
    [enc setBuffer:impl_->u_nom_buf offset:0 atIndex:1];
    [enc setBuffer:mp_buf offset:0 atIndex:2];
    [enc setBuffer:cp_buf offset:0 atIndex:3];
    [enc setBuffer:impl_->costs_buf offset:0 atIndex:4];
    [enc setBytes:&sdim length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&cdim length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:7];
    [enc setBytes:&S length:sizeof(uint32_t) atIndex:8];
    [enc setBuffer:impl_->umin_buf offset:0 atIndex:9];
    [enc setBuffer:impl_->umax_buf offset:0 atIndex:10];
    [enc setVisibleFunctionTable:impl_->dynamics_table atBufferIndex:11];
    [enc setVisibleFunctionTable:impl_->stage_cost_table atBufferIndex:12];
    [enc setVisibleFunctionTable:impl_->terminal_cost_table atBufferIndex:13];
    [enc setBuffer:impl_->sigma_buf offset:0 atIndex:14];
    [enc setBytes:&seed length:sizeof(uint32_t) atIndex:15];
    [enc setBytes:&step_idx32 length:sizeof(uint32_t) atIndex:16];
    [enc dispatchThreads:MTLSizeMake(S, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(
                                  std::min<NSUInteger>(
                                      impl_->rollout_pipeline
                                          .maxTotalThreadsPerThreadgroup,
                                      256),
                                  1, 1)];
    // [enc endEncoding];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Pass 2: Weights
    // enc = [cmd_buf computeCommandEncoder];
    [enc setComputePipelineState:impl_->weights_pipeline];
    [enc setBuffer:impl_->costs_buf offset:0 atIndex:0];
    [enc setBuffer:impl_->weights_buf offset:0 atIndex:1];
    [enc setBuffer:impl_->diagnostics_buf offset:0 atIndex:2];
    [enc setBytes:&S length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&lambda length:sizeof(float) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(
                                  std::min<NSUInteger>(
                                      impl_->weights_pipeline
                                          .maxTotalThreadsPerThreadgroup,
                                      1024),
                                  1, 1)];
    // [enc endEncoding];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Pass 3: Reduce
    // enc = [cmd_buf computeCommandEncoder];
    [enc setComputePipelineState:impl_->reduce_pipeline];
    [enc setBuffer:impl_->weights_buf offset:0 atIndex:0];
    [enc setBuffer:impl_->u_nom_buf offset:0 atIndex:1];
    [enc setBuffer:impl_->u_out_buf offset:0 atIndex:2];
    [enc setBuffer:impl_->sigma_buf offset:0 atIndex:3];
    [enc setBuffer:impl_->umin_buf offset:0 atIndex:4];
    [enc setBuffer:impl_->umax_buf offset:0 atIndex:5];
    [enc setBytes:&S length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:7];
    [enc setBytes:&cdim length:sizeof(uint32_t) atIndex:8];
    [enc setBytes:&seed length:sizeof(uint32_t) atIndex:9];
    [enc setBytes:&step_idx32 length:sizeof(uint32_t) atIndex:10];
    [enc dispatchThreads:MTLSizeMake(seq_len, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(
                                  std::min<NSUInteger>(
                                      impl_->reduce_pipeline
                                          .maxTotalThreadsPerThreadgroup,
                                      256),
                                  1, 1)];
    // [enc endEncoding];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Pass 4: Propagate & Shift
    // enc = [cmd_buf computeCommandEncoder];
    [enc setComputePipelineState:impl_->propagate_pipeline];
    [enc setBuffer:impl_->x0_buf offset:0 atIndex:0];
    [enc setBuffer:impl_->u_nom_buf offset:0 atIndex:1];
    [enc setBuffer:impl_->u_out_buf offset:0 atIndex:2];
    [enc setBuffer:his_state offset:0 atIndex:3];
    [enc setBuffer:his_ctrl offset:0 atIndex:4];
    [enc setBuffer:his_cost offset:0 atIndex:5];
    [enc setBuffer:impl_->diagnostics_buf offset:0 atIndex:6];
    [enc setBuffer:mp_buf offset:0 atIndex:7];
    [enc setBytes:&sdim length:sizeof(uint32_t) atIndex:8];
    [enc setBytes:&cdim length:sizeof(uint32_t) atIndex:9];
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:10];
    [enc setBytes:&step_idx32 length:sizeof(uint32_t) atIndex:11];
    [enc setVisibleFunctionTable:impl_->propagate_dynamics_table
                   atBufferIndex:12];
    [enc dispatchThreads:MTLSizeMake(seq_len, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(
                                  std::min<NSUInteger>(
                                      impl_->propagate_pipeline
                                          .maxTotalThreadsPerThreadgroup,
                                      256),
                                  1, 1)];
    // [enc endEncoding];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }

  [enc endEncoding];
  [cmd_buf commit];
  [cmd_buf waitUntilCompleted];

  if (cmd_buf.status == MTLCommandBufferStatusError) {
    set_error(error, "Simulation Command buffer failed: " +
                         to_std_string(cmd_buf.error.localizedDescription));
    return false;
  }

  // Read back history
  if (results.states_out)
    std::memcpy(results.states_out, impl_->his_state_buf.contents,
                num_steps * sdim * sizeof(float));
  if (results.controls_out)
    std::memcpy(results.controls_out, impl_->his_ctrl_buf.contents,
                num_steps * cdim * sizeof(float));
  if (results.costs_out)
    std::memcpy(results.costs_out, impl_->his_cost_buf.contents,
                num_steps * sizeof(float));

  return true;
}

// ---------------------------------------------------------------------------
// dispatch_batch_simulation
// ---------------------------------------------------------------------------

bool MetalBackend::dispatch_batch_simulation(
    uint32_t num_agents, uint32_t num_steps, const DriverConfig &config,
    uint64_t start_step_index, const float *x0_packed,
    const float *u_nominal_packed, const ControlNoiseConfig &noise,
    const ByteView &model_params, const ByteView &cost_params,
    const BatchSimulationResults &results, std::string *error) {
  if (!impl_->initialized) {
    set_error(error, "MetalBackend is not initialized.");
    return false;
  }

  const uint32_t N = num_agents;
  const uint32_t S = config.sample_count;
  const uint32_t H = config.horizon;
  const uint32_t sdim = config.state_dim;
  const uint32_t cdim = config.control_dim;
  const size_t seq_len = static_cast<size_t>(H) * cdim;

  const bool need_state_history = (results.states_out != nullptr);
  const bool need_control_history = (results.controls_out != nullptr);
  const bool need_cost_history = (results.costs_out != nullptr);
  const bool need_terminal_states = (results.terminal_states_out != nullptr);

  impl_->ensure_scratchpad(config, 0);
  impl_->ensure_batch_scratchpad(config, N, num_steps);

  // Upload packed data into batch buffers.
  std::memcpy(impl_->batch_x0_buf.contents, x0_packed,
              N * sdim * sizeof(float));
  std::memcpy(impl_->batch_u_nom_buf.contents, u_nominal_packed,
              N * seq_len * sizeof(float));
  std::memcpy(impl_->sigma_buf.contents, noise.sigma, cdim * sizeof(float));
  std::memcpy(impl_->umin_buf.contents, config.bounds.u_min,
              cdim * sizeof(float));
  std::memcpy(impl_->umax_buf.contents, config.bounds.u_max,
              cdim * sizeof(float));

  uint32_t seed = impl_->rng_seed;
  float lambda = config.lambda;

  auto resolve_param_buf = [&](const ByteView &bv,
                               id<MTLBuffer> bound) -> id<MTLBuffer> {
    if (bv.bytes > 0 && bv.data != nullptr)
      return [impl_->device newBufferWithBytes:bv.data
                                        length:bv.bytes
                                       options:MTLResourceStorageModeShared];
    if (bound != nil)
      return bound;
    uint8_t z = 0;
    return [impl_->device newBufferWithBytes:&z
                                      length:1
                                     options:MTLResourceStorageModeShared];
  };
  id<MTLBuffer> mp_buf =
      resolve_param_buf(model_params, impl_->model_params_buffer);
  id<MTLBuffer> cp_buf =
      resolve_param_buf(cost_params, impl_->cost_params_buffer);

  id<MTLCommandBuffer> cmd_buf = [impl_->command_queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

  NSUInteger rollout_tpg = std::min<NSUInteger>(
      impl_->rollout_batch_pipeline.maxTotalThreadsPerThreadgroup, 256);
  NSUInteger weights_tpg = std::min<NSUInteger>(
      impl_->weights_batch_pipeline.maxTotalThreadsPerThreadgroup, 1024);
  NSUInteger reduce_tpg = std::min<NSUInteger>(
      impl_->reduce_batch_pipeline.maxTotalThreadsPerThreadgroup, 256);
  NSUInteger propagate_tpg = std::min<NSUInteger>(
      impl_->propagate_batch_pipeline.maxTotalThreadsPerThreadgroup, 256);

  for (uint32_t step = 0; step < num_steps; step++) {
    uint32_t step_idx32 = static_cast<uint32_t>(start_step_index + step);

    // ---- Pass 1: Batch Rollout ----
    [enc setComputePipelineState:impl_->rollout_batch_pipeline];
    [enc setBuffer:impl_->batch_x0_buf offset:0 atIndex:0];
    [enc setBuffer:impl_->batch_u_nom_buf offset:0 atIndex:1];
    [enc setBuffer:mp_buf offset:0 atIndex:2];
    [enc setBuffer:cp_buf offset:0 atIndex:3];
    [enc setBuffer:impl_->batch_costs_buf offset:0 atIndex:4];
    [enc setBytes:&sdim length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&cdim length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:7];
    [enc setBytes:&S length:sizeof(uint32_t) atIndex:8];
    [enc setBuffer:impl_->umin_buf offset:0 atIndex:9];
    [enc setBuffer:impl_->umax_buf offset:0 atIndex:10];
    [enc setVisibleFunctionTable:impl_->batch_dynamics_table atBufferIndex:11];
    [enc setVisibleFunctionTable:impl_->batch_stage_cost_table
                   atBufferIndex:12];
    [enc setVisibleFunctionTable:impl_->batch_terminal_cost_table
                   atBufferIndex:13];
    [enc setBuffer:impl_->sigma_buf offset:0 atIndex:14];
    [enc setBytes:&seed length:sizeof(uint32_t) atIndex:15];
    [enc setBytes:&step_idx32 length:sizeof(uint32_t) atIndex:16];
    [enc setBytes:&N length:sizeof(uint32_t) atIndex:17];
    [enc dispatchThreads:MTLSizeMake(S, N, 1)
        threadsPerThreadgroup:MTLSizeMake(rollout_tpg, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // ---- Pass 2: Batch Weights ----
    [enc setComputePipelineState:impl_->weights_batch_pipeline];
    [enc setBuffer:impl_->batch_costs_buf offset:0 atIndex:0];
    [enc setBuffer:impl_->batch_weights_buf offset:0 atIndex:1];
    [enc setBuffer:impl_->batch_diag_buf offset:0 atIndex:2];
    [enc setBytes:&S length:sizeof(uint32_t) atIndex:3];
    [enc setBytes:&lambda length:sizeof(float) atIndex:4];
    [enc setBytes:&N length:sizeof(uint32_t) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(weights_tpg, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // ---- Pass 3: Batch Reduce ----
    [enc setComputePipelineState:impl_->reduce_batch_pipeline];
    [enc setBuffer:impl_->batch_weights_buf offset:0 atIndex:0];
    [enc setBuffer:impl_->batch_u_nom_buf offset:0 atIndex:1];
    [enc setBuffer:impl_->batch_u_out_buf offset:0 atIndex:2];
    [enc setBuffer:impl_->sigma_buf offset:0 atIndex:3];
    [enc setBuffer:impl_->umin_buf offset:0 atIndex:4];
    [enc setBuffer:impl_->umax_buf offset:0 atIndex:5];
    [enc setBytes:&S length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:7];
    [enc setBytes:&cdim length:sizeof(uint32_t) atIndex:8];
    [enc setBytes:&seed length:sizeof(uint32_t) atIndex:9];
    [enc setBytes:&step_idx32 length:sizeof(uint32_t) atIndex:10];
    [enc setBytes:&N length:sizeof(uint32_t) atIndex:11];
    [enc dispatchThreads:MTLSizeMake(seq_len, N, 1)
        threadsPerThreadgroup:MTLSizeMake(reduce_tpg, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // ---- Pass 4: Batch Propagate & Shift ----
    [enc setComputePipelineState:impl_->propagate_batch_pipeline];
    [enc setBuffer:impl_->batch_x0_buf offset:0 atIndex:0];
    [enc setBuffer:impl_->batch_u_nom_buf offset:0 atIndex:1];
    [enc setBuffer:impl_->batch_u_out_buf offset:0 atIndex:2];
    [enc setBuffer:(need_state_history ? impl_->batch_his_state_buf : nil)
            offset:0
           atIndex:3];
    [enc setBuffer:(need_control_history ? impl_->batch_his_ctrl_buf : nil)
            offset:0
           atIndex:4];
    [enc setBuffer:(need_cost_history ? impl_->batch_his_cost_buf : nil)
            offset:0
           atIndex:5];
    [enc setBuffer:impl_->batch_diag_buf offset:0 atIndex:6];
    [enc setBuffer:mp_buf offset:0 atIndex:7];
    [enc setBytes:&sdim length:sizeof(uint32_t) atIndex:8];
    [enc setBytes:&cdim length:sizeof(uint32_t) atIndex:9];
    [enc setBytes:&H length:sizeof(uint32_t) atIndex:10];
    [enc setBytes:&step_idx32 length:sizeof(uint32_t) atIndex:11];
    [enc setVisibleFunctionTable:impl_->batch_propagate_dynamics_table
                   atBufferIndex:12];
    [enc setBytes:&N length:sizeof(uint32_t) atIndex:13];
    [enc setBytes:&num_steps length:sizeof(uint32_t) atIndex:14];
    [enc
        setBuffer:(need_terminal_states ? impl_->batch_terminal_state_buf : nil)
           offset:0
          atIndex:15];
    [enc dispatchThreads:MTLSizeMake(seq_len, N, 1)
        threadsPerThreadgroup:MTLSizeMake(propagate_tpg, 1, 1)];
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }

  [enc endEncoding];
  [cmd_buf commit];
  [cmd_buf waitUntilCompleted];

  if (cmd_buf.status == MTLCommandBufferStatusError) {
    set_error(error, "Batch simulation command buffer failed: " +
                         to_std_string(cmd_buf.error.localizedDescription));
    return false;
  }

  // Read back batch history.
  if (results.states_out)
    std::memcpy(results.states_out, impl_->batch_his_state_buf.contents,
                N * num_steps * sdim * sizeof(float));
  if (results.controls_out)
    std::memcpy(results.controls_out, impl_->batch_his_ctrl_buf.contents,
                N * num_steps * cdim * sizeof(float));
  if (results.costs_out)
    std::memcpy(results.costs_out, impl_->batch_his_cost_buf.contents,
                N * num_steps * sizeof(float));
  if (results.terminal_states_out && num_steps > 0) {
    std::memcpy(results.terminal_states_out,
                impl_->batch_terminal_state_buf.contents,
                N * sdim * sizeof(float));
  }

  return true;
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

bool MetalBackend::reset(std::string *error) { return true; }

} // namespace mppi_metal
