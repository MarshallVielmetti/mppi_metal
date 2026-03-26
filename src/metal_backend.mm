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
  if (str == nil) return {};
  return std::string([str UTF8String]);
}

} // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct MetalBackend::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> command_queue = nil;

  // Pass 1: rollout pipeline (linked with user callables).
  id<MTLComputePipelineState> rollout_pipeline = nil;
  id<MTLVisibleFunctionTable> dynamics_table = nil;
  id<MTLVisibleFunctionTable> stage_cost_table = nil;
  id<MTLVisibleFunctionTable> terminal_cost_table = nil;

  // Pass 2: weights pipeline.
  id<MTLComputePipelineState> weights_pipeline = nil;

  // Pass 3: reduction pipeline (no linked functions needed).
  id<MTLComputePipelineState> reduce_pipeline = nil;

  id<MTLBuffer> diagnostics_buf = nil;

  id<MTLLibrary> library_metallib = nil;
  id<MTLLibrary> user_metallib = nil;
  id<MTLBuffer> model_params_buffer = nil;
  id<MTLBuffer> cost_params_buffer = nil;
  NSUInteger rollout_max_tpg = 0;
  NSUInteger reduce_max_tpg = 0;
  uint32_t rng_seed = 0;
  bool initialized = false;
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

  NSString *lib_path = [NSString stringWithUTF8String:MPPI_ROLLOUT_METALLIB_PATH];
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

  // ---- Load user metallib ----
  std::string user_path_str = model.user_metallib_path.string();
  if (user_path_str.empty()) {
    set_error(error, "user_metallib_path is empty.");
    return false;
  }
  impl_->user_metallib =
      [impl_->device newLibraryWithURL:
          [NSURL fileURLWithPath:
              [NSString stringWithUTF8String:user_path_str.c_str()]]
                                 error:&ns_err];
  if (impl_->user_metallib == nil) {
    set_error(error, "Failed to load user metallib '" + user_path_str +
              "': " + to_std_string(ns_err.localizedDescription));
    return false;
  }

  // Dynamics (required).
  std::string dynamics_name(model.callables.dynamics);
  id<MTLFunction> dynamics_fn =
      [impl_->user_metallib newFunctionWithName:
          [NSString stringWithUTF8String:dynamics_name.c_str()]];
  if (dynamics_fn == nil) {
    set_error(error, "Required callable '" + dynamics_name +
              "' not found in user metallib.");
    return false;
  }

  // Stage cost (optional → default).
  id<MTLFunction> stage_cost_fn = nil;
  if (!model.callables.stage_cost.empty()) {
    std::string sc(model.callables.stage_cost);
    stage_cost_fn = [impl_->user_metallib newFunctionWithName:
        [NSString stringWithUTF8String:sc.c_str()]];
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
    terminal_cost_fn = [impl_->user_metallib newFunctionWithName:
        [NSString stringWithUTF8String:tc.c_str()]];
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
  linked.functions = @[dynamics_fn, stage_cost_fn, terminal_cost_fn];

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
      [impl_->device newComputePipelineStateWithFunction:weights_fn error:&ns_err];
  if (impl_->weights_pipeline == nil) {
    set_error(error, "Failed to create weights pipeline: " +
              to_std_string(ns_err.localizedDescription));
    return false;
  }

  // ---- Build reduce pipeline (no linked functions) ----
  impl_->reduce_pipeline =
      [impl_->device newComputePipelineStateWithFunction:reduce_fn error:&ns_err];
  if (impl_->reduce_pipeline == nil) {
    set_error(error, "Failed to create reduce pipeline: " +
              to_std_string(ns_err.localizedDescription));
    return false;
  }

  impl_->diagnostics_buf = [impl_->device newBufferWithLength:3 * sizeof(float)
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
      if (handle != nil) [table setFunction:handle atIndex:0];
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

  impl_->command_queue = [impl_->device newCommandQueue];
  if (impl_->command_queue == nil) {
    set_error(error, "Failed to create command queue.");
    return false;
  }

  impl_->rollout_max_tpg =
      impl_->rollout_pipeline.maxTotalThreadsPerThreadgroup;
  impl_->reduce_max_tpg =
      impl_->reduce_pipeline.maxTotalThreadsPerThreadgroup;
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

bool MetalBackend::dispatch_rollout(const DriverConfig &config,
                                    uint64_t step_index,
                                    const float *x0, const float *u_nominal,
                                    const ControlNoiseConfig &noise,
                                    const ByteView &model_params,
                                    const ByteView &cost_params, float *u_out,
                                    StepDiagnostics *diagnostics,
                                    std::string *error) {
  if (!impl_->initialized) {
    set_error(error, "MetalBackend is not initialized.");
    return false;
  }

  const uint32_t S = config.sample_count;
  const uint32_t H = config.horizon;
  const uint32_t sdim = config.state_dim;
  const uint32_t cdim = config.control_dim;
  const size_t seq_len = static_cast<size_t>(H) * cdim;

  // ---- Shared buffers (used by both passes) ----
  id<MTLBuffer> u_nom_buf =
      [impl_->device newBufferWithBytes:u_nominal
                                length:seq_len * sizeof(float)
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> sigma_buf =
      [impl_->device newBufferWithBytes:noise.sigma
                                length:cdim * sizeof(float)
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> umin_buf =
      [impl_->device newBufferWithBytes:config.bounds.u_min
                                length:cdim * sizeof(float)
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> umax_buf =
      [impl_->device newBufferWithBytes:config.bounds.u_max
                                length:cdim * sizeof(float)
                               options:MTLResourceStorageModeShared];
  uint32_t seed = impl_->rng_seed;
  id<MTLBuffer> seed_buf =
      [impl_->device newBufferWithBytes:&seed length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
  uint32_t step_idx32 = static_cast<uint32_t>(step_index);
  id<MTLBuffer> step_buf =
      [impl_->device newBufferWithBytes:&step_idx32 length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> lambda_buf =
      [impl_->device newBufferWithBytes:&config.lambda length:sizeof(float)
                               options:MTLResourceStorageModeShared];

  // ---- Pass 1: Rollout ----
  id<MTLBuffer> x0_buf =
      [impl_->device newBufferWithBytes:x0
                                length:sdim * sizeof(float)
                               options:MTLResourceStorageModeShared];

  auto resolve_param_buf = [&](const ByteView &bv,
                               id<MTLBuffer> bound) -> id<MTLBuffer> {
    if (bv.bytes > 0 && bv.data != nullptr)
      return [impl_->device newBufferWithBytes:bv.data length:bv.bytes
                                      options:MTLResourceStorageModeShared];
    if (bound != nil) return bound;
    uint8_t z = 0;
    return [impl_->device newBufferWithBytes:&z length:1
                                    options:MTLResourceStorageModeShared];
  };
  id<MTLBuffer> mp_buf = resolve_param_buf(model_params, impl_->model_params_buffer);
  id<MTLBuffer> cp_buf = resolve_param_buf(cost_params, impl_->cost_params_buffer);

  id<MTLBuffer> costs_buf =
      [impl_->device newBufferWithLength:S * sizeof(float)
                                options:MTLResourceStorageModeShared];

  // Scalar constant buffers.
  id<MTLBuffer> sdim_buf =
      [impl_->device newBufferWithBytes:&sdim length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> cdim_buf =
      [impl_->device newBufferWithBytes:&cdim length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> horizon_buf =
      [impl_->device newBufferWithBytes:&H length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> sample_buf =
      [impl_->device newBufferWithBytes:&S length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];

  id<MTLCommandBuffer> cmd_buf = [impl_->command_queue commandBuffer];

  // Encode rollout.
  {
    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
    [enc setComputePipelineState:impl_->rollout_pipeline];
    [enc setBuffer:x0_buf      offset:0 atIndex:0];
    [enc setBuffer:u_nom_buf   offset:0 atIndex:1];
    [enc setBuffer:mp_buf      offset:0 atIndex:2];
    [enc setBuffer:cp_buf      offset:0 atIndex:3];
    [enc setBuffer:costs_buf   offset:0 atIndex:4];
    [enc setBuffer:sdim_buf    offset:0 atIndex:5];
    [enc setBuffer:cdim_buf    offset:0 atIndex:6];
    [enc setBuffer:horizon_buf offset:0 atIndex:7];
    [enc setBuffer:sample_buf  offset:0 atIndex:8];
    [enc setBuffer:umin_buf    offset:0 atIndex:9];
    [enc setBuffer:umax_buf    offset:0 atIndex:10];
    [enc setVisibleFunctionTable:impl_->dynamics_table      atBufferIndex:11];
    [enc setVisibleFunctionTable:impl_->stage_cost_table    atBufferIndex:12];
    [enc setVisibleFunctionTable:impl_->terminal_cost_table atBufferIndex:13];
    [enc setBuffer:sigma_buf   offset:0 atIndex:14];
    [enc setBuffer:seed_buf    offset:0 atIndex:15];
    [enc setBuffer:step_buf    offset:0 atIndex:16];

    NSUInteger tpg = std::min<NSUInteger>(impl_->rollout_max_tpg, 256);
    [enc dispatchThreads:MTLSizeMake(S, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
    [enc endEncoding];
  }

  // ---- Pass 2: Weights ----
  id<MTLBuffer> weights_buf =
      [impl_->device newBufferWithLength:S * sizeof(float)
                                options:MTLResourceStorageModeShared];

  {
    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
    [enc setComputePipelineState:impl_->weights_pipeline];
    [enc setBuffer:costs_buf offset:0 atIndex:0];
    [enc setBuffer:weights_buf offset:0 atIndex:1];
    [enc setBuffer:impl_->diagnostics_buf offset:0 atIndex:2];
    [enc setBuffer:sample_buf offset:0 atIndex:3];
    [enc setBuffer:lambda_buf offset:0 atIndex:4];

    NSUInteger tpg_w = std::min<NSUInteger>(impl_->weights_pipeline.maxTotalThreadsPerThreadgroup, 1024);
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg_w, 1, 1)];
    [enc endEncoding];
  }

  // ---- Pass 3: Reduce ----
  id<MTLBuffer> u_out_buf =
      [impl_->device newBufferWithLength:seq_len * sizeof(float)
                                options:MTLResourceStorageModeShared];

  {
    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];
    [enc setComputePipelineState:impl_->reduce_pipeline];
    [enc setBuffer:weights_buf offset:0 atIndex:0];
    [enc setBuffer:u_nom_buf   offset:0 atIndex:1];
    [enc setBuffer:u_out_buf   offset:0 atIndex:2];
    [enc setBuffer:sigma_buf   offset:0 atIndex:3];
    [enc setBuffer:umin_buf    offset:0 atIndex:4];
    [enc setBuffer:umax_buf    offset:0 atIndex:5];
    [enc setBuffer:sample_buf  offset:0 atIndex:6];
    [enc setBuffer:horizon_buf offset:0 atIndex:7];
    [enc setBuffer:cdim_buf    offset:0 atIndex:8];
    [enc setBuffer:seed_buf    offset:0 atIndex:9];
    [enc setBuffer:step_buf    offset:0 atIndex:10];

    NSUInteger tpg2 = std::min<NSUInteger>(impl_->reduce_max_tpg, 256);
    [enc dispatchThreads:MTLSizeMake(seq_len, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tpg2, 1, 1)];
    [enc endEncoding];
  }

  // Submit all 3 passes and wait
  [cmd_buf commit];
  [cmd_buf waitUntilCompleted];

  if (cmd_buf.status == MTLCommandBufferStatusError) {
    set_error(error, "Command buffer failed: " +
              to_std_string(cmd_buf.error.localizedDescription));
    return false;
  }

  if (diagnostics != nullptr) {
    const float *d = static_cast<const float *>(impl_->diagnostics_buf.contents);
    diagnostics->best_cost = d[0];
    diagnostics->mean_cost = d[1];
    diagnostics->effective_sample_size = d[2];
    diagnostics->converged = false;
  }

  // ---- Read back updated controls ----
  std::memcpy(u_out, u_out_buf.contents, seq_len * sizeof(float));

  return true;
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

bool MetalBackend::reset(std::string *error) {
  return true;
}

} // namespace mppi_metal
