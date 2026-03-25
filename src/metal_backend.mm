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
  id<MTLComputePipelineState> pipeline = nil;
  id<MTLLibrary> library_metallib = nil;
  id<MTLLibrary> user_metallib = nil;
  id<MTLVisibleFunctionTable> dynamics_table = nil;
  id<MTLVisibleFunctionTable> stage_cost_table = nil;
  id<MTLVisibleFunctionTable> terminal_cost_table = nil;
  id<MTLBuffer> model_params_buffer = nil;
  id<MTLBuffer> cost_params_buffer = nil;
  NSUInteger max_threads_per_threadgroup = 0;
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
  NSError *lib_error = nil;
  impl_->library_metallib =
      [impl_->device newLibraryWithURL:[NSURL fileURLWithPath:lib_path]
                                 error:&lib_error];
  if (impl_->library_metallib == nil) {
    set_error(error, "Failed to load library metallib: " +
              to_std_string(lib_error.localizedDescription));
    return false;
  }

  id<MTLFunction> kernel_fn =
      [impl_->library_metallib newFunctionWithName:@"mppi_rollout"];
  if (kernel_fn == nil) {
    set_error(error, "Required library symbol 'mppi_rollout' missing.");
    return false;
  }

  std::string user_path_str = model.user_metallib_path.string();
  if (user_path_str.empty()) {
    set_error(error, "user_metallib_path is empty.");
    return false;
  }
  NSError *user_error = nil;
  impl_->user_metallib =
      [impl_->device newLibraryWithURL:
          [NSURL fileURLWithPath:
              [NSString stringWithUTF8String:user_path_str.c_str()]]
                                 error:&user_error];
  if (impl_->user_metallib == nil) {
    set_error(error, "Failed to load user metallib '" + user_path_str +
              "': " + to_std_string(user_error.localizedDescription));
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

  // Link all three callables.
  MTLLinkedFunctions *linked = [[MTLLinkedFunctions alloc] init];
  linked.functions = @[dynamics_fn, stage_cost_fn, terminal_cost_fn];

  MTLComputePipelineDescriptor *pipe_desc =
      [[MTLComputePipelineDescriptor alloc] init];
  pipe_desc.computeFunction = kernel_fn;
  pipe_desc.linkedFunctions = linked;

  NSError *pipeline_error = nil;
  impl_->pipeline =
      [impl_->device newComputePipelineStateWithDescriptor:pipe_desc
                                                  options:0
                                               reflection:nil
                                                    error:&pipeline_error];
  if (impl_->pipeline == nil) {
    set_error(error, "Failed to create linked compute pipeline: " +
              to_std_string(pipeline_error.localizedDescription));
    return false;
  }

  // Create visible function tables (1 entry each).
  auto make_table = [&](id<MTLFunction> fn) -> id<MTLVisibleFunctionTable> {
    MTLVisibleFunctionTableDescriptor *td =
        [[MTLVisibleFunctionTableDescriptor alloc] init];
    td.functionCount = 1;
    id<MTLVisibleFunctionTable> table =
        [impl_->pipeline newVisibleFunctionTableWithDescriptor:td];
    if (table != nil) {
      id<MTLFunctionHandle> handle =
          [impl_->pipeline functionHandleWithFunction:fn];
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

  impl_->max_threads_per_threadgroup =
      impl_->pipeline.maxTotalThreadsPerThreadgroup;
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
// dispatch_rollout
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
  const size_t total_noise = static_cast<size_t>(S) * seq_len;

  // ---- GPU buffers ----
  id<MTLBuffer> x0_buf =
      [impl_->device newBufferWithBytes:x0
                                length:sdim * sizeof(float)
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> u_nom_buf =
      [impl_->device newBufferWithBytes:u_nominal
                                length:seq_len * sizeof(float)
                               options:MTLResourceStorageModeShared];

  // Noise output buffer (kernel writes, host reads back for MPPI update).
  id<MTLBuffer> noise_buf =
      [impl_->device newBufferWithLength:total_noise * sizeof(float)
                                options:MTLResourceStorageModeShared];

  // Param buffers.
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

  // Bounds.
  id<MTLBuffer> umin_buf =
      [impl_->device newBufferWithBytes:config.bounds.u_min
                                length:cdim * sizeof(float)
                               options:MTLResourceStorageModeShared];
  id<MTLBuffer> umax_buf =
      [impl_->device newBufferWithBytes:config.bounds.u_max
                                length:cdim * sizeof(float)
                               options:MTLResourceStorageModeShared];

  // Sigma (noise covariance diagonal).
  id<MTLBuffer> sigma_buf =
      [impl_->device newBufferWithBytes:noise.sigma
                                length:cdim * sizeof(float)
                               options:MTLResourceStorageModeShared];

  // Scalar constants.
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
  uint32_t seed = impl_->rng_seed;
  id<MTLBuffer> seed_buf =
      [impl_->device newBufferWithBytes:&seed length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
  uint32_t step_idx32 = static_cast<uint32_t>(step_index);
  id<MTLBuffer> step_buf =
      [impl_->device newBufferWithBytes:&step_idx32 length:sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];

  // ---- Encode and dispatch ----
  id<MTLCommandBuffer> cmd_buf = [impl_->command_queue commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

  [enc setComputePipelineState:impl_->pipeline];
  [enc setBuffer:x0_buf      offset:0 atIndex:0];
  [enc setBuffer:u_nom_buf   offset:0 atIndex:1];
  [enc setBuffer:noise_buf   offset:0 atIndex:2];
  [enc setBuffer:mp_buf      offset:0 atIndex:3];
  [enc setBuffer:cp_buf      offset:0 atIndex:4];
  [enc setBuffer:costs_buf   offset:0 atIndex:5];
  [enc setBuffer:sdim_buf    offset:0 atIndex:6];
  [enc setBuffer:cdim_buf    offset:0 atIndex:7];
  [enc setBuffer:horizon_buf offset:0 atIndex:8];
  [enc setBuffer:sample_buf  offset:0 atIndex:9];
  [enc setBuffer:umin_buf    offset:0 atIndex:10];
  [enc setBuffer:umax_buf    offset:0 atIndex:11];
  [enc setVisibleFunctionTable:impl_->dynamics_table      atBufferIndex:12];
  [enc setVisibleFunctionTable:impl_->stage_cost_table    atBufferIndex:13];
  [enc setVisibleFunctionTable:impl_->terminal_cost_table atBufferIndex:14];
  [enc setBuffer:sigma_buf   offset:0 atIndex:15];
  [enc setBuffer:seed_buf    offset:0 atIndex:16];
  [enc setBuffer:step_buf    offset:0 atIndex:17];

  NSUInteger tpg = std::min<NSUInteger>(impl_->max_threads_per_threadgroup, 256);
  [enc dispatchThreads:MTLSizeMake(S, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
  [enc endEncoding];

  [cmd_buf commit];
  [cmd_buf waitUntilCompleted];

  if (cmd_buf.status == MTLCommandBufferStatusError) {
    set_error(error, "Metal command buffer failed: " +
              to_std_string(cmd_buf.error.localizedDescription));
    return false;
  }

  // ---- MPPI update on host ----
  const float *costs = static_cast<const float *>(costs_buf.contents);
  const float *noise_host = static_cast<const float *>(noise_buf.contents);

  // Min cost for numerical stability.
  float min_cost = costs[0];
  for (uint32_t k = 1; k < S; ++k)
    if (costs[k] < min_cost) min_cost = costs[k];

  // Weights: w_k = exp(-(cost_k - min_cost) / lambda).
  const float inv_lambda = 1.0f / config.lambda;
  std::vector<float> weights(S);
  float weight_sum = 0.0f;
  for (uint32_t k = 0; k < S; ++k) {
    weights[k] = std::exp(-(costs[k] - min_cost) * inv_lambda);
    weight_sum += weights[k];
  }
  for (uint32_t k = 0; k < S; ++k) weights[k] /= weight_sum;

  // Updated controls: u*_i = u_nominal_i + Σ_k w_k * noise_{k,i}.
  for (size_t i = 0; i < seq_len; ++i) {
    float wn = 0.0f;
    for (uint32_t k = 0; k < S; ++k)
      wn += weights[k] * noise_host[k * seq_len + i];
    u_out[i] = u_nominal[i] + wn;
  }

  // Clamp to bounds.
  for (uint32_t t = 0; t < H; ++t)
    for (uint32_t d = 0; d < cdim; ++d) {
      size_t idx = t * cdim + d;
      u_out[idx] = std::max(config.bounds.u_min[d],
                   std::min(config.bounds.u_max[d], u_out[idx]));
    }

  // Diagnostics.
  if (diagnostics != nullptr) {
    diagnostics->best_cost = min_cost;
    float sum = 0.0f;
    for (uint32_t k = 0; k < S; ++k) sum += costs[k];
    diagnostics->mean_cost = sum / static_cast<float>(S);
    float w2 = 0.0f;
    for (uint32_t k = 0; k < S; ++k) w2 += weights[k] * weights[k];
    diagnostics->effective_sample_size = 1.0f / w2;
    diagnostics->converged = false;
  }

  return true;
}

// ---------------------------------------------------------------------------
// reset
// ---------------------------------------------------------------------------

bool MetalBackend::reset(std::string *error) {
  return true;
}

} // namespace mppi_metal
