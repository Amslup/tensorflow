/* Copyright 2019 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Defines the GpuStream type - the CUDA-specific implementation of the generic
// StreamExecutor Stream interface.

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_STREAM_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_STREAM_H_

#include <optional>
#include <variant>

#include "absl/log/check.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
namespace gpu {

// Wraps a GpuStreamHandle in order to satisfy the platform-independent
// StreamInterface.
//
// Thread-safe post-initialization.
class GpuStream : public StreamCommon {
 public:
  GpuStream(StreamExecutor* parent,
            std::optional<std::variant<StreamPriority, int>> priority)
      : StreamCommon(parent) {
    if (priority.has_value()) {
      stream_priority_ = priority.value();
    }
  }

  std::variant<StreamPriority, int> priority() const override {
    return stream_priority_;
  }

  // Returns true if no work is pending or executing on the stream.
  bool IsIdle() const;

  // Returns the GpuStreamHandle value for passing to the CUDA API.
  //
  // Precond: this GpuStream has been allocated (otherwise passing a nullptr
  // into the NVIDIA library causes difficult-to-understand faults).
  GpuStreamHandle gpu_stream() const {
    DCHECK(gpu_stream_ != nullptr);
    return gpu_stream_;
  }

  absl::Status WaitFor(Stream* other) override;
  absl::Status WaitFor(Event* event) override;
  absl::Status RecordEvent(Event* event) override;
  absl::Status MemZero(DeviceMemoryBase* location, uint64_t size) override;
  absl::Status Memset32(DeviceMemoryBase* location, uint32_t pattern,
                        uint64_t size) override;
  absl::Status Memcpy(DeviceMemoryBase* gpu_dst, const void* host_src,
                      uint64_t size) override;
  absl::Status Memcpy(void* host_dst, const DeviceMemoryBase& gpu_src,
                      uint64_t size) override;
  absl::Status Memcpy(DeviceMemoryBase* gpu_dst,
                      const DeviceMemoryBase& gpu_src, uint64_t size) override;
  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback) override;

  void set_name(absl::string_view name) override;
  absl::StatusOr<std::unique_ptr<EventBasedTimer>> CreateEventBasedTimer(
      bool use_delay_kernel) override;
  absl::Status Launch(const ThreadDim& thread_dims, const BlockDim& block_dims,
                      const Kernel& k, const KernelArgs& args) override;
  absl::Status Launch(const ThreadDim& thread_dims, const BlockDim& block_dims,
                      const ClusterDim& cluster_dims, const Kernel& k,
                      const KernelArgs& args) override;

 private:
  // Helper method to launch a kernel with optional cluster dimensions.
  virtual absl::Status Launch(const ThreadDim& thread_dims,
                              const BlockDim& block_dims,
                              const std::optional<ClusterDim>& cluster_dims,
                              const Kernel& kernel, const KernelArgs& args) = 0;

  std::variant<StreamPriority, int> stream_priority_;
};

// Helper functions to simplify extremely common flows.
// Converts a Stream to the underlying GpuStream implementation.
GpuStream* AsGpuStream(Stream* stream);

// Extracts a GpuStreamHandle from a GpuStream-backed Stream object.
GpuStreamHandle AsGpuStreamValue(Stream* stream);
}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_STREAM_H_
