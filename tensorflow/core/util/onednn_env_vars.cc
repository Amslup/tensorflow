/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifdef INTEL_MKL

#include "tensorflow/core/util/onednn_env_vars.h"
#include "absl/base/call_once.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

bool AreWeightsFrozen() {
  static bool weights_const = false;
  static absl::once_flag once;
  absl::call_once(once, [&] {
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ONEDNN_ASSUME_FROZEN_WEIGHTS",
                                   /*default_value*/ false, &weights_const));
  });
  return weights_const;
}

bool UseSystemAlloc() {
  static bool use_sys_alloc = false;
  static absl::once_flag once;
  absl::call_once(once, [&] {
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ONEDNN_USE_SYSTEM_ALLOCATOR",
                                   /*default_value*/ false, &use_sys_alloc));
  });
  return use_sys_alloc;
}

bool IsMKLEnabled() {
#ifndef INTEL_MKL
  return false;
#endif  // !INTEL_MKL
  static absl::once_flag once;
#ifdef ENABLE_MKL
  // Keeping TF_DISABLE_MKL env variable for legacy reasons.
  static bool oneDNN_disabled = false;
  absl::call_once(once, [&] {
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_DISABLE_MKL", false, &oneDNN_disabled));
    if (oneDNN_disabled) VLOG(2) << "TF-MKL: Disabling oneDNN";
  });
  return (!oneDNN_disabled);
#else
  // Linux: Turn oneDNN on by default for CPUs with neural network features.
  // Windows: oneDNN is off by default.
  // No need to guard for other platforms here because INTEL_MKL is only defined
  // for non-mobile Linux or Windows.
  static bool oneDNN_enabled =
#ifdef __linux__
      port::TestCPUFeature(port::CPUFeature::AVX512_VNNI) ||
      port::TestCPUFeature(port::CPUFeature::AVX512_BF16) ||
      port::TestCPUFeature(port::CPUFeature::AVX_VNNI) ||
      port::TestCPUFeature(port::CPUFeature::AMX_TILE) ||
      port::TestCPUFeature(port::CPUFeature::AMX_INT8) ||
      port::TestCPUFeature(port::CPUFeature::AMX_BF16);
#else
      false;
#endif  // __linux__
  absl::call_once(once, [&] {
    auto status = ReadBoolFromEnvVar("TF_ENABLE_ONEDNN_OPTS", oneDNN_enabled,
                                     &oneDNN_enabled);
    if (!status.ok()) {
      LOG(WARNING) << "TF_ENABLE_ONEDNN_OPTS is not set to either '0', 'false',"
                   << " '1', or 'true'. Using the default setting: "
                   << oneDNN_enabled;
    }
    if (oneDNN_enabled) {
      LOG(INFO) << "oneDNN custom operations are on. "
                << "You may see slightly different numerical results due to "
                << "floating-point round-off errors from different computation "
                << "orders. To turn them off, set the environment variable "
                << "`TF_ENABLE_ONEDNN_OPTS=0`.";
    }
  });
  return oneDNN_enabled;
#endif  // ENABLE_MKL
}

}  // namespace tensorflow
#endif  // INTEL_MKL
