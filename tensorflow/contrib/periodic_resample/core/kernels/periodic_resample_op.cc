// =============================================================================
// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/contrib/periodic_resample/core/kernels/periodic_resample_op.h"

namespace tensorflow {

REGISTER_KERNEL_BUILDER(Name("PeriodicResample")
                            .Device(DEVICE_CPU)
                            .HostMemory("desired_shape"),
                        PeriodicResampleOp);

// #define REGISTER_GPU_KERNEL(type)                          \
//   REGISTER_KERNEL_BUILDER(Name("PeriodicResample")         \
//                               .Device(DEVICE_GPU)          \
//                               .HostMemory("desired_shape") \
//                               .TypeConstraint<type>("T"),  \
//                           PeriodicResampleOp);
// TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNEL);
// #undef REGISTER_GPU_KERNEL

// #ifdef TENSORFLOW_USE_SYCL
// REGISTER_KERNEL_BUILDER(Name("PeriodicResample")
//                             .Device(DEVICE_SYCL)
//                             .HostMemory("desired_shape"),
//                         PeriodicResampleOp);
// #endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow
