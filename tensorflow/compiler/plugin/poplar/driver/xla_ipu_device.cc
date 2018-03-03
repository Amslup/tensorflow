/* Copyright 2017 Graphcore Ltd
 */

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/jit/kernels/xla_launch_op.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/tf2xla/kernels/index_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform.h"

#include "tensorflow/core/kernels/no_op.h"

namespace se = ::perftools::gputools;
namespace sep = ::perftools::gputools::poplarplugin;

namespace tensorflow {

const char* const DEVICE_XLA_IPU = "IPU";
const char* const DEVICE_IPU_XLA_JIT = "XLA_IPU_JIT";
const char* const PLATFORM_NAME = "Poplar";

constexpr std::array<DataType, 6> kIpuAllTypes =
        {{DT_INT32, DT_INT64, DT_FLOAT, DT_HALF, DT_BOOL, DT_RESOURCE}};

class IpuDevice : public XlaDevice {
 public:
  IpuDevice(const SessionOptions& options,
            const DeviceAttributes& attrs, int device_ordinal,
            const DeviceType& jit_device_name, se::Platform* platform) :
      XlaDevice(options, attrs, device_ordinal, jit_device_name, platform),
      ordinal_(device_ordinal) {}

  virtual ~IpuDevice() {
    auto platform = se::MultiPlatformManager::PlatformWithName(PLATFORM_NAME);
    if (!platform.ok()) {
      return;
    }
    auto* p = static_cast<sep::PoplarPlatform*>(platform.ValueOrDie());
    p->ClosePoplarDevice(ordinal_);
  }

 private:
  int ordinal_;
};

class XlaIpuDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override;
};

Status XlaIpuDeviceFactory::CreateDevices(const SessionOptions& options,
                                          const string& name_prefix,
                                          std::vector<Device*>* devices) {
  static XlaDeviceOpRegistrations* registrations =
      RegisterXlaDeviceKernels(DEVICE_XLA_IPU, DEVICE_IPU_XLA_JIT);
  (void)registrations;

  auto platform = se::MultiPlatformManager::PlatformWithName(PLATFORM_NAME);
  if (!platform.ok()) {
    return StreamExecutorUtil::ConvertStatus(platform.status());
  }

  auto* p = static_cast<sep::PoplarPlatform*>(platform.ValueOrDie());
  TF_RETURN_IF_ERROR(p->ConfigurePoplarDevices(options.config.ipu_options()));

  int visible_devices = p->VisibleDeviceCount();
  for (int ordinal=0; ordinal<visible_devices; ordinal++) {

    XlaOpRegistry::DeviceRegistration registration;
    registration.compilation_device_name = DEVICE_IPU_XLA_JIT;
    registration.requires_compilation = true;
    registration.enable_jit_by_default = false;
    registration.compile_resource_ops = true;
    XlaOpRegistry::RegisterCompilationDevice(DEVICE_XLA_IPU, registration);

    const DeviceAttributes attrs = Device::BuildDeviceAttributes(
        strings::StrCat(name_prefix, "/device:IPU:", ordinal),
        DeviceType(DEVICE_XLA_IPU), Bytes(16ULL << 30), DeviceLocality(),
        "IPU Device");

    auto* device = new IpuDevice(options, attrs, ordinal,
                                 DeviceType(DEVICE_IPU_XLA_JIT), p);

    devices->push_back(device);
  }

  return Status::OK();
}

REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_XLA_IPU, XlaIpuDeviceFactory);

// Kernel registrations

static bool OpFilter(KernelDef* kdef) {
  // TODO - probably remove int32/bool for some set of operators
  // (or keep them for some given set)
  return true;
}

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_XLA_IPU, XlaLocalLaunchOp, kIpuAllTypes);
REGISTER_XLA_DEVICE_KERNELS(DEVICE_XLA_IPU, kIpuAllTypes);
REGISTER_XLA_BACKEND(DEVICE_IPU_XLA_JIT, kIpuAllTypes, OpFilter);

// Additional ops not explicitly defined by standard JIT
REGISTER_XLA_OP(Name("ArgMax")
                  .Device(DEVICE_IPU_XLA_JIT)
                  .CompileTimeConstInput("dimension"), XlaArgMaxOp);

REGISTER_XLA_OP(Name("Enter").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_XLA_OP(Name("RefEnter").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_XLA_OP(Name("Exit").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_XLA_OP(Name("RefExit").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_XLA_OP(Name("LoopCond").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_XLA_OP(Name("Merge").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_XLA_OP(Name("RefMerge").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_XLA_OP(Name("NextIteration").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_XLA_OP(Name("RefNextIteration").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_XLA_OP(Name("Switch").Device(DEVICE_IPU_XLA_JIT), NoOp);
REGISTER_XLA_OP(Name("RefSwitch").Device(DEVICE_IPU_XLA_JIT), NoOp);

}  // namespace tensorflow
