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

#include "tensorflow/compiler/plugin/poplar/driver/transfer_manager.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform_id.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

#include <string>
#include <utility>
#include <vector>

namespace sep = ::perftools::gputools::poplarplugin;

namespace xla {
namespace poplarplugin {

PoplarTransferManager::PoplarTransferManager() {
}

se::Platform::Id PoplarTransferManager::PlatformId() const {
  return se::poplarplugin::kPoplarPlatformId;
}

Status PoplarTransferManager::TransferLiteralFromDevice(
        se::StreamExecutor* executor, const se::DeviceMemoryBase& source,
        const Shape& device_shape, const Shape& literal_shape, Literal* literal) {
  TF_RET_CHECK(ShapeUtil::Compatible(device_shape, literal_shape));

  // Tuples are a special case and contain one or more shapes inside of them to
  // an arbitrary nesting depth.
  if (device_shape.element_type() == TUPLE) {
    *literal->mutable_shape() = literal_shape;
    TF_ASSIGN_OR_RETURN(
            std::vector<se::DeviceMemoryBase> element_buffers,
            ShallowCopyTupleFromDevice(executor, source, device_shape));
    TF_RET_CHECK(element_buffers.size() ==
                 ShapeUtil::TupleElementCount(device_shape));
    for (size_t i = 0; i < element_buffers.size(); ++i) {
      const Shape& element_device_shape = device_shape.tuple_shapes(i);
      const Shape& element_literal_shape = literal_shape.tuple_shapes(i);
      Literal* element_literal = literal->add_tuple_literals();
      // Recursively call TransferFromDevice to copy over the data in the
      // element array.
      TF_RETURN_IF_ERROR(TransferLiteralFromDevice(
              executor, element_buffers[i], element_device_shape,
              element_literal_shape, element_literal));
    }
    return Status::OK();
  }

  *literal->mutable_shape() = device_shape;
  literal->Reserve(ShapeUtil::ElementsIn(device_shape));
  TF_RETURN_IF_ERROR(TransferBufferFromDevice(
          executor, source, ShapeUtil::ByteSizeOf(device_shape),
          literal->MutableInternalData()));
  if (!ShapeUtil::Equal(literal_shape, device_shape)) {
    *literal = std::move(*literal->Relayout(literal_shape.layout()));
  }
  TF_RET_CHECK(ShapeUtil::Equal(literal_shape, literal->shape()));
  return Status::OK();
}

StatusOr<std::vector<se::DeviceMemoryBase>>
PoplarTransferManager::ShallowCopyTupleFromDevice(
        se::StreamExecutor* executor, const se::DeviceMemoryBase& source,
        const Shape& shape) {
  TF_RET_CHECK(ShapeUtil::IsTuple(shape));

  std::vector<void*> element_pointers(ShapeUtil::TupleElementCount(shape),
                                      nullptr);
  int64 tuple_size =
          ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  auto copy_status = executor->SynchronousMemcpyD2H(source, tuple_size,
                                                    element_pointers.data());
  if (!copy_status.ok()) {
    return AddStatus(
            Status(static_cast<tensorflow::error::Code>(copy_status.code()),
                   copy_status.error_message()),
            "failed transfer of tuple buffer " + ShapeUtil::HumanString(shape));
  }

  // Create a DeviceMemoryBase from each void* pointer.
  std::vector<se::DeviceMemoryBase> destination;
  for (size_t i = 0; i < element_pointers.size(); ++i) {
    if (element_pointers[i] == nullptr &&
        !ShapeUtil::HasZeroElements(shape.tuple_shapes(i))) {
      return FailedPrecondition("tuple contains nullptr at element %lu", i);
    }
    int64 buffer_size = ShapeUtil::ByteSizeOf(shape.tuple_shapes(i),
            sizeof(void*));
    destination.emplace_back(element_pointers[i], buffer_size);
  }
  return std::move(destination);
}

Status PoplarTransferManager::WriteTuplePointersToDevice(
        se::StreamExecutor* executor,
        tensorflow::gtl::ArraySlice<se::DeviceMemoryBase>
        elements,
        const Shape& shape, se::DeviceMemoryBase* region) {
  TF_RET_CHECK(elements.size() == ShapeUtil::TupleElementCount(shape));

  std::vector<const void*> element_pointers;
  for (const se::DeviceMemoryBase& element : elements) {
    element_pointers.push_back(element.opaque());
  }
  int64 tuple_size =
          ShapeUtil::ByteSizeOf(shape, /*pointer_size=*/sizeof(void*));

  return TransferBufferToDevice(executor, tuple_size, element_pointers.data(),
                                region);
}

Status PoplarTransferManager::TransferLiteralToDevice(
        se::StreamExecutor* executor, const Literal& literal,
        se::DeviceMemoryBase* destination) {
  const Shape& shape = literal.shape();

  if (ShapeUtil::IsTuple(literal.shape())) {
    std::vector<void*> tuple_elements_on_device;
    for (const Literal& tuple_element : literal.tuple_literals()) {
      se::DeviceMemoryBase allocation = executor->AllocateArray<uint8>(
              GetByteSizeRequirement(tuple_element.shape()));
      TF_RETURN_IF_ERROR(
              TransferLiteralToDevice(executor, tuple_element, &allocation));
      tuple_elements_on_device.push_back(allocation.opaque());
    }
    return TransferBufferToDevice(
            executor, tuple_elements_on_device.size() * sizeof(void*),
            tuple_elements_on_device.data(), destination);
  }
  return TransferBufferToDevice(executor, GetByteSizeRequirement(shape),
                                literal.InternalData(), destination);
}

Status
PoplarTransferManager::TransferLiteralToInfeed(se::StreamExecutor *executor,
                                               const Literal &literal) {
  return Unimplemented("TransferLiteralToInfeed");
}

Status
PoplarTransferManager::TransferBufferToInfeed(se::StreamExecutor* executor,
                                              int64 size, const void* source) {
  return Unimplemented("TransferBufferToInfeed");
}

Status
PoplarTransferManager::TransferLiteralFromOutfeed(
        perftools::gputools::StreamExecutor* executor,
        const Shape& literal_shape,
        Literal* literal) {
  return Unimplemented("TransferLiteralFromOutfeed");
}


Status PoplarTransferManager::ResetDevices(
        tensorflow::gtl::ArraySlice<perftools::gputools::StreamExecutor*>
        executors) {
  return Unimplemented("Device reset not supported");
}

int64 PoplarTransferManager::GetByteSizeRequirement(const Shape& shape) {
  return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
}

}  // namespace poplarplugin
}  // namespace xla

static std::unique_ptr<xla::TransferManager> CreatePoplarTransferManager() {
  return xla::MakeUnique<xla::poplarplugin::PoplarTransferManager>();
}

static bool InitModule() {
  xla::TransferManager::RegisterTransferManager(sep::kPoplarPlatformId,
                                                &CreatePoplarTransferManager);
  return true;
}
static bool module_initialized = InitModule();
