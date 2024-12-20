// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_OPERAND_MAP_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_OPERAND_MAP_H_

#include <cstdint>
#include <map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/mediatek/neuron_adapter.h"

namespace litert::mediatek {

class OperandType : public NeuronOperandType {
 public:
  static Expected<OperandType> Create(const Tensor& t);

  OperandType(const OperandType&) = delete;

  OperandType(OperandType&& other) : dimensions_(std::move(other.dimensions_)) {
    // Copy all the scalar fields from other.
    *static_cast<NeuronOperandType*>(this) =
        *static_cast<NeuronOperandType*>(&other);
    // Reset the pointer fields by using own data.
    dimensions = dimensions_.data();
  };

  OperandType& operator=(const OperandType&) = delete;
  OperandType& operator=(OperandType&& other) = delete;

 private:
  explicit OperandType(int32_t mtk_type, std::vector<uint32_t>&& mtk_dimensions)
      : dimensions_(std::move(mtk_dimensions)) {
    this->type = mtk_type;
    this->dimensionCount = dimensions_.size();
    this->dimensions = dimensions_.data();
  };

  std::vector<uint32_t> dimensions_;
};

class OperandMap {
 public:
  OperandMap(const NeuronAdapter& neuron_adapter, NeuronModel* model)
      : neuron_adapter_(neuron_adapter), model_(model) {}

  Expected<uint32_t> Register(const NeuronOperandType& operand_type);

  // Find the ID for a given tensor and register the tensor with the model if
  // necessary.
  Expected<uint32_t> GetOperandIndex(const Tensor& t) {
    auto i = map_.find(t.Get());
    if (i != map_.end()) {
      return i->second;
    } else {
      return Register(t);
    }
  }

 private:
  Expected<uint32_t> Register(const Tensor& t);

  uint32_t AllocateOperandIndex() { return next_operand_index_++; }

  const NeuronAdapter& neuron_adapter_;
  NeuronModel* model_;
  int next_operand_index_ = 0;
  absl::flat_hash_map<LiteRtTensor, uint32_t> map_;
};

}  // namespace litert::mediatek

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_MEDIATEK_COMPILER_LEGALIZATIONS_OPERAND_MAP_H_
