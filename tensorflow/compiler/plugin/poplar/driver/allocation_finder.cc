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

#include "tensorflow/compiler/plugin/poplar/driver/allocation_finder.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

class FindAllocatingInstructions : public DfsHloVisitorWithDefault {
public:
  FindAllocatingInstructions() {}

  ~FindAllocatingInstructions() override = default;

  Status DefaultAction(HloInstruction* hlo_instruction) override {
    return Status::OK();
  }

  Status HandleConstant(HloInstruction* inst, const Literal& literal) override {
    allocating_instructions.push_back(inst);
    return Status::OK();
  }

  Status HandleRng(HloInstruction* inst, RandomDistribution) override {
    allocating_instructions.push_back(inst);
    return Status::OK();
  }

  // TODO also find Rng in fusion instructions

  Status HandleParameter(HloInstruction* inst) override {
    allocating_instructions.push_back(inst);
    return Status::OK();
  }

  Status HandleSelect(HloInstruction* inst,
                      HloInstruction* /*pred*/,
                      HloInstruction* /*on_true*/,
                      HloInstruction* /*on_false*/) override {
    allocating_instructions.push_back(inst);
    return Status::OK();
  }

  Status HandleReduce(HloInstruction* inst,
                      HloInstruction* /*arg*/,
                      HloInstruction* /*init_value*/,
                      tensorflow::gtl::ArraySlice<int64> /*dimensions*/,
                      HloComputation* /*function*/) override {
    allocating_instructions.push_back(inst);
    return Status::OK();
  }

  Status HandleReduceWindow(HloInstruction* inst,
                            HloInstruction* /*operand*/,
                            const Window& /*window*/,
                            HloComputation* /*function*/) override {
    allocating_instructions.push_back(inst);
    return Status::OK();
  }

  Status HandleSelectAndScatter(HloInstruction* inst) override {
    allocating_instructions.push_back(inst);
    return Status::OK();
  }

  std::vector<HloInstruction*> allocating_instructions;
};

TensorTarget
AllocationFinder::FindConsumers(HloInstruction* inst) {
  for (auto user : inst->users()) {
    int64 op_index = user->operand_index(inst);
    switch (user->opcode()) {
      case HloOpcode::kConvolution:
      {
        return std::make_pair(user, op_index);
      }
      case HloOpcode::kCall:
      {
        HloComputation* comp = user->to_apply();
        HloInstruction* param = comp->parameter_instruction(op_index);
        TensorTarget target = FindConsumers(param);
        if (target.first != nullptr) {
          return target;
        }
        // if nothing appears then check the users of the call
        target = FindConsumers(user);
        if (target.first != nullptr) {
          return target;
        }
        break;
      }
      // TODO add other opcodes in here (reduce, ???)
      default:
        TensorTarget target = FindConsumers(user);
        if (target.first != nullptr) {
          return target;
        }
        break;
    }
  }
  return std::make_pair(nullptr, 0);
}

Status AllocationFinder::CreateAllocationMap(HloModule* module) {
  FindAllocatingInstructions finder;

  auto entry = module->entry_computation();
  TF_RETURN_IF_ERROR(entry->Accept(&finder));

  for (auto inst : finder.allocating_instructions) {
    TensorTarget target = FindConsumers(inst);
    if (target.first != nullptr) {
      tensor_allocation_map.insert(std::make_pair(inst, target));
    }
  }

  return Status::OK();
}

}
}
