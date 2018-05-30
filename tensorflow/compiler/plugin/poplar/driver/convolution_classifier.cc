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

#include "tensorflow/compiler/plugin/poplar/driver/convolution_classifier.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

using ArgMap = std::multimap<const HloInstruction*, const HloInstruction*>;

/*
 * 1) find groups of convolutions which share the same inputs
 * 2) if any such group has >= 1 conv which has a graph parameter as an input,
 *    and >= 1 conv which does not have a graph parameter as an input, then
 *    mark the ones with a graph parameter as forwards, and the rest as
 *    backprop-filters
 * 3) any remaining convs which share the same weights as one of the forward
 *    convs is a backprop-input
 * 4) any remaining ones are inference only
 */

namespace {

// Find the actual source of an input. Entry/Exit from tuples and kCall
// instructions are traced though.
const HloInstruction* FindOperand(
    const HloInstruction* inst,
    const std::unique_ptr<CallGraph>& call_graph) {
  const HloInstruction* source = inst;
  std::vector<int64> tuple_stack;
  bool done = false;
  while (!done) {
    if (source->opcode() == HloOpcode::kParameter) {
      const auto* comp = source->parent();
      const auto& sites = call_graph->GetNode(comp).caller_callsites();
      if (sites.size() > 0) {
        int64 param_num = source->parameter_number();
        source = sites[0].instruction()->operand(param_num);
      } else {
        done = true;
      }
    }
    else if (source->opcode() == HloOpcode::kGetTupleElement) {
      // push tuple element index onto stack
      tuple_stack.push_back(source->tuple_index());
      source = source->operand(0);
    }
    else if (source->opcode() == HloOpcode::kTuple) {
      // pull tuple element index off stack and move to that operand
      int64 op_num = tuple_stack.back();
      tuple_stack.pop_back();
      source = source->operand(op_num);
    }
    else if (source->opcode() == HloOpcode::kTranspose) {
      // We allow ourselves to look through transpose ops
      source = source->operand(0);
    }
    else {
      done = true;
    }
  }
  return source;
}

}

StatusOr<bool> ConvolutionClassifier::Run(HloModule* module) {
  std::set<const HloInstruction*> variable_inputs;
  int64 n_vars = module->config().resource_update_count();
  for (int p = n_vars; p < module->entry_computation()->num_parameters(); p++) {
    variable_inputs.insert(
        module->entry_computation()->parameter_instruction(p));
  }

  for (const auto& comp : module->computations()) {
    if (!tensorflow::str_util::StartsWith(comp->name(), "_")) {
      for (const auto* inst : comp->instructions()) {
        switch (inst->opcode()) {
          case HloOpcode::kConvolution: {
            classification_[inst] = ClassificationType::INFERENCE;
            break;
          }
          case HloOpcode::kDot: {
            classification_[inst] = ClassificationType::INFERENCE;
            break;
          }
          case HloOpcode::kCall: {
            std::string name = inst->to_apply()->name();
            if (name == "_pop_op_depthwise_conv" ||
                name == "_pop_op_conv_with_reverse" ||
                name == "_pop_op_depthwise_filter") {
              classification_[inst] = ClassificationType::INFERENCE;
            }
            break;
          }
          default:
            break;
        }
      }
    }
  }

  std::unique_ptr<CallGraph> call_graph = CallGraph::Build(module);

  ArgMap arg0_fwd_map;
  ArgMap arg1_fwd_map;
  ArgMap arg1_rev_map;

  for (auto it : classification_) {
    const auto* arg0 = FindOperand(it.first->operand(0), call_graph);
    arg0_fwd_map.insert(std::make_pair(arg0, it.first));
    const auto* arg1 = FindOperand(it.first->operand(1), call_graph);
    arg1_fwd_map.insert(std::make_pair(arg1, it.first));
    arg1_rev_map.insert(std::make_pair(it.first, arg1));
  }

  std::set<const HloInstruction*> arg0_set;
  for (auto it : arg0_fwd_map) {
    arg0_set.insert(it.first);
  }

  std::set<const HloInstruction*> fwd;
  std::set<const HloInstruction*> wu;

  for (auto it : arg0_set) {
    if (arg0_fwd_map.count(it) > 1) {
      const auto& targets = arg0_fwd_map.equal_range(it);

      for (auto t = targets.first; t != targets.second; ++t) {
        auto weight = arg1_rev_map.find(t->second);
        if (weight != arg1_rev_map.end()) {
          if (variable_inputs.count(weight->second) > 0) {
            fwd.insert(t->second);
          } else {
            wu.insert(t->second);
          }
        }
      }

      if (fwd.size() > 0 && wu.size() > 0) {
        for (const auto* i : fwd) {
          classification_[i] = ClassificationType::FORWARD;
        }
        for (const auto* i : wu) {
          classification_[i] = ClassificationType::BACKPROP_FILTER;
        }
      }
    }
  }

  for (auto it : classification_) {
    if (it.second == ClassificationType::INFERENCE) {
      auto weight = arg1_rev_map.find(it.first);
      auto targets = arg1_fwd_map.equal_range(weight->second);
      for (auto t = targets.first; t != targets.second; ++t) {
        if (classification_[t->second] == ClassificationType::FORWARD) {
          classification_[it.first] = ClassificationType::BACKPROP_INPUT;
        }
      }
    }
  }

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
