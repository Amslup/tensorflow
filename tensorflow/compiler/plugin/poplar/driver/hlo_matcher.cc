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

#include "tensorflow/compiler/plugin/poplar/driver/hlo_matcher.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace xla {
namespace poplarplugin {

HloMatcher::HloMatcher(const std::vector<HloMatcherPattern>& patterns,
                       bool root_computation_only)
        : root_computation_only_(root_computation_only)
        , patterns_(std::move(patterns)) {
  matches_.resize(patterns.size());
}

bool HloMatcher::MatchPattern(HloInstruction* root,
                              const HloMatcherPattern& pattern,
                              HloMatcherMatched& match) {
  match.instructions[0] = root;

  for (unsigned int node_num = 1; node_num < pattern.size(); node_num++) {
    match.instructions[node_num] = nullptr;
  }

  for (unsigned int node_num=0; node_num < pattern.size(); node_num++) {
    HloInstruction* inst = match.instructions[node_num];
    if (inst == nullptr) {
      return false;
    }

    const HloMatcherNode& node(pattern[node_num]);

    if (node.opcode != inst->opcode()) {
      return false;
    }

    if (node.verification_fn && !node.verification_fn(inst)) {
      return false;
    }

    if ((node.operands.size() > 0) &&
        (inst->operand_count() != node.operands.size())) {
      return false;
    }

    for (unsigned int i=0; i<node.operands.size(); i++) {
      HloInstruction* operand = inst->mutable_operand(i);
      int n = node.operands[i];
      if (n < 0) {
        // When n<0, we are verifying an input to the fusion
        if (input_map_.count(n) > 0) {
          if (input_map_[n] != operand) {
            // An input label refers to only one instruction
            return false;
          }
        } else {
          if (input_set_.count(operand) > 0) {
            // An instruction cannot supply more than one input label
            return false;
          }

          input_map_[n] = operand;
          input_set_.insert(operand);
        }
      } else {
        // When n>0, we are verifying an instruction operand
        if (n <= node_num) {
          // Backward references are not allowed
          return false;
        }

        if (match.instructions[n] != nullptr &&
            match.instructions[n] != operand) {
          // The operand's opcode must match
          return false;
        }

        match.instructions[n] = operand;
      }
    }
  }

  ReplacedInstructions replaced;
  for (unsigned int node_num=0; node_num < pattern.size(); node_num++) {
    if (pattern[node_num].include_in_replacement) {
      replaced.push_back(match.instructions[node_num]);
    }
  }
  match.instructions = std::move(replaced);

  return true;
}

void HloMatcher::AddMatch(unsigned pattern, const HloMatcherMatched& match) {
  matches_[pattern].push_back(match);
  for (unsigned i=0; i<match.instructions.size(); i++) {
    match_map_.insert(std::make_pair(match.instructions[i],
                                     &matches_[pattern].back()));
  }
}

// TODO - make this non-recursive
void HloMatcher::MatchPatternStart(HloComputation* computation,
                                   HloInstruction* instruction) {

  visited_.insert(instruction);

  for (unsigned i=0; i<patterns_.size(); i++) {
    if (instruction->opcode() == patterns_[i][0].opcode) {
      // Try matching the whole pattern
      HloMatcherMatched match;
      match.ok = true;
      match.computation = computation;
      match.instructions.resize(patterns_[i].size());

      input_map_.clear();
      input_set_.clear();

      if (MatchPattern(instruction, patterns_[i], match)) {
        AddMatch(i, match);
      }
    }
  }

  for (HloInstruction* operand : instruction->operands()) {
    if (visited_.count(operand) == 0) {
      MatchPatternStart(computation, operand);
    }
  }
}



StatusOr<bool> HloMatcher::Run(HloModule *module) {

  if (root_computation_only_) {
    HloComputation* comp = module->entry_computation();
    visited_.clear();
    MatchPatternStart(comp, comp->root_instruction());

  } else {
    // Copy list of computations as we will be introducing new ones
    std::vector<HloComputation*> comps(module->computations().begin(),
                                       module->computations().end());

    for (auto* comp : comps) {
      if (!comp->IsFusionComputation()) {
        visited_.clear();
        MatchPatternStart(comp, comp->root_instruction());
      }
    }
  }

  unsigned int replacement_count = 0;
  for (int pattern=0; pattern<matches_.size(); pattern++) {
    for (HloMatcherMatched& match :  matches_[pattern]) {
      if (match.ok) {
        const ReplacedInstructions& replaced = ReplaceNodes(pattern, match);
        for (auto i : replaced) {
          auto range = match_map_.equal_range(i);
          for (auto m = range.first; m != range.second; ++m) {
            m->second->ok = false;
          }

          replacement_count++;
        }
      }
    }
  }

  patterns_.clear();
  visited_.clear();
  matches_.clear();
  match_map_.clear();
  input_set_.clear();
  input_map_.clear();

  return replacement_count != 0;
}

ReplacedInstructions HloMatcher::OutlineExpressionFromComputation(
        const HloMatcherMatched& matched,
        const std::string& outlined_computation_name,
        const char metadata_index) {

  auto& instructions_to_outline = matched.instructions;
  HloModule* module = matched.computation->parent();
  HloInstruction* root = instructions_to_outline[0];

  std::vector<HloInstruction*> to_outline = instructions_to_outline;
  std::reverse(to_outline.begin(), to_outline.end());

  auto builder = HloComputation::Builder(outlined_computation_name);

  // A map from original instructions to their counterparts in the new outlined
  // function.
  std::unordered_map<HloInstruction*, HloInstruction*> outlined_instructions;

  // A set that contains all instructions to be outlined.
  std::unordered_set<HloInstruction*> instruction_set_to_outline(
          to_outline.begin(), to_outline.end());

  std::vector<HloInstruction*> arguments;
  int64 parameter_count = 0;

  for (HloInstruction* instruction_to_outline : to_outline) {

    if (outlined_instructions.find(instruction_to_outline) ==
        outlined_instructions.end()) {

      HloInstruction* outlined_instruction =
              builder.AddInstruction(instruction_to_outline->Clone());

      for (int64 operand_num = 0;
           operand_num < outlined_instruction->operand_count(); ++operand_num) {
        HloInstruction* old_operand =
                outlined_instruction->mutable_operand(operand_num);

        HloInstruction** operand_slot = &(outlined_instructions[old_operand]);
        if (*operand_slot == nullptr) {
          arguments.push_back(old_operand);
          *operand_slot = builder.AddInstruction(HloInstruction::CreateParameter(
                  parameter_count, old_operand->shape(), "arg"));
          ++parameter_count;
        }
        TF_CHECK_OK(
                outlined_instruction->ReplaceOperandWith(operand_num,
                                                         *operand_slot));
      }

      // Insert the new instruction into the outlined_instructions map.
      InsertOrDie(&outlined_instructions, instruction_to_outline,
                  outlined_instruction);
    }
  }

  // Creates a call to the nested computation.
  HloComputation* nested_computation = module->AddEmbeddedComputation(
          builder.Build(FindOrDie(outlined_instructions, root)));
  HloInstruction* call = matched.computation->AddInstruction(
          HloInstruction::CreateCall(root->shape(), arguments,
                                     nested_computation));

  call->set_metadata(instructions_to_outline[metadata_index]->metadata());

  TF_CHECK_OK(root->ReplaceAllUsesWith(call));

  ReplacedInstructions replaced;
  for (auto i = instruction_set_to_outline.begin();
       i != instruction_set_to_outline.end(); ++i) {
    HloInstruction* inst = *i;
    if (inst->user_count() == 0) {
      TF_CHECK_OK(matched.computation->RemoveInstruction(inst));
      replaced.push_back(inst);
    }
  }

  return replaced;
}







}
}
