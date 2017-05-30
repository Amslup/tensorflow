#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_map.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_call.h"

#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <popstd/ActivationMapping.hpp>

namespace xla {
namespace poplarplugin {

class ParallelMapTester : public DfsHloVisitorWithDefault {
public:
  ParallelMapTester() : _is_ok(true) {}

  Status DefaultAction(HloInstruction* inst) override {
    if (inst->IsElementwise()) {
      return Status::OK();
    } else if (inst->opcode() == HloOpcode::kParameter) {
      return Status::OK();
    } else {
      _is_ok = false;
      return Status::OK();
    }
  }

  bool _is_ok;
};

port::StatusOr<bool>
IsComputationParallelMap(HloComputation* computation) {
  HloInstruction* root(computation->root_instruction());

  ParallelMapTester tester;
  TF_RETURN_IF_ERROR(root->Accept(&tester, false));

  return tester._is_ok;
}

port::StatusOr<poplar::program::Program>
CreateParallelMap(poplar::Graph &graph,
                  CompilerResources& res,
                  const HloInstruction *inst,
                  const xla::Shape& output,
                  TensorMap& tensor_map) {

  int64 op_count(inst->operand_count());
  std::vector<poplar::Tensor> inputs;

  for (int64 i = 0; i < op_count; i++) {
    poplar::Tensor t;
    TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, i, 0));
    inputs.push_back(t);
  }

  MapVisitor visitor(&graph, res, inputs, output);
  TF_RETURN_IF_ERROR(inst->to_apply()->Accept(&visitor));

  for (size_t i=0; i<visitor.output().size(); i++) {
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i,
                                       visitor.output()[i]));
  }

  return visitor.sequence;
}

port::StatusOr<poplar::program::Program>
CreateCallOp(poplar::Graph &graph,
             CompilerResources& res,
             const HloInstruction *inst,
             const xla::Shape& output,
             TensorMap& tensor_map) {

  int64 op_count(inst->operand_count());
  HloComputation* comp = inst->to_apply();
  poplar::program::Sequence seq;

  auto visitor(res.computation_map.find(comp));
  if (visitor == res.computation_map.end()) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Couldn't find sub-computation for Call op");
  }

  for (int64 i = 0; i < op_count; i++) {
    poplar::Tensor t;
    TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, i, 0));
    seq.add(poplar::program::Copy(t, visitor->second.inputs()[i]));
  }

  seq.add(visitor->second.sequence);

  for (size_t i=0; i<visitor->second.outputs().size(); i++) {
    // TODO use 'clone' when poplar supports it
    poplar::Tensor o = graph.addTensor(
            visitor->second.outputs()[i].elementType(),
            visitor->second.outputs()[i].shape());
    popstd::mapTensor(graph, o);    
    seq.add(poplar::program::Copy(visitor->second.outputs()[i], o));
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, o));
  }

  return seq;
}

port::StatusOr<poplar::program::Program>
CreateWhileOp(poplar::Graph &graph,
              CompilerResources& res,
              const HloInstruction *inst,
              const xla::Shape& output,
              TensorMap& tensor_map) {

  if (ShapeUtil::IsTuple(inst->operand(0)->shape())) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Poplar doesn't support tuple arguments to 'while' "
                        "operations");
  }
  if (ShapeUtil::IsTuple(inst->shape())) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Poplar doesn't support tuple return from 'while' "
                                "operations");
  }

  auto body_visitor(res.computation_map.find(inst->while_body()));
  if (body_visitor == res.computation_map.end()) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Couldn't find body sub-computation for while op");
  }

  auto condition_visitor(res.computation_map.find(inst->while_condition()));
  if (condition_visitor == res.computation_map.end()) {
    return port::Status(port::error::FAILED_PRECONDITION,
                        "Couldn't find condition sub-computation for while op");
  }

  poplar::Tensor body_input = body_visitor->second.inputs()[0];
  poplar::Tensor body_output = body_visitor->second.outputs()[0];
  poplar::Tensor cond_input = condition_visitor->second.inputs()[0];

  poplar::Tensor init;
  TF_ASSIGN_OR_RETURN(init, FindInstructionInput(tensor_map, inst, 0, 0));

  poplar::program::Sequence main_seq;
  main_seq.add(poplar::program::Copy(init, body_input));

  // Body
  body_visitor->second.sequence.add(poplar::program::Copy(body_output,
                                                          body_input));
  body_visitor->second.sequence.add(poplar::program::Copy(body_output,
                                                          cond_input));
  
  // Condition
  std::vector<poplar::Tensor> condition_inputs(1, body_output);

  poplar::program::RepeatWhileTrue repeat_while_true(
          condition_visitor->second.sequence,
          body_visitor->second.sequence);

  main_seq.add(repeat_while_true);

  // TODO use 'clone' when poplar supports it
  poplar::Tensor o = graph.addTensor(body_output.elementType(),
                                     body_output.shape());
  popstd::mapTensor(graph, o);
  main_seq.add(poplar::program::Copy(body_output, o));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, o));

  return main_seq;
}

}
}

