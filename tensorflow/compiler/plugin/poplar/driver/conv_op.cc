#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

#include <popconv/Convolution.hpp>

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>

namespace xla {
namespace poplarplugin {

port::StatusOr<popconv::ConvParams>
GetConvolutionParameters(const HloInstruction* inst) {

  const Shape& input = inst->operand(0)->shape();
  const Shape& kernel = inst->operand(1)->shape();

  const Window& window(inst->window());

  if (ShapeUtil::Rank(input) != 4 || ShapeUtil::Rank(kernel) != 4) {
    return port::Status(
            port::error::FAILED_PRECONDITION,
            port::StrCat("Poplar supports 2D convolution only: ", inst->name()));
  }

  if (window.dimensions().size() != 2) {
    return port::Status(
            port::error::FAILED_PRECONDITION,
            port::StrCat("Invalid window dimension count on ", inst->name()));
  }

  std::string dtype;
  TF_ASSIGN_OR_RETURN(dtype, PoplarDataType(input));

  std::vector<size_t> input_dims = PoplarShapeFromXlaShape(input);
  std::vector<size_t> kernel_dims = PoplarShapeFromXlaShape(kernel);

  const ConvolutionDimensionNumbers& dims(inst->convolution_dimension_numbers());
  unsigned int n_b = input_dims[dims.input_batch_dimension()];
  unsigned int n_i = input_dims[dims.input_feature_dimension()];
  unsigned int n_o = kernel_dims[dims.kernel_output_feature_dimension()];
  unsigned int n_y = input_dims[dims.spatial_dimensions(0)];
  unsigned int n_x = input_dims[dims.spatial_dimensions(1)];
  unsigned int f_y = kernel_dims[dims.kernel_spatial_dimensions(0)];
  unsigned int f_x = kernel_dims[dims.kernel_spatial_dimensions(1)];

  unsigned int s_y = window.dimensions(0).stride();
  unsigned int s_x = window.dimensions(1).stride();

  int pl_y = window.dimensions(0).padding_low();
  int pl_x = window.dimensions(1).padding_low();

  int pu_y = window.dimensions(0).padding_high();
  int pu_x = window.dimensions(1).padding_high();

  unsigned int di_y = window.dimensions(0).base_dilation();
  unsigned int di_x = window.dimensions(1).base_dilation();

  unsigned int dw_y = window.dimensions(0).window_dilation();
  unsigned int dw_x = window.dimensions(1).window_dilation();

  popconv::ConvParams params(dtype,
                             n_b,
                             {n_y, n_x},
                             {f_y, f_x},
                             n_i, n_o,
                             {s_y, s_x},
                             {pl_y, pl_x},
                             {pu_y, pu_x},
                             {di_y, di_x},
                             {0, 0},
                             {0, 0},
                             {dw_y, dw_x},
                             1);

  return params;
}

port::StatusOr<popconv::ConvParams>
GetDepthConvolutionParameters(const HloInstruction* inst) {

  const Shape& input = inst->operand(0)->shape();
  const Shape& kernel = inst->operand(1)->shape();

  const Window& window(inst->window());

  if (ShapeUtil::Rank(input) != 4 || ShapeUtil::Rank(kernel) != 4) {
    return port::Status(
            port::error::FAILED_PRECONDITION,
            port::StrCat("Poplar supports 2D convolution only: ", inst->name()));
  }

  if (window.dimensions().size() != 2) {
    return port::Status(
            port::error::FAILED_PRECONDITION,
            port::StrCat("Invalid window dimension count on ", inst->name()));
  }

  std::string dtype;
  TF_ASSIGN_OR_RETURN(dtype, PoplarDataType(input));

  std::vector<size_t> input_dims = PoplarShapeFromXlaShape(input);
  std::vector<size_t> kernel_dims = PoplarShapeFromXlaShape(kernel);

  const ConvolutionDimensionNumbers& dims(inst->convolution_dimension_numbers());
  unsigned int n_b = input_dims[dims.input_batch_dimension()];
  unsigned int n_i = input_dims[dims.input_feature_dimension()];
  unsigned int n_o = kernel_dims[dims.kernel_output_feature_dimension()];
  unsigned int n_y = input_dims[dims.spatial_dimensions(0)];
  unsigned int n_x = input_dims[dims.spatial_dimensions(1)];
  unsigned int f_y = kernel_dims[dims.kernel_spatial_dimensions(0)];
  unsigned int f_x = kernel_dims[dims.kernel_spatial_dimensions(1)];

  unsigned int s_y = window.dimensions(0).stride();
  unsigned int s_x = window.dimensions(1).stride();

  int pl_y = window.dimensions(0).padding_low();
  int pl_x = window.dimensions(1).padding_low();

  int pu_y = window.dimensions(0).padding_high();
  int pu_x = window.dimensions(1).padding_high();

  unsigned int di_y = window.dimensions(0).base_dilation();
  unsigned int di_x = window.dimensions(1).base_dilation();

  unsigned int dw_y = window.dimensions(0).window_dilation();
  unsigned int dw_x = window.dimensions(1).window_dilation();

  n_o = n_o / n_i;

  popconv::ConvParams params(dtype,
                             n_b,
                             {n_y, n_x},
                             {f_y, f_x},
                             1, n_o,
                             {s_y, s_x},
                             {pl_y, pl_x},
                             {pu_y, pu_x},
                             {di_y, di_x},
                             {0, 0},
                             {0, 0},
                             {dw_y, dw_x},
                             n_i);

  return params;
}

static popconv::Pass GetConvolutionPass(const HloInstruction* inst) {
  const ConvolutionDimensionNumbers& dims(inst->convolution_dimension_numbers());
  if (dims.kernel_input_feature_dimension() == 0) {
    return popconv::Pass::TRAINING_WU;
  }
  return popconv::Pass::TRAINING_FWD;
}

static bool is_identity_shuffle(const std::vector<unsigned int> shuffle) {
  for (unsigned int i=0; i<4; i++) {
    if (shuffle[i] != i) return false;
  }
  return true;
}

port::StatusOr<poplar::Tensor>
ShuffleConvolutionInput(const HloInstruction* inst,
                        const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(inst->convolution_dimension_numbers());

  std::vector<unsigned int> shuffle(4);
  shuffle[d.input_batch_dimension()] = 0;
  shuffle[d.spatial_dimensions(0)] = 1;
  shuffle[d.spatial_dimensions(1)] = 2;
  shuffle[d.input_feature_dimension()] = 3;

  return is_identity_shuffle(shuffle) ? tensor : tensor.dimShuffle(shuffle);
}

port::StatusOr<poplar::Tensor>
ShuffleConvolutionWeights(const HloInstruction* inst,
                          const poplar::Tensor& tensor) {
  const ConvolutionDimensionNumbers& d(inst->convolution_dimension_numbers());

  std::vector<unsigned int> shuffle(4);
  shuffle[d.kernel_spatial_dimensions(0)] = 0;
  shuffle[d.kernel_spatial_dimensions(1)] = 1;
  shuffle[d.kernel_output_feature_dimension()] = 2;
  shuffle[d.kernel_input_feature_dimension()] = 3;

  return is_identity_shuffle(shuffle) ? tensor : tensor.dimShuffle(shuffle);
}

poplar::Tensor RemoveGroupsDimensionFromWeights(const poplar::Tensor& t) {
  return t.reshape({t.dim(1), t.dim(2), t.dim(3), t.dim(4)});
}

poplar::Tensor AddGroupsDimensionToWeights(const poplar::Tensor& t) {
  return t.reshape({1, t.dim(0), t.dim(1), t.dim(2), t.dim(3)});
}

port::StatusOr <poplar::program::Program>
CreateConv2D(poplar::Graph &graph,
             CompilerResources& res,
             const HloInstruction *inst,
             const xla::Shape &output_shape,
             TensorMap &tensor_map) {

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 0));

  // Find the kernel tensor
  poplar::Tensor kernel;
  TF_ASSIGN_OR_RETURN(kernel, FindInstructionInput(tensor_map, inst, 1));

  popconv::ConvOptions opts;
  opts.cache = &res.convolution_cache;
  opts.pass = GetConvolutionPass(inst);

  const ConvolutionDimensionNumbers& d(inst->convolution_dimension_numbers());

  popconv::ConvParams params;
  TF_ASSIGN_OR_RETURN(params, GetConvolutionParameters(inst));

  poplar::program::Sequence prog;

  std::vector<unsigned int> shuffle(4);
  shuffle[0] = d.input_batch_dimension();
  shuffle[1] = d.spatial_dimensions(0);
  shuffle[2] = d.spatial_dimensions(1);
  shuffle[3] = d.input_feature_dimension();

  if (!is_identity_shuffle(shuffle)) {
    in = in.dimShuffle(shuffle);
    auto name = port::StrCat(inst->name(), "_input_copy");
    poplar::Tensor conv_in = popconv::createInput(graph, params, name, opts);
    prog.add(poplar::program::Copy(in, conv_in));
    in = conv_in;
  }

  shuffle[0] = d.kernel_spatial_dimensions(0);
  shuffle[1] = d.kernel_spatial_dimensions(1);
  shuffle[2] = d.kernel_output_feature_dimension();
  shuffle[3] = d.kernel_input_feature_dimension();

  if (!is_identity_shuffle(shuffle)) {
    kernel = kernel.dimShuffle(shuffle);
    kernel = AddGroupsDimensionToWeights(kernel);
    auto name = port::StrCat(inst->name(), "_weights_copy");
    poplar::Tensor conv_kernel = popconv::createWeights(graph, params, name, opts);
    prog.add(poplar::program::Copy(kernel, conv_kernel));
    kernel = conv_kernel;
  } else {
    kernel = AddGroupsDimensionToWeights(kernel);
  }

  // Add the convolution
  poplar::Tensor out = popconv::convolution(graph, in, kernel, params,
                                            false, prog, inst->name(), opts);

  shuffle[d.output_batch_dimension()] = 0;
  shuffle[d.spatial_dimensions(0)] = 1;
  shuffle[d.spatial_dimensions(1)] = 2;
  shuffle[d.output_feature_dimension()] = 3;

  if (!is_identity_shuffle(shuffle)) {
    out = out.dimShuffle(shuffle);
  }

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return prog;
}

port::StatusOr <poplar::program::Program>
CreateBiasAddOp(poplar::Graph &graph,
                CompilerResources& res,
                const HloInstruction *inst,
                const xla::Shape &output_shape,
                TensorMap &tensor_map) {
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 0));

  poplar::Tensor bias;
  TF_ASSIGN_OR_RETURN(bias, FindInstructionInput(tensor_map, inst, 1));

  poplar::program::Sequence prog;
  popconv::addBias(graph, in, bias, prog, inst->name());

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, in));
  return prog;
}

port::StatusOr <poplar::program::Program>
CreateBiasAddBcastOp(poplar::Graph &graph,
                     CompilerResources& res,
                     const HloInstruction *inst,
                     const xla::Shape &output_shape,
                     TensorMap &tensor_map) {
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 1));

  poplar::Tensor bias;
  TF_ASSIGN_OR_RETURN(bias, FindInstructionInput(tensor_map, inst, 0));

  poplar::program::Sequence prog;
  popconv::addBias(graph, in, bias, prog, inst->name());

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, in));
  return prog;
}

port::StatusOr<poplar::program::Program>
ConvBiasApply(poplar::Graph &graph,
              CompilerResources& res,
              const HloInstruction *inst,
              const xla::Shape& output_shape,
              TensorMap& tensor_map) {

  const HloInstruction* root =
          inst->to_apply()->root_instruction();

  // Find the deltas
  poplar::Tensor deltas;
  TF_ASSIGN_OR_RETURN(deltas, FindInstructionInput(tensor_map, inst, 0));

  // Find the biases
  poplar::Tensor biases;
  TF_ASSIGN_OR_RETURN(biases, FindInstructionInput(tensor_map, inst, 1));

  // Find the learning rate constant
  auto literal = root->operand(1)->operand(0)->operand(0)->literal();

  std::unique_ptr<Literal> float_lit;
  TF_ASSIGN_OR_RETURN(float_lit, literal.Convert(F32));

  float learning_rate = float_lit->GetFirstElement<float>();

  poplar::program::Sequence prog;
  popconv::convolutionBiasUpdate(graph, deltas, biases, learning_rate, "float",
                                 prog, inst->name());

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, biases));

  return prog;
}

port::StatusOr<poplar::program::Program>
CreateDepthwiseConvolutionOp(poplar::Graph &graph,
                             CompilerResources& res,
                             const HloInstruction *inst,
                             const xla::Shape& output_shape,
                             TensorMap& tensor_map) {
  const HloInstruction* root =
          inst->to_apply()->root_instruction();

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 1));

  // Find the kernel tensor
  poplar::Tensor kernel;
  TF_ASSIGN_OR_RETURN(kernel, FindInstructionInput(tensor_map, inst, 0));

  popconv::ConvOptions opts;
  opts.cache = &res.convolution_cache;
  opts.pass = GetConvolutionPass(root);

  const ConvolutionDimensionNumbers& d(root->convolution_dimension_numbers());

  popconv::ConvParams params;
  TF_ASSIGN_OR_RETURN(params, GetDepthConvolutionParameters(root));

  poplar::program::Sequence prog;

  std::vector<unsigned int> shuffle(4);
  shuffle[0] = d.input_batch_dimension();
  shuffle[1] = d.spatial_dimensions(0);
  shuffle[2] = d.spatial_dimensions(1);
  shuffle[3] = d.input_feature_dimension();

  if (!is_identity_shuffle(shuffle)) {
    in = in.dimShuffle(shuffle);
    auto name = port::StrCat(root->name(), "_input_copy");
    poplar::Tensor conv_in = popconv::createInput(graph, params, name, opts);
    prog.add(poplar::program::Copy(in, conv_in));
    in = conv_in;
  }

  shuffle[0] = d.kernel_spatial_dimensions(0);
  shuffle[1] = d.kernel_spatial_dimensions(1);
  shuffle[2] = d.kernel_output_feature_dimension();
  shuffle[3] = d.kernel_input_feature_dimension();

  if (!is_identity_shuffle(shuffle)) {
    kernel = kernel.dimShuffle(shuffle);
    kernel = AddGroupsDimensionToWeights(kernel);
    kernel = kernel.dimShuffle({4, 1, 2, 3, 0});
    auto name = port::StrCat(root->name(), "_weights_copy");
    poplar::Tensor conv_kernel = popconv::createWeights(graph, params, name, opts);
    prog.add(poplar::program::Copy(kernel, conv_kernel));
    kernel = conv_kernel;
  } else {
    kernel = AddGroupsDimensionToWeights(kernel);
    kernel = kernel.dimShuffle({4, 1, 2, 3, 0});
  }

  // Add the convolution
  poplar::Tensor out = popconv::convolution(graph, in, kernel, params,
                                            false, prog, inst->name(), opts);

  shuffle[d.output_batch_dimension()] = 0;
  shuffle[d.spatial_dimensions(0)] = 1;
  shuffle[d.spatial_dimensions(1)] = 2;
  shuffle[d.output_feature_dimension()] = 3;

  if (!is_identity_shuffle(shuffle)) {
    out = out.dimShuffle(shuffle);
  }

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return prog;
}

port::StatusOr<poplar::program::Program>
Create2DConvWithReverse(poplar::Graph &graph,
                        CompilerResources& res,
                        const HloInstruction *inst,
                        const xla::Shape& output_shape,
                        TensorMap& tensor_map) {
  const HloInstruction* conv =
          inst->to_apply()->root_instruction();

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 1));

  // Find the kernel tensor
  poplar::Tensor kernel;
  TF_ASSIGN_OR_RETURN(kernel, FindInstructionInput(tensor_map, inst, 0));

  popconv::ConvOptions opts;
  opts.cache = &res.convolution_cache;
  opts.pass = popconv::Pass::TRAINING_BWD;

  const ConvolutionDimensionNumbers& d(conv->convolution_dimension_numbers());

  popconv::ConvParams params;
  TF_ASSIGN_OR_RETURN(params, GetConvolutionParameters(conv));

  poplar::program::Sequence prog;

  std::vector<unsigned int> shuffle(4);
  shuffle[0] = d.input_batch_dimension();
  shuffle[1] = d.spatial_dimensions(0);
  shuffle[2] = d.spatial_dimensions(1);
  shuffle[3] = d.input_feature_dimension();

  if (!is_identity_shuffle(shuffle)) {
    in = in.dimShuffle(shuffle);
  }

  shuffle[0] = d.kernel_spatial_dimensions(0);
  shuffle[1] = d.kernel_spatial_dimensions(1);
  shuffle[2] = d.kernel_input_feature_dimension();
  shuffle[3] = d.kernel_output_feature_dimension();

  if (!is_identity_shuffle(shuffle)) {
    kernel = kernel.dimShuffle(shuffle);
  }
  kernel = AddGroupsDimensionToWeights(kernel);

  // Add the convolution
  poplar::Tensor out = popconv::convolution(graph, in, kernel, params,
                                            true, prog, conv->name(), opts);

  shuffle[d.output_batch_dimension()] = 0;
  shuffle[d.spatial_dimensions(0)] = 1;
  shuffle[d.spatial_dimensions(1)] = 2;
  shuffle[d.output_feature_dimension()] = 3;

  if (!is_identity_shuffle(shuffle)) {
    out = out.dimShuffle(shuffle);
  }

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return prog;
}

}
}
