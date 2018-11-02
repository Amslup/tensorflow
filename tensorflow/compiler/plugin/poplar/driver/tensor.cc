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

#include "absl/strings/str_cat.h"

#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/conversions.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/util/bcast.h"
#include "tensorflow/stream_executor/lib/status.h"

#include <poplar/Engine.hpp>
#include <poplar/OptionFlags.hpp>
#include <poputil/TileMapping.hpp>

using ::absl::StrCat;
using ::tensorflow::str_util::Join;

namespace xla {
namespace poplarplugin {

StatusOr<poplar::Type> PoplarDataType(const xla::Shape& shape) {
  switch (shape.element_type()) {
    case PRED:
      return poplar::BOOL;
    case S8:
    case U8:
      return poplar::CHAR;
    case S16:
    case U16:
      return poplar::SHORT;
    case S32:
    case U32:
      return poplar::INT;
    case S64:
    case U64:
      return poplar::INT;
    case F16:
      return poplar::HALF;
    case F32:
      return poplar::FLOAT;
    default:
      return xla::FailedPrecondition("unsupported primitive type in poplar %s",
                                     PrimitiveType_Name(shape.element_type()));
  }
}

std::vector<size_t> PoplarShapeFromXlaShape(const xla::Shape& xla_shape) {
  std::vector<size_t> shape;
  for (auto d : xla_shape.dimensions()) {
    shape.push_back(d);
  }
  return shape;
}

xla::Shape XlaShapeFromPoplarShape(PrimitiveType element_type,
                                   const std::vector<size_t>& poplar_shape) {
  xla::Shape shape;
  shape.set_element_type(element_type);
  for (int64 dimension : poplar_shape) {
    shape.add_dimensions(dimension);
  }
  LayoutUtil::SetToDefaultLayout(&shape);
  return shape;
}

poplar::Tensor ConvertToDeviceLayout(const Shape& shape,
                                     const poplar::Tensor& tensor) {
  // Reshape then dimshuffle
  poplar::Tensor out = tensor;
  if (!LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    unsigned int rank = tensor.rank();
    std::vector<std::size_t> dim(rank);
    std::vector<unsigned int> shuffle(rank);
    for (unsigned int i = 0; i < rank; i++) {
      shuffle[shape.layout().minor_to_major(i)] = rank - i - 1;
      dim[rank - i - 1] = tensor.dim(shape.layout().minor_to_major(i));
    }

    out = out.reshape(dim);
    out = out.dimShuffle(shuffle);
  }
  return out;
}

poplar::Tensor ConvertFromDeviceLayout(const Shape& shape,
                                       const poplar::Tensor& tensor) {
  // Dimshuffle then reshape
  poplar::Tensor out = tensor;
  if (!LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    unsigned int rank = tensor.rank();
    std::vector<unsigned int> shuffle(rank);
    for (unsigned int i = 0; i < rank; i++) {
      shuffle[rank - i - 1] = shape.layout().minor_to_major(i);
    }
    out = out.dimShuffle(shuffle);
    out = out.reshape(tensor.shape());
  }
  return out;
}

StatusOr<poplar::Tensor> AddPlainTensor(poplar::Graph& graph,
                                        const std::string& debug_name,
                                        const xla::Shape& shape) {
  poplar::Tensor out;
  std::vector<std::size_t> dim = PoplarShapeFromXlaShape(shape);
  poplar::Type poplar_type;
  TF_ASSIGN_OR_RETURN(poplar_type, PoplarDataType(shape));

  out = graph.addVariable(poplar_type, dim, debug_name);
  poputil::mapTensorLinearly(graph, out);
  return out;
}

StatusOr<poplar::Tensor> AddRnnSequence(poplar::Graph& graph,
                                        const std::string& debug_name,
                                        const xla::Shape& shape) {
  poplar::Tensor out;
  std::vector<std::size_t> dim = PoplarShapeFromXlaShape(shape);
  poplar::Type poplar_type;
  TF_ASSIGN_OR_RETURN(poplar_type, PoplarDataType(shape));

  out = graph.addVariable(poplar_type, dim, debug_name);

  for (auto i = 0; i != dim[0]; ++i) {
    poputil::mapTensorLinearly(graph, out[i]);
  }

  return out;
}

template <typename IIter1, typename IIter2, typename OIter, typename Zipper>
static void zip(IIter1 ibegin1, IIter1 iend1, IIter2 ibegin2, OIter obegin,
                Zipper zipper) {
  for (; ibegin1 != iend1; ++ibegin1, ++ibegin2, ++obegin) {
    *obegin = zipper(*ibegin1, *ibegin2);
  }
}

// Find a value for G s.t. D / G <= T, and G | D.
static StatusOr<std::size_t> FindG(const std::size_t D, const std::size_t T) {
  for (std::size_t g = (D + T - 1) / T; g <= D; ++g) {
    if (D % g == 0) {
      return g;
    }
  }

  return tensorflow::errors::FailedPrecondition(
      "Cannot find a value of G that is both a factor of D and satisfies D / G "
      "<= T");
}

// Find the sequence dimension, if there is one
static StatusOr<std::size_t> FindSeqDim(const xla::Shape& shape_xla,
                                        const xla::Shape& slice_shape_xla) {
  const auto shape = PoplarShapeFromXlaShape(shape_xla);
  const auto slice_shape = PoplarShapeFromXlaShape(slice_shape_xla);
  const auto volume =
      std::accumulate(shape.begin(), shape.end(), std::size_t(1),
                      std::multiplies<std::size_t>());
  const auto slice_volume =
      std::accumulate(slice_shape.begin(), slice_shape.end(), std::size_t(1),
                      std::multiplies<std::size_t>());

  // If the desired shape is 1D, then no special work is required.
  // If the slice shape is the same as the input shape, this is just a copy
  if (ShapeUtil::Rank(shape_xla) > 1 && shape != slice_shape && volume > 1 &&
      slice_volume > 1) {
    // Calculate the element-wise ratio between the slice the input rank
    std::vector<float> dimension_ratios(shape.size());
    zip(slice_shape.begin(), slice_shape.end(), shape.begin(),
        dimension_ratios.begin(), std::divides<float>());

    // Assumes the sequence dimension is the dimension with the smallest ratio
    // between the input and the slice.
    return std::distance(
        dimension_ratios.begin(),
        std::min_element(dimension_ratios.begin(), dimension_ratios.end()));
  }

  return tensorflow::errors::FailedPrecondition(
      "Cannot compute slice sequence dimension");
}

StatusOr<poplar::Tensor> AddDynamicSliceTensor(
    poplar::Graph& graph, const std::string& debug_name,
    const xla::Shape& shape_xla, const xla::Shape& slice_shape_xla) {
  poplar::Tensor unused;
  return AddDynamicSliceTensor(graph, debug_name, shape_xla, slice_shape_xla,
                               unused);
}

StatusOr<poplar::Tensor> AddDynamicSliceTensor(
    poplar::Graph& graph, const std::string& debug_name,
    const xla::Shape& shape_xla, const xla::Shape& slice_shape_xla,
    poplar::Tensor& physical_layout) {
  const auto shape = PoplarShapeFromXlaShape(shape_xla);
  const auto volume =
      std::accumulate(shape.begin(), shape.end(), std::size_t(1),
                      std::multiplies<std::size_t>());

  // If we are able to compute the sequence_dimension
  const auto sequence_dimension_status = FindSeqDim(shape_xla, slice_shape_xla);
  if (!sequence_dimension_status.ok()) {
    TF_ASSIGN_OR_RETURN(physical_layout,
                        AddPlainTensor(graph, debug_name, shape_xla));
    return physical_layout;
  }

  const auto sequence_dimension = sequence_dimension_status.ValueOrDie();

  // Create a tensor of the form [D/G, S, G] where D is the product of the N-1
  // dimensions that are not the sequence dimension, S is the size of the
  // sequence dimension, and G is a factor of D chosen to ensure that
  // D/G <= T, where T is the number of tiles.
  const auto T = graph.getTarget().getNumTiles();
  const auto D = volume / shape[sequence_dimension];
  const auto S = shape[sequence_dimension];
  const auto G_status = FindG(D, T);
  if (!G_status.ok()) {
    TF_ASSIGN_OR_RETURN(physical_layout,
                        AddPlainTensor(graph, debug_name, shape_xla));
    return physical_layout;
  }

  const auto G = G_status.ValueOrDie();
  if (D == G) {
    TF_ASSIGN_OR_RETURN(physical_layout,
                        AddPlainTensor(graph, debug_name, shape_xla));
    return physical_layout;
  }

  // If a value for G was found
  TF_ASSIGN_OR_RETURN(poplar::Type poplar_type, PoplarDataType(shape_xla));

  poplar::Tensor out =
      graph.addVariable(poplar_type, {D / G, S, G}, debug_name);
  physical_layout = out;

  // Map the sequence dimension across the tiles
  for (std::size_t i = 0; i < out.dim(0); ++i) {
    graph.setTileMapping(out[i], i);
  }

  // Reshape, with the sequence dimension being the last dimension
  auto shape_tmp = shape;
  std::swap(shape_tmp[sequence_dimension], shape_tmp.back());
  out = out.reshape(shape_tmp);

  // Shuffle the dimensions back into the desired order
  // out.dimSwap(sequence_dimension, shape.size() - 1)
  std::vector<unsigned> permutation(shape.size());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation[sequence_dimension], permutation.back());
  out = out.dimShuffle(permutation);

  return out;
}

static StatusOr<poplar::Tensor> AddConvolutionInput(
    poplar::Graph& graph, const std::string& debug_name,
    const HloInstruction* op_target, const HloInstruction* conv_target,
    CompilerResources& resources) {
  poplin::ConvParams params;
  TF_ASSIGN_OR_RETURN(params,
                      GetConvolutionParameters(op_target, conv_target, 0, 1));

  auto name = StrCat(debug_name, "_input");
  poplar::OptionFlags opts;
  poplar::Tensor out = poplin::createInput(graph, params, name, opts,
                                           &resources.convolution_cache);
  return ShuffleConvolutionInputToTensorflow(conv_target, out);
}

static StatusOr<poplar::Tensor> AddConvolutionWeights(
    poplar::Graph& graph, const std::string& debug_name,
    const HloInstruction* op_target, const HloInstruction* conv_target,
    CompilerResources& resources) {
  poplin::ConvParams params;
  TF_ASSIGN_OR_RETURN(params,
                      GetConvolutionParameters(op_target, conv_target, 0, 1));

  auto name = StrCat(debug_name, "_weights");
  poplar::OptionFlags opts;
  poplar::Tensor out = poplin::createWeights(graph, params, name, opts,
                                             &resources.convolution_cache);

  out = RemoveGroupsDimensionFromWeights(params, out, false);

  return ShuffleConvolutionWeightsToTensorflow(conv_target, out);
}

static StatusOr<poplar::Tensor> AddLeftMatMul(poplar::Graph& graph,
                                              const std::string& debug_name,
                                              const xla::Shape& shape,
                                              const HloInstruction* target,
                                              CompilerResources& resources) {
  poplar::Type type;
  TF_ASSIGN_OR_RETURN(type, PoplarDataType(shape));
  const auto& aShape = PoplarShapeFromXlaShape(target->operand(0)->shape());
  const auto& bShape = PoplarShapeFromXlaShape(target->operand(1)->shape());
  auto name = StrCat(debug_name, "_lhs");
  poplar::OptionFlags opts;
  return poplin::createMatMulInputLHS(graph, type, aShape, bShape, name, opts,
                                      &resources.dot_cache);
}

static StatusOr<poplar::Tensor> AddRightMatMul(poplar::Graph& graph,
                                               const std::string& debug_name,
                                               const xla::Shape& shape,
                                               const HloInstruction* target,
                                               CompilerResources& resources) {
  poplar::Type type;
  TF_ASSIGN_OR_RETURN(type, PoplarDataType(shape));
  const auto& aShape = PoplarShapeFromXlaShape(target->operand(0)->shape());
  const auto& bShape = PoplarShapeFromXlaShape(target->operand(1)->shape());
  auto name = StrCat(debug_name, "_rhs");
  poplar::OptionFlags opts;
  return poplin::createMatMulInputRHS(graph, type, aShape, bShape, name, opts,
                                      &resources.dot_cache);
}

static StatusOr<poplar::Tensor> PathTransform(
    poplar::Graph& graph, poplar::Tensor in,
    const std::vector<const HloInstruction*>& forward,
    const std::vector<const HloInstruction*>& backward) {
  // Now apply any transformations required by the path from the source to
  // the target
  if (forward.size() > 1) {
    for (auto itr = std::next(forward.begin()); itr != forward.end(); ++itr) {
      auto& inst = *itr;
      switch (inst->opcode()) {
        case HloOpcode::kTranspose: {
          const auto permutation =
              convert_array<std::vector<unsigned>>(inst->dimensions());
          in = in.dimShuffle(permutation);
          break;
        }
        case HloOpcode::kReshape: {
          const auto dims = PoplarShapeFromXlaShape(inst->shape());
          in = in.reshape(dims);
          break;
        }
        case HloOpcode::kAdd: {
          break;
        }
        default: {
          const auto& name = GetDebugName(backward.front());
          in = AddPlainTensor(graph, name, inst->shape()).ValueOrDie();
          break;
        }
      }
    }
  }

  for (auto i = backward.rbegin(); i != backward.rend(); ++i) {
    auto& inst = *i;
    switch (inst->opcode()) {
      case HloOpcode::kTranspose: {
        std::vector<unsigned> permutation(
            convert_array<std::vector<unsigned>>(inst->dimensions()));
        std::vector<unsigned> shuffle(permutation.size());
        for (int d = 0; d < permutation.size(); d++) {
          shuffle[permutation[d]] = d;
        }
        in = in.dimShuffle(shuffle);
        break;
      }
      case HloOpcode::kReshape: {
        std::vector<size_t> dims(
            PoplarShapeFromXlaShape(inst->operand(0)->shape()));
        in = in.reshape(dims);
        break;
      }
      case HloOpcode::kBroadcast: {
        std::vector<unsigned> permutation(in.rank());
        std::iota(permutation.begin(), permutation.end(), 0);
        std::swap(permutation.front(), permutation[inst->dimensions(0)]);

        in = in.dimShuffle(permutation);
        in = in[0];
      }
      case HloOpcode::kAdd: {
        break;
      }
      default: { break; }
    }
  }

  return in;
}

StatusOr<poplar::Tensor> AddTensor(poplar::Graph& graph,
                                   const TensorSource& src,
                                   const xla::Shape& shape,
                                   CompilerResources& resources,
                                   const TensorMap& tensor_map) {
  const auto& name = GetDebugName(src.first);
  poplar::Tensor out;

  auto target = resources.annotations.tensor_allocation_map.find(src);
  if (target != resources.annotations.tensor_allocation_map.end()) {
    const auto* tgt = target->second.tgt;
    auto tshape = tgt->operand(target->second.input_index)->shape();

    // Temporarily don't do biasadd
    if (IsPopOpsCall(tgt, "biasadd")) {
      TF_ASSIGN_OR_RETURN(out, AddPlainTensor(graph, name, shape));
      return out;
    }

    switch (tgt->opcode()) {
      case HloOpcode::kConvolution: {
        switch (target->second.input_index) {
          case 0: {
            TF_ASSIGN_OR_RETURN(
                out, AddConvolutionInput(graph, name, tgt, tgt, resources));
            break;
          }
          case 1: {
            TF_ASSIGN_OR_RETURN(
                out, AddConvolutionWeights(graph, name, tgt, tgt, resources));
            break;
          }
          default:
            return xla::FailedPrecondition(
                "invalid operand for tensor allocation on %s",
                src.first->name().c_str());
        }
        break;
      }
      case HloOpcode::kDot: {
        switch (target->second.input_index) {
          case 0: {
            TF_ASSIGN_OR_RETURN(
                out, AddLeftMatMul(graph, name, tshape, tgt, resources));
            break;
          }
          case 1: {
            TF_ASSIGN_OR_RETURN(
                out, AddRightMatMul(graph, name, tshape, tgt, resources));
            break;
          }
          default:
            return xla::FailedPrecondition(
                "invalid operand for tensor allocation on %s",
                src.first->name().c_str());
        }
        break;
      }
      case HloOpcode::kDynamicSlice: {
        if (target->second.input_index == 0) {
          TF_ASSIGN_OR_RETURN(
              out, AddDynamicSliceTensor(graph, name, tshape,
                                         target->second.tgt->shape()));
        } else {
          TF_ASSIGN_OR_RETURN(out, AddPlainTensor(graph, name, tshape));
        }
        break;
      }
      case HloOpcode::kDynamicUpdateSlice: {
        if (target->second.input_index == 0) {
          TF_ASSIGN_OR_RETURN(
              out, AddDynamicSliceTensor(graph, name, tshape,
                                         target->second.tgt->shape()));
        } else {
          TF_ASSIGN_OR_RETURN(out, AddPlainTensor(graph, name, tshape));
        }
        break;
      }
      case HloOpcode::kCall: {
        const HloComputation* comp = tgt->to_apply();
        if (IsPopOpsCall(comp)) {
          auto end = comp->name().find('.');
          std::string name = comp->name().substr(8, end - 8);
          if (name == "depthwise_conv") {
            const HloInstruction* conv_inst = comp->root_instruction();
            switch (target->second.input_index) {
              case 0: {
                TF_ASSIGN_OR_RETURN(
                    out, AddConvolutionInput(graph, name, tgt, conv_inst,
                                             resources));
                break;
              }
              case 1: {
                TF_ASSIGN_OR_RETURN(
                    out, AddConvolutionWeights(graph, name, tgt, conv_inst,
                                               resources));
                break;
              }
              default:
                return xla::FailedPrecondition(
                    "invalid operand for tensor allocation on %s",
                    src.first->name().c_str());
            }
          } else {
            return xla::FailedPrecondition(
                "Unknown poplibs fusion for tensor %s: %s",
                src.first->name().c_str(), name.c_str());
          }
        } else {
          TF_ASSIGN_OR_RETURN(out, AddPlainTensor(graph, name, tshape));
        }
        break;
      }
      default:
        return xla::FailedPrecondition("Unknown tensor target for %s: %s",
                                       src.first->name().c_str(),
                                       tgt->name().c_str());
    }

    TF_ASSIGN_OR_RETURN(out,
                        PathTransform(graph, out, target->second.forward_path,
                                      target->second.backward_path));
  } else {
    TF_ASSIGN_OR_RETURN(out, AddPlainTensor(graph, name, shape));
  }
  return out;
}

template <typename TYPE>
static void AddConstantTensor(poplar::Graph& graph, const xla::Literal& literal,
                              const xla::Shape& shape, const poplar::Type& type,
                              poplar::Tensor& tensor) {
  int64 num_elements(ShapeUtil::ElementsIn(literal.shape()));
  std::vector<std::size_t> dim = PoplarShapeFromXlaShape(shape);
  const TYPE* data(static_cast<const TYPE*>(literal.untyped_data()));

  if (num_elements == 0) {
    tensor = graph.addConstant(type, {0}, (TYPE)0);
  } else if (num_elements == 1) {
    tensor = graph.addConstant(type, dim, data[0]);
  } else {
    tensor = graph.addConstant(type, dim, data);
  }

  tensor = ConvertToDeviceLayout(shape, tensor);
}

static void AddFp16ConstantTensor(poplar::Graph& graph,
                                  const xla::Literal& literal,
                                  const xla::Shape& shape,
                                  const poplar::Type& type,
                                  poplar::Tensor& tensor) {
  int64 num_elements(ShapeUtil::ElementsIn(literal.shape()));
  std::vector<std::size_t> dim = PoplarShapeFromXlaShape(shape);
  const uint16_t* data(static_cast<const uint16_t*>(literal.untyped_data()));

  if (num_elements == 0) {
    tensor = graph.addConstantHalf(type, {0}, (uint16_t)0);
  } else if (num_elements == 1) {
    tensor = graph.addConstantHalf(type, dim, data[0]);
  } else {
    tensor = graph.addConstantHalf(type, dim, (uint16_t*)data);
  }

  tensor = ConvertToDeviceLayout(shape, tensor);
}

static void Add64BitConstantTensor(poplar::Graph& graph,
                                   const xla::Literal& literal,
                                   const xla::Shape& shape,
                                   const poplar::Type& type,
                                   poplar::Tensor& tensor) {
  int64 num_elements(ShapeUtil::ElementsIn(literal.shape()));
  std::vector<std::size_t> dim = PoplarShapeFromXlaShape(shape);
  const void* data(static_cast<const void*>(literal.untyped_data()));

  std::vector<char> converted =
      ConvInt64ToInt32(data, num_elements * sizeof(int64), 0);

  const int32* data32 = reinterpret_cast<const int32*>(converted.data());

  if (num_elements == 0) {
    tensor = graph.addConstant(type, {0}, (int32)0);
  } else if (num_elements == 1) {
    tensor = graph.addConstant(type, dim, data32[0]);
  } else {
    tensor = graph.addConstant(type, dim, data32);
  }
}

template <typename TYPE>
static void SetInitialTensorValue(poplar::Graph& graph, poplar::Tensor& tensor,
                                  const xla::Literal& literal) {
  const TYPE* data(static_cast<const TYPE*>(literal.untyped_data()));
  size_t element_count = literal.element_count();
  poplar::ArrayRef<TYPE> array(data, element_count);
  graph.setInitialValue<TYPE>(tensor, array);
}

static void SetFp16InitialTensorValue(poplar::Graph& graph,
                                      poplar::Tensor& tensor,
                                      const xla::Literal& literal) {
  const uint16_t* data(static_cast<const uint16_t*>(literal.untyped_data()));
  size_t element_count = literal.element_count();
  poplar::ArrayRef<uint16_t> array(data, element_count);
  graph.setInitialValueHalf(tensor, array);
}

static void Set64BitInitialTensorValue(poplar::Graph& graph,
                                       poplar::Tensor& tensor,
                                       const xla::Literal& literal) {
  size_t element_count = literal.element_count();
  const void* data(static_cast<const void*>(literal.untyped_data()));
  std::vector<char> converted =
      ConvInt64ToInt32(data, element_count * sizeof(int64), 0);

  int32* data32 = reinterpret_cast<int32*>(converted.data());
  poplar::ArrayRef<int32> array(data32, element_count);
  graph.setInitialValue<int>(tensor, array);
}

StatusOr<poplar::Tensor> AddConstantTensor(poplar::Graph& graph,
                                           const TensorSource& src,
                                           const xla::Shape& shape,
                                           const xla::Literal& literal,
                                           CompilerResources& resources,
                                           const TensorMap& tensor_map) {
  poplar::Tensor tensor;

  poplar::Type type;
  TF_ASSIGN_OR_RETURN(type, PoplarDataType(literal.shape()));

  if (ShapeUtil::ElementsIn(literal.shape()) > 32) {
    TF_ASSIGN_OR_RETURN(tensor,
                        AddTensor(graph, src, shape, resources, tensor_map));
    switch (literal.shape().element_type()) {
      case PRED:
        SetInitialTensorValue<bool>(graph, tensor, literal);
        break;
      case S32:
      case U32:
        SetInitialTensorValue<int>(graph, tensor, literal);
        break;
      case U64:
      case S64:
        Set64BitInitialTensorValue(graph, tensor, literal);
        break;
      case F16:
        SetFp16InitialTensorValue(graph, tensor, literal);
        break;
      case F32:
        SetInitialTensorValue<float>(graph, tensor, literal);
        break;
      default:
        // The unsupported cases were caught in the call to PoplarDataType above
        break;
    }
    return ConvertToDeviceLayout(shape, tensor);
  } else {
    switch (literal.shape().element_type()) {
      case PRED:
        AddConstantTensor<bool>(graph, literal, shape, type, tensor);
        break;
      case S32:
      case U32:
        AddConstantTensor<int>(graph, literal, shape, type, tensor);
        break;
      case U64:
      case S64:
        Add64BitConstantTensor(graph, literal, shape, type, tensor);
        break;
      case F16:
        AddFp16ConstantTensor(graph, literal, shape, type, tensor);
        break;
      case F32:
        AddConstantTensor<float>(graph, literal, shape, type, tensor);
        break;
      default:
        // The unsupported cases were caught in the call to PoplarDataType above
        break;
    }

    std::vector<std::size_t> dim = PoplarShapeFromXlaShape(shape);
    return tensor.reshape(dim);
  }
}

template <typename TYPE>
static Literal GetIotaLiteral(int64 len) {
  std::vector<TYPE> data(len);
  std::iota(data.begin(), data.end(), 0);
  return LiteralUtil::CreateR1<TYPE>(data);
}

StatusOr<poplar::Tensor> AddIotaTensor(poplar::Graph& graph,
                                       const TensorSource& src,
                                       const xla::Shape& shape,
                                       int64 iota_dimension,
                                       CompilerResources& resources,
                                       const TensorMap& tensor_map) {
  poplar::Type type;
  TF_ASSIGN_OR_RETURN(type, PoplarDataType(shape));

  int64 len = shape.dimensions(iota_dimension);
  Literal literal;

  switch (shape.element_type()) {
    case S32:
    case U32: {
      literal = GetIotaLiteral<int>(len);
      break;
    }
    case F32: {
      literal = GetIotaLiteral<float>(len);
      break;
    }
    default:
      return xla::FailedPrecondition("unsupported primitive type for iota: %s",
                                     PrimitiveType_Name(shape.element_type()));
  }
  poplar::Tensor t;
  auto iota_shape = ShapeUtil::MakeShape(shape.element_type(),
                                         {shape.dimensions(iota_dimension)});
  TF_ASSIGN_OR_RETURN(t, AddConstantTensor(graph, src, iota_shape, literal,
                                           resources, tensor_map));
  return BroadcastTensor(t, shape, {iota_dimension});
}

template <typename T>
poplar::Tensor TileTensor(const T& multiples, const poplar::Tensor& in) {
  poplar::Tensor out = in;
  for (unsigned d = 0; d < multiples.size(); d++) {
    int m = multiples[d];
    out = out.broadcast(m, d);
  }
  return out;
}

template poplar::Tensor TileTensor<tensorflow::BCast::Vec>(
    const tensorflow::BCast::Vec&, const poplar::Tensor&);

template poplar::Tensor TileTensor<std::vector<std::size_t>>(
    const std::vector<std::size_t>&, const poplar::Tensor&);

StatusOr<poplar::Tensor> PadTensor(const PaddingConfig& cfg,
                                   const poplar::Tensor& in,
                                   const poplar::Tensor& pad) {
  if (pad.numElements() != 1) {
    return xla::FailedPrecondition(
        "PadTensor: pad tensor is not single valued");
  }

  poplar::Tensor p(pad.reshape(std::vector<std::size_t>(in.rank(), 1)));

  poplar::Tensor out = in;
  for (unsigned d = 0; d < in.rank(); d++) {
    std::vector<std::size_t> shape(out.shape());

    if (cfg.dimensions(d).interior_padding() > 0 && shape[d] > 0) {
      shape[d] = cfg.dimensions(d).interior_padding();
      poplar::Tensor padded = TileTensor(shape, p);
      poplar::Tensor interleaved = out.slice(0, 1, d);
      for (unsigned int slice = 1; slice < out.dim(d); slice++) {
        interleaved = poplar::concat(interleaved, padded, d);
        interleaved =
            poplar::concat(interleaved, out.slice(slice, slice + 1, d), d);
      }
      out = interleaved;
    }

    if (cfg.dimensions(d).edge_padding_low() > 0) {
      shape[d] = cfg.dimensions(d).edge_padding_low();
      poplar::Tensor padded = TileTensor(shape, p);
      out = poplar::concat(padded, out, d);
    }

    if (cfg.dimensions(d).edge_padding_high() > 0) {
      shape[d] = cfg.dimensions(d).edge_padding_high();
      poplar::Tensor padded = TileTensor(shape, p);
      out = poplar::concat(out, padded, d);
    }
  }

  return out;
}

StatusOr<poplar::Tensor> ReverseTensor(const poplar::Tensor& in,
                                       const std::vector<int64>& dimensions) {
  poplar::Tensor out = in;
  if (in.numElements() > 0) {
    for (int64 d : dimensions) {
      out = out.reverse(d);
    }
  }
  return out;
}

StatusOr<poplar::Tensor> BroadcastTensor(const poplar::Tensor& in,
                                         const xla::Shape& out,
                                         const std::vector<int64>& dimensions) {
  if (PoplarShapeMatchesXLAShape(in, out)) {
    return in;
  }

  tensorflow::BCast::Vec bcast_shape =
      convert_array<tensorflow::BCast::Vec>(out.dimensions());

  tensorflow::BCast::Vec tensor_shape(ShapeUtil::Rank(out), 1);
  if (dimensions.size() > 0) {
    for (size_t d = 0; d < dimensions.size(); d++) {
      tensor_shape[dimensions[d]] = in.dim(d);
    }
  } else {
    for (size_t d = 0; d < in.rank(); d++) {
      tensor_shape[d] = in.dim(d);
    }
  }

  tensorflow::BCast bcast(tensor_shape, bcast_shape);
  if (!bcast.IsValid()) {
    return xla::FailedPrecondition("Incompatible broadcast from (%s) to (%s)",
                                   Join(tensor_shape, ",").c_str(),
                                   Join(bcast_shape, ",").c_str());
  }

  poplar::Tensor o = in;
  o = in.reshape(convert_array<std::vector<size_t>>(bcast.x_reshape()));
  o = TileTensor(bcast.x_bcast(), o);
  return o.reshape(PoplarShapeFromXlaShape(out));
}

bool PoplarShapeMatchesXLAShape(const poplar::Tensor& tensor,
                                const xla::Shape& shape) {
  if (tensor.rank() != ShapeUtil::Rank(shape)) return false;
  for (size_t d = 0; d < tensor.rank(); d++) {
    if (tensor.dim(d) != (unsigned)shape.dimensions(d)) return false;
  }

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
