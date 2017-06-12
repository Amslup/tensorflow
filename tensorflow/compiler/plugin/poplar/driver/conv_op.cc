#include <algorithm>

#include "tensorflow/compiler/plugin/poplar/driver/vertex_templates.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

#include <popstd/ActivationMapping.hpp>
#include <popconv/Convolution.hpp>

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>

namespace xla {
namespace poplarplugin {


port::StatusOr <poplar::program::Program>
CreateConv2D(poplar::Graph &graph,
             CompilerResources& res,
             const HloInstruction *inst,
             const xla::Shape &output_shape,
             TensorMap &tensor_map) {

  // Find the input tensor
  poplar::Tensor in;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 0, 0));

  // Find the input tensor
  poplar::Tensor kernel;
  TF_ASSIGN_OR_RETURN(kernel, FindInstructionInput(tensor_map, inst, 1, 0));

  popconv::ConvOptions opts;
  opts.cache = &res.convolution_cache;

  const Window& window(inst->window());

  if (in.rank() != 4 || kernel.rank() != 4) {
    return port::Status(
            port::error::FAILED_PRECONDITION,
            port::StrCat("Poplar supports 2D convolution only: ", inst->name()));
  }

  if (window.dimensions().size() != 2) {
    return port::Status(
            port::error::FAILED_PRECONDITION,
            port::StrCat("Invalid window dimension count on ", inst->name()));
  }

  if (window.dimensions(0).window_dilation() != 1 ||
      window.dimensions(1).window_dilation() != 1) {
    return port::Status(
            port::error::FAILED_PRECONDITION,
            port::StrCat("Window dilation not supported on ", inst->name()));
  }

  const std::string& dtype(in.elementType());

  const std::vector<size_t> &input_dims = in.shape();
  const std::vector<size_t> &kernel_dims = kernel.shape();

  const ConvolutionDimensionNumbers& dims(inst->convolution_dimension_numbers());
  unsigned int n_b = input_dims[dims.batch_dimension()];
  unsigned int n_i = input_dims[dims.feature_dimension()];
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

  popconv::ConvParams params(dtype,
                             {n_b, n_y, n_x, n_i},
                             {f_y, f_x, n_o, n_i},
                             {s_y, s_x},
                             {pl_y, pl_x},
                             {pu_y, pu_x},
                             {di_y, di_x});

  poplar::program::Sequence prog;

  // TODO : create these at original tensor creation time
  in = in.dimShuffle({
    (unsigned int)dims.batch_dimension(),
    (unsigned int)dims.spatial_dimensions(0),
    (unsigned int)dims.spatial_dimensions(1),
    (unsigned int)dims.feature_dimension()
  });
  poplar::Tensor conv_in = popconv::createInput(graph, params, "", opts);
  prog.add(poplar::program::Copy(in, conv_in));

  kernel = kernel.dimShuffle({
    (unsigned int)dims.kernel_spatial_dimensions(0),
    (unsigned int)dims.kernel_spatial_dimensions(1),
    (unsigned int)dims.kernel_output_feature_dimension(),
    (unsigned int)dims.kernel_input_feature_dimension()
  });
  poplar::Tensor conv_kernel = popconv::createWeights(graph, params, "", opts);
  prog.add(poplar::program::Copy(kernel, conv_kernel));

  // TODO If the weight input and output channels are reversed, then we can use
  // TODO the poplar feature the reorder them internally. - this would require
  // TODO the reverse op to be fused with the conv op in the backward pass.

  // Add the convolution
  poplar::Tensor out = popconv::convolution(graph, conv_in, conv_kernel, params,
                                            false, prog, "", opts);

  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));

  return prog;
}

}
}
