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

#ifndef TENSORFLOW_COMPILER_POPLAR_DRIVER_VISITOR_MAP_H_
#define TENSORFLOW_COMPILER_POPLAR_DRIVER_VISITOR_MAP_H_

#include "tensorflow/compiler/poplar/driver/visitor_base.h"

namespace poplar {
class Graph;
class Tensor;
}

namespace xla {
namespace poplarplugin {

class CompilerResources;

class PoplarMapVisitor : public PoplarBaseVisitor {
public:
  PoplarMapVisitor(poplar::Graph* graph,
                   CompilerResources& res,
                   const std::vector<poplar::Tensor>& inputs,
                   const xla::Shape& shape);


  Status HandleParameter(HloInstruction* inst) override;
  Status FinishVisit(HloInstruction* inst) override;

  const Shape& GetOutputShape(HloInstruction*) const override {
    return shape_;
  }

  const std::vector<poplar::Tensor>& output() {
    return output_;
  }

private:
  std::vector<poplar::Tensor> operands_;
  std::vector<poplar::Tensor> output_;
  xla::Shape shape_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_POPLAR_DRIVER_VISITOR_MAP_H_
