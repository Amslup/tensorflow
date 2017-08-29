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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_INPLACE_FINDER_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_INPLACE_FINDER_H_

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"

namespace xla {

class HloModule;
class HloInstruction;

namespace poplarplugin {

using InplaceInstructions = std::set<const HloInstruction*>;
using InplaceRoute = std::vector<HloInstruction*>;

/**
 * This class finds all instructions that explicitly add tensors to the
 * graph.  For each one of them, it locates the downstream consumers of that
 * tensor, and if any of those instructions require a specific tensor allocation
 * method (e.g. convolution), then it notes the downstream instruction
 */
class InplaceFinder {
public:
  InplaceFinder() {}

  ~InplaceFinder() = default;

  Status FindInplaceInstructions(HloModule* module);

  InplaceInstructions inplace_instructions;

private:
  void RouteFinder(HloInstruction* inst);

  std::multimap<HloInstruction*, InplaceRoute> routes;

  InplaceRoute current_route;
};

}
}

#endif
