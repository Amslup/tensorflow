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

#include <dlfcn.h>
#include <stdlib.h>
#include <unistd.h>
#include <fstream>

#include "tensorflow/compiler/plugin/poplar/driver/compiler.h"

#include "tensorflow/compiler/plugin/poplar/driver/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/casts_elimination.h"
#include "tensorflow/compiler/plugin/poplar/driver/commutative_instruction_reorder_operands.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/computation_flattener.h"
#include "tensorflow/compiler/plugin/poplar/driver/entry_visitor.h"
#include "tensorflow/compiler/plugin/poplar/driver/executable.h"
#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/expression_outliner.h"
#include "tensorflow/compiler/plugin/poplar/driver/fuse_max_pool.h"
#include "tensorflow/compiler/plugin/poplar/driver/fuse_ops_early.h"
#include "tensorflow/compiler/plugin/poplar/driver/fuse_ops_late.h"
#include "tensorflow/compiler/plugin/poplar/driver/fuse_wide_const.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform_id.h"
#include "tensorflow/compiler/plugin/poplar/driver/scheduler.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/update_op_dependencies.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"
#include "tensorflow/compiler/plugin/poplar/driver/while_loop_condition_simplify.h"
#include "tensorflow/compiler/plugin/poplar/driver/wide_const_finder.h"

#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"
#include "tensorflow/compiler/xla/service/batchnorm_expander.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/dot_decomposer.h"
#include "tensorflow/compiler/xla/service/gather_expander.h"
#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/hlo_subcomputation_unification.h"
#include "tensorflow/compiler/xla/service/hlo_tfgraph_builder.h"
#include "tensorflow/compiler/xla/service/inliner.h"
#include "tensorflow/compiler/xla/service/reshape_mover.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/service/zero_sized_hlo_elimination.h"
#include "tensorflow/compiler/xla/status_macros.h"

#include "tensorflow/stream_executor/lib/initialize.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/strings/strcat.h"

#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>

#include <poplin/codelets.hpp>
#include <popnn/codelets.hpp>
#include <popops/codelets.hpp>
#include <poprand/codelets.hpp>

namespace se = ::stream_executor;

using ::tensorflow::strings::StrCat;

namespace xla {
namespace poplarplugin {

static std::string GetPathToGraphProgFile() {
  Dl_info dlInfo;
  static const void* dummy;
  if (dladdr(&dummy, &dlInfo)) {
    std::string path(dlInfo.dli_fname);
    path = path.substr(0, path.find_last_of('/') + 1);
    path = path + "../compiler/plugin/poplar/tf.gp";
    if (access(path.c_str(), R_OK) != -1) {
      return path;
    }
  }

  // This is for unit tests
  {
    char buf[256];
    getcwd(buf, 255);
    std::string path(buf);
    path = path + "/tensorflow/compiler/plugin/poplar/tf.gp";
    if (access(path.c_str(), R_OK) != -1) {
      return path;
    }
  }

  return "";
}

static bool GetConstantOutput(const HloInstruction* root, const Shape& layout,
                              std::vector<Literal>& result) {
  if (root->opcode() == HloOpcode::kConstant) {
    auto literal = root->literal().Relayout(layout);
    result.emplace_back(std::move(literal));
    return true;
  }
  if (root->opcode() == HloOpcode::kTuple) {
    for (unsigned int i = 0; i < root->operand_count(); i++) {
      auto& sub_shape = layout.tuple_shapes(i);
      if (!GetConstantOutput(root->operand(i), sub_shape, result)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

static std::string SerializeComputationToGraphDef(const HloComputation& comp) {
  std::string buffer;
  hlo_graph_dumper::HloTfGraphBuilder builder;
  TF_CHECK_OK(builder.AddComputation(comp));
  builder.GetGraphDef().SerializeToString(&buffer);
  return buffer;
}

StatusOr<std::unique_ptr<HloModule>> PoplarCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module,
    perftools::gputools::StreamExecutor* executor,
    DeviceMemoryAllocator* device_allocator) {
  return std::move(module);
}

StatusOr<std::unique_ptr<Executable>> PoplarCompiler::RunBackend(
    std::unique_ptr<HloModule> module,
    perftools::gputools::StreamExecutor* stream_exec,
    DeviceMemoryAllocator* device_allocator) {
  if (stream_exec == nullptr) {
    return tensorflow::errors::Unknown(
        "NULL stream pointer in poplar compiler");
  }

  VLOG(1) << "Begin compilation: " << module->name() << " for ordinal  "
          << stream_exec->device_ordinal();

  PoplarExecutor* poplarExecutor(
      static_cast<PoplarExecutor*>(stream_exec->implementation()));

  std::unique_ptr<HloProfileIndexMap> profile_index_map;
  std::unique_ptr<HloProfilePrinterData> profile_printer;
  if (module->config().hlo_profiling_enabled()) {
    HloCostAnalysis cost_analysis(ShapeSizeBytesFunction());
    profile_index_map = absl::make_unique<HloProfileIndexMap>(*module);
    profile_printer =
        CreateHloProfilePrinterData(*profile_index_map, cost_analysis);
  }

  std::string filename;
  if (poplarExecutor->HaveExecutableCache()) {
    filename = poplarExecutor->CachedExecutableFilename(*module);

    if (poplarExecutor->HaveCachedExecutable(filename)) {
      PoplarExecutable* poplar_executable;
      TF_ASSIGN_OR_RETURN(poplar_executable,
                          PoplarExecutable::Deserialize(
                              std::move(module), std::move(profile_printer),
                              std::move(profile_index_map), filename));

      std::unique_ptr<Executable> executable;
      executable.reset(poplar_executable);

      return std::move(executable);
    }
  }

  const poplar::Device& dev = poplarExecutor->GetPoplarDevice();

  std::lock_guard<std::mutex> g(static_mu_);

  poplar::Graph graph(dev);
  graph.addCodelets(GetPathToGraphProgFile());
  poplin::addCodelets(graph);
  popnn::addCodelets(graph);
  popops::addCodelets(graph);
  poprand::addCodelets(graph);

  uint64 start_micros = tensorflow::Env::Default()->NowMicros();

  uint64 seed = module->config().seed();
  if (seed == 0) {
    seed = tensorflow::random::New64();
  }

  CompilerResources resources(seed + 1, poplarExecutor->GetRandomGenMode());
  resources.annotations.num_resource_inputs =
      module->config().resource_input_count();
  resources.annotations.num_resource_outputs =
      module->config().resource_update_count();

  {
    HloPassPipeline pipeline("IPU");
    pipeline.AddPass<BatchNormExpander>(true, true, true);
    pipeline.AddPass<GatherExpander>();
    pipeline.AddPass<DotDecomposer>();
    pipeline.AddPass<HloPassFix<FuseOpsEarly>>(resources.annotations);
    pipeline.AddPass<HloCSE>(false);
    pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(
        false, [](const Shape&, const Shape&) { return false; }, false, false);
    pipeline.AddPass<ReshapeMover>();
    pipeline.AddPass<Inliner>();
    pipeline.AddPass<HloPassFix<AlgebraicSimplifier>>(
        false, [](const Shape&, const Shape&) { return false; }, false, false);
    pipeline.AddPass<ZeroSizedHloElimination>();
    pipeline.AddPass<ComputationFlattener>();
    pipeline.AddPass<TupleSimplifier>(true);
    // pipeline.AddPass<WhileLoopSimplifier>();
    // pass.AddPass<ConditionalSimplifier>();
    pipeline.AddPass<HloConstantFolding>();
    pipeline.AddPass<HloPassFix<CastsElimination>>(resources.annotations);
    pipeline.AddPass<HloCSE>(true);
    pipeline.AddPass<WideConstFinder>();
    pipeline.AddPass<CommutativeInstructionReorderOperands>();
    pipeline.AddPass<ConvolutionClassifier>(resources.annotations);
    pipeline.AddPass<HloDCE>();
    pipeline.AddPass<HloPassFix<FuseMaxPool>>(resources.annotations);
    pipeline.AddPass<HloPassFix<FuseOpsLate>>(resources.annotations);
    pipeline.AddPass<FuseWideConst>(resources.annotations);
    pipeline.AddPass<InplaceFinder>(resources.annotations);
    pipeline.AddPass<UpdateOpDependenctOrdering>(resources.annotations);
    pipeline.AddPass<ExpressionOutliner>(resources.annotations);
    pipeline.AddPass<HloSubcomputationUnification>();
    pipeline.AddPass<WhileLoopConditionSimplify>();
    pipeline.AddPass<HloDCE>();
    pipeline.AddPass<ConvolutionClassifier>(resources.annotations);
    pipeline.AddPass<AllocationFinder>(resources.annotations);

    bool ok;
    TF_ASSIGN_OR_RETURN(ok, pipeline.Run(module.get()));
  }

  HloComputation* entry = module->entry_computation();

  if (poplarExecutor->CompilerReportingEnabled()) {
    poplarExecutor->AddEventRecord(tensorflow::IpuTraceEvent::COMPILE_BEGIN,
                                   module->name(),
                                   SerializeComputationToGraphDef(*entry), 0);
  }

  // Set layout if there isn't one
  auto comp_layout =
      module->mutable_entry_computation_layout()->mutable_result_layout();
  if (!comp_layout->LayoutIsSet()) {
    auto shape = entry->root_instruction()->shape();
    TF_CHECK_OK(comp_layout->CopyLayoutFromShape(shape));
  }

  VLOG(1) << "Compiling main computation " << entry->name();
  XLA_VLOG_LINES(1, entry->ToString());

  std::vector<const HloInstruction*> instruction_order;
  TF_ASSIGN_OR_RETURN(instruction_order, Scheduler::schedule(entry));

  uint64 num_inputs = entry->num_parameters();
  uint64 num_outputs = CountShapes(entry->root_instruction()->shape());

  std::shared_ptr<poplar::Engine> engine;
  std::vector<poplar::program::Program> progs;
  std::vector<Literal> constant_output;

  EntryVisitor visitor(graph, resources, num_inputs, num_outputs);

  if (!GetConstantOutput(entry->root_instruction(), comp_layout->shape(),
                         constant_output)) {
    try {
      TF_RETURN_IF_ERROR(entry->AcceptOrdered(&visitor, instruction_order));
    } catch (std::logic_error e) {
      return tensorflow::errors::Unknown(StrCat("[Poplar Compile] ", e.what()));
    }

    progs.push_back(visitor.sequence);
    progs.push_back(visitor.GetHostToDevice());
    progs.push_back(visitor.GetDeviceToHost());

    char* vertex_filename = getenv("TF_DUMP_VERTEX_GRAPH");
    if (vertex_filename) {
      std::ofstream stream(vertex_filename);
      graph.outputVertexGraph(stream, progs);
    }

    if (visitor.AreAllOutputsParameters()) {
      VLOG(1) << "Skip engine compilation - all outputs are inputs";
    } else {
      try {
        VLOG(1) << "Compile engine " << module->name();

        auto opts = poplarExecutor->GetOptionsFlags();
        engine.reset(new poplar::Engine(graph, progs, opts));
      } catch (std::logic_error e) {
        return tensorflow::errors::Unknown(
            StrCat("[Poplar Engine] ", e.what()));
      }
    }
  } else {
    VLOG(1) << "Skip engine compilation - output is constant";
  }

  if (poplarExecutor->CompilerReportingEnabled()) {
    std::stringstream stream;

    if (engine != nullptr) {
      poplar::OptionFlags opts;
      opts.set("includeVarStorageReport", "true");

      auto rep = engine->getGraphReport(opts);
      if (poplarExecutor->CompilerReportingTextFormat()) {
        rep.printSummary(stream);
      } else {
        rep.serialize(stream, poplar::SerializationFormat::JSON);
      }
    }

    uint64 duration = tensorflow::Env::Default()->NowMicros() - start_micros;

    poplarExecutor->AddEventRecord(tensorflow::IpuTraceEvent::COMPILE_END,
                                   module->name(), stream.str(), duration);
  }

  std::unique_ptr<Executable> executable;
  PoplarExecutable* poplar_executable;
  poplar_executable = new PoplarExecutable(
      std::move(module), std::move(profile_printer),
      std::move(profile_index_map), std::move(engine),
      std::move(visitor.GetOutputMap()), std::move(constant_output),
      std::move(visitor.GetParameterStreamed()),
      std::move(visitor.GetOutputStreamed()));
  executable.reset(poplar_executable);

  if (poplarExecutor->HaveExecutableCache()) {
    if (!poplarExecutor->HaveCachedExecutable(filename)) {
      TF_RETURN_IF_ERROR(
          PoplarExecutable::Serialize(*poplar_executable, filename));
    }
  }

  return std::move(executable);
}

StatusOr<std::vector<std::unique_ptr<Executable>>> PoplarCompiler::Compile(
    std::vector<std::unique_ptr<HloModule>> modules,
    std::vector<std::vector<perftools::gputools::StreamExecutor*>> stream_execs,
    DeviceMemoryAllocator* device_allocator) {
  std::vector<std::unique_ptr<Executable>> result;
  for (size_t i = 0; i < modules.size(); i++) {
    if (stream_execs[i].size() != 1) {
      return Unimplemented("Model partitioning not implemented for Poplar");
    }

    TF_ASSIGN_OR_RETURN(modules[i],
                        RunHloPasses(std::move(modules[i]), stream_execs[i][0],
                                     device_allocator));

    TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> executable,
                        RunBackend(std::move(modules[i]), stream_execs[i][0],
                                   device_allocator));

    result.push_back(std::move(executable));
  }

  return {std::move(result)};
}

StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
PoplarCompiler::CompileAheadOfTime(
    std::vector<std::unique_ptr<HloModule>> hlo_modules,
    const AotCompilationOptions& aot_options) {
  return xla::InvalidArgument("AOT compilation not supported on Poplar");
}

se::Platform::Id PoplarCompiler::PlatformId() const {
  return kPoplarPlatformId;
}

HloCostAnalysis::ShapeSizeFunction PoplarCompiler::ShapeSizeBytesFunction()
    const {
  return PoplarExecutable::ShapeSizeBytes;
}

std::mutex PoplarCompiler::static_mu_;

}  // namespace poplarplugin
}  // namespace xla

static std::unique_ptr<xla::ComputationPlacer> CreateComputationPlacer() {
  return absl::make_unique<xla::ComputationPlacer>();
}

static bool RegisterComputationPlacer() {
  xla::ComputationPlacer::RegisterComputationPlacer(
      xla::poplarplugin::kPoplarPlatformId, &CreateComputationPlacer);
  return true;
}

bool placer_registration = RegisterComputationPlacer();

static bool InitModule() {
  xla::Compiler::RegisterCompilerFactory(
      xla::poplarplugin::kPoplarPlatformId,
      []() { return absl::make_unique<xla::poplarplugin::PoplarCompiler>(); });
  return true;
}
static bool module_initialized = InitModule();
