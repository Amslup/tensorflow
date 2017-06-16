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

#include "tensorflow/compiler/plugin/poplar/driver/compiler.h"
#include "tensorflow/compiler/plugin/poplar/driver/conversions.h"
#include "tensorflow/compiler/plugin/poplar/driver/executable.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace sep = ::perftools::gputools::poplarplugin;

namespace xla {
namespace poplarplugin {

/* The compilation process produces an executable which contains a map of
 * which input tensors are also outputs.  This test checks that this map is
 * correct */

class GraphCompileIoMapTest : public HloTestBase {
public:
  const sep::OutputMap& GetMap(PoplarExecutable* e) {
    return e->output_map_;
  }

  const sep::ConversionList& GetInputList(PoplarExecutable* e) {
    return e->input_convertors_;
  }

  const sep::ConversionList& GetOutputList(PoplarExecutable* e) {
    return e->output_convertors_;
  }
};

namespace {

TEST_F(GraphCompileIoMapTest, NoShared) {
  Shape image_shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
          HloInstruction::CreateParameter(0, image_shape, "input1"));
  auto in2 = builder.AddInstruction(
          HloInstruction::CreateParameter(1, image_shape, "input2"));
  auto add = builder.AddInstruction(
          HloInstruction::CreateBinary(image_shape, HloOpcode::kAdd, in1, in2));
  builder.AddInstruction(
        HloInstruction::CreateTuple({add}));

  auto computation = builder.Build();


  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));

  PoplarCompiler compiler;
  auto hlo_dumper = [](const HloModule& module, const string& label) {};

  std::unique_ptr<Executable> executable =
    compiler.Compile(std::move(hlo_module),
                     hlo_dumper,
                     nullptr).ConsumeValueOrDie();

  PoplarExecutable* e = static_cast<PoplarExecutable*>(executable.get());
  EXPECT_EQ(0, GetMap(e).size());
}

TEST_F(GraphCompileIoMapTest, Input1Shared) {
  Shape image_shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
          HloInstruction::CreateParameter(0, image_shape, "input1"));
  auto in2 = builder.AddInstruction(
          HloInstruction::CreateParameter(1, image_shape, "input2"));
  auto add = builder.AddInstruction(
          HloInstruction::CreateBinary(image_shape, HloOpcode::kAdd, in1, in2));
  builder.AddInstruction(
          HloInstruction::CreateTuple({add}));

  OpMetadata metadata;
  metadata.set_op_name("grad%1");
  metadata.set_op_type("ResourceApplyGradientDescent");
  add->set_metadata(metadata);

  auto computation = builder.Build();


  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));

  PoplarCompiler compiler;
  auto hlo_dumper = [](const HloModule& module, const string& label) {};

  std::unique_ptr<Executable> executable =
          compiler.Compile(std::move(hlo_module),
                           hlo_dumper,
                           nullptr).ConsumeValueOrDie();

  PoplarExecutable* e = static_cast<PoplarExecutable*>(executable.get());
  EXPECT_EQ(1, GetMap(e).size());
  EXPECT_EQ(0, GetMap(e).at(0));
}

TEST_F(GraphCompileIoMapTest, Input2Shared) {
  Shape image_shape = ShapeUtil::MakeShape(F32, {1, 4, 4, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
          HloInstruction::CreateParameter(0, image_shape, "input1"));
  auto in2 = builder.AddInstruction(
          HloInstruction::CreateParameter(1, image_shape, "input2"));
  auto add = builder.AddInstruction(
          HloInstruction::CreateBinary(image_shape, HloOpcode::kAdd, in2, in1));
  builder.AddInstruction(
          HloInstruction::CreateTuple({add}));

  OpMetadata metadata;
  metadata.set_op_name("grad%1");
  metadata.set_op_type("ResourceApplyGradientDescent");
  add->set_metadata(metadata);

  auto computation = builder.Build();


  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));

  PoplarCompiler compiler;
  auto hlo_dumper = [](const HloModule& module, const string& label) {};

  std::unique_ptr<Executable> executable =
          compiler.Compile(std::move(hlo_module),
                           hlo_dumper,
                           nullptr).ConsumeValueOrDie();

  PoplarExecutable* e = static_cast<PoplarExecutable*>(executable.get());
  EXPECT_EQ(1, GetMap(e).size());
  EXPECT_EQ(1, GetMap(e).at(0));
}

TEST_F(GraphCompileIoMapTest, NoConversion) {
  Shape image_shape = ShapeUtil::MakeShape(S32, {2, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
          HloInstruction::CreateParameter(0, image_shape, "input1"));
  auto in2 = builder.AddInstruction(
          HloInstruction::CreateParameter(1, image_shape, "input2"));
  auto add = builder.AddInstruction(
          HloInstruction::CreateBinary(image_shape, HloOpcode::kAdd, in1, in2));
  builder.AddInstruction(
          HloInstruction::CreateTuple({add}));

  auto computation = builder.Build();


  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));

  PoplarCompiler compiler;
  auto hlo_dumper = [](const HloModule& module, const string& label) {};

  std::unique_ptr<Executable> executable =
          compiler.Compile(std::move(hlo_module),
                           hlo_dumper,
                           nullptr).ConsumeValueOrDie();

  PoplarExecutable* e = static_cast<PoplarExecutable*>(executable.get());
  EXPECT_EQ(2, GetInputList(e).size());
  EXPECT_EQ(nullptr, GetInputList(e)[0]);
  EXPECT_EQ(nullptr, GetInputList(e)[1]);
  EXPECT_EQ(1, GetOutputList(e).size());
  EXPECT_EQ(nullptr, GetOutputList(e)[0]);
}

TEST_F(GraphCompileIoMapTest, Int64Conversion) {
  Shape image_shape = ShapeUtil::MakeShape(S64, {2, 2});

  auto builder = HloComputation::Builder(TestName());
  auto in1 = builder.AddInstruction(
          HloInstruction::CreateParameter(0, image_shape, "input1"));
  auto in2 = builder.AddInstruction(
          HloInstruction::CreateParameter(1, image_shape, "input2"));
  auto add = builder.AddInstruction(
          HloInstruction::CreateBinary(image_shape, HloOpcode::kAdd, in1, in2));
  builder.AddInstruction(
          HloInstruction::CreateTuple({add}));

  auto computation = builder.Build();


  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));

  PoplarCompiler compiler;
  auto hlo_dumper = [](const HloModule& module, const string& label) {};

  std::unique_ptr<Executable> executable =
          compiler.Compile(std::move(hlo_module),
                           hlo_dumper,
                           nullptr).ConsumeValueOrDie();

  PoplarExecutable* e = static_cast<PoplarExecutable*>(executable.get());
  EXPECT_EQ(2, GetInputList(e).size());
  EXPECT_EQ(&ConvInt64ToInt32, GetInputList(e)[0]);
  EXPECT_EQ(&ConvInt64ToInt32, GetInputList(e)[1]);
  EXPECT_EQ(1, GetOutputList(e).size());
  EXPECT_EQ(&ConvInt32ToInt64, GetOutputList(e)[0]);
}

}
}
}
