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

#include "tensorflow/compiler/plugin/poplar/driver/allocation_finder.h"

#include "tensorflow/compiler/xla/service/shape_inference.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using AllocationFinderTest = HloTestBase;

static Window GetConv1Window() {
  Window window;
  for (int i = 0; i < 2; ++i) {
    auto dim = window.add_dimensions();
    dim->set_size(3);
    dim->set_stride(1);
    dim->set_padding_low(1);
    dim->set_padding_high(1);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
  return window;
}

static Window GetConv2Window() {
  Window window;
  for (int i = 0; i < 2; ++i) {
    auto dim = window.add_dimensions();
    dim->set_size(3);
    dim->set_stride(2);
    dim->set_padding_low(1);
    dim->set_padding_high(1);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
  return window;
}

static ConvolutionDimensionNumbers GetConvDimensions() {
  ConvolutionDimensionNumbers dimension;
  dimension.set_input_batch_dimension(0);
  dimension.add_input_spatial_dimensions(1);
  dimension.add_input_spatial_dimensions(2);
  dimension.set_input_feature_dimension(3);

  dimension.set_output_batch_dimension(0);
  dimension.add_output_spatial_dimensions(1);
  dimension.add_output_spatial_dimensions(2);
  dimension.set_output_feature_dimension(3);

  dimension.add_kernel_spatial_dimensions(0);
  dimension.add_kernel_spatial_dimensions(1);
  dimension.set_kernel_input_feature_dimension(2);
  dimension.set_kernel_output_feature_dimension(3);
  return dimension;
}


// Check basic parameter matching
TEST_F(AllocationFinderTest, FindBasicTensorAllocations) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {3, 3, 2, 1});

  Shape conv_shape = ShapeInference::InferConvolveShape(
        input_shape, weight_shape,
        GetConv1Window(), GetConvDimensions()).ConsumeValueOrDie();

  auto builder = HloComputation::Builder(TestName());
  auto op0 = builder.AddInstruction(
    HloInstruction::CreateParameter(0, input_shape, "op0"));
  auto op1 = builder.AddInstruction(
    HloInstruction::CreateParameter(1, input_shape, "op1"));
  auto op2 = builder.AddInstruction(
    HloInstruction::CreateParameter(2, weight_shape, "op2"));

  auto add = builder.AddInstruction(
    HloInstruction::CreateBinary(input_shape, HloOpcode::kAdd, op0, op1));

  auto conv = builder.AddInstruction(
    HloInstruction::CreateConvolve(conv_shape, op1, op2,
                                   GetConv1Window(), GetConvDimensions()));

  builder.AddInstruction(
          HloInstruction::CreateTuple({add, conv}));

  auto computation = builder.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));

  AllocationFinder finder;
  TF_EXPECT_OK(finder.CreateAllocationMap(hlo_module.get()));

  const HloInstruction* c_conv = conv;

  ASSERT_EQ(finder.tensor_allocation_map.size(), 2);
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op1,0)),
        std::make_pair(c_conv,0ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op2,0)),
        std::make_pair(c_conv,1ll));
}

// Check it goes through call sites
TEST_F(AllocationFinderTest, FindSubCompTensorAllocations) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {3, 3, 2, 1});

  Shape conv_shape = ShapeInference::InferConvolveShape(
          input_shape, weight_shape,
          GetConv1Window(), GetConvDimensions()).ConsumeValueOrDie();

  /* Create convolution sub-computation */
  auto builder_sub = HloComputation::Builder(TestName());
  auto op0_sub = builder_sub.AddInstruction(
        HloInstruction::CreateParameter(0, input_shape, "input"));
  auto op1_sub = builder_sub.AddInstruction(
        HloInstruction::CreateParameter(1, weight_shape, "weights"));

  auto conv = builder_sub.AddInstruction(
        HloInstruction::CreateConvolve(conv_shape, op0_sub, op1_sub,
                                       GetConv1Window(), GetConvDimensions()));

  auto computation_sub = builder_sub.Build();

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto op0 = builder_main.AddInstruction(
          HloInstruction::CreateParameter(0, input_shape, "op0"));
  auto op1 = builder_main.AddInstruction(
          HloInstruction::CreateParameter(1, input_shape, "op1"));
  auto op2 = builder_main.AddInstruction(
          HloInstruction::CreateParameter(2, weight_shape, "op2"));

  auto add = builder_main.AddInstruction(
          HloInstruction::CreateBinary(input_shape, HloOpcode::kAdd, op0, op1));

  auto call = builder_main.AddInstruction(
          HloInstruction::CreateCall(conv_shape, {op1, op2},
                                     computation_sub.get()));

  builder_main.AddInstruction(
          HloInstruction::CreateTuple({add, call}));

  auto computation_main = builder_main.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEmbeddedComputation(std::move(computation_sub));
  hlo_module->AddEntryComputation(std::move(computation_main));

  AllocationFinder finder;
  TF_EXPECT_OK(finder.CreateAllocationMap(hlo_module.get()));

  const HloInstruction* c_conv = conv;

  ASSERT_EQ(finder.tensor_allocation_map.size(), 4);
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op1,0)),
        std::make_pair(c_conv,0ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op2,0)),
        std::make_pair(c_conv,1ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op0_sub,0)),
        std::make_pair(c_conv,0ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op1_sub,0)),
        std::make_pair(c_conv,1ll));
}


// Check it works for multiple valid destinations (perferred one first)
TEST_F(AllocationFinderTest, FindMultiCompTensorAllocations1) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {3, 3, 2, 1});

  Shape conv1_shape = ShapeInference::InferConvolveShape(
          input_shape, weight_shape,
          GetConv1Window(), GetConvDimensions()).ConsumeValueOrDie();

  Shape conv2_shape = ShapeInference::InferConvolveShape(
          input_shape, weight_shape,
          GetConv2Window(), GetConvDimensions()).ConsumeValueOrDie();

  /* Create convolution sub-computation 1 */
  auto builder_sub1 = HloComputation::Builder(TestName());
  auto op0_sub1 = builder_sub1.AddInstruction(
          HloInstruction::CreateParameter(0, input_shape, "input"));
  auto op1_sub1 = builder_sub1.AddInstruction(
          HloInstruction::CreateParameter(1, weight_shape, "weights"));

  auto conv1 = builder_sub1.AddInstruction(
          HloInstruction::CreateConvolve(conv1_shape, op0_sub1, op1_sub1,
                                         GetConv1Window(), GetConvDimensions()));
  {
    OpMetadata metadata;
    metadata.set_op_name("Conv1");
    metadata.set_op_type("Conv2D");
    conv1->set_metadata(metadata);
  }

  auto computation_sub1 = builder_sub1.Build();

  /* Create convolution sub-computation 2 */
  auto builder_sub2 = HloComputation::Builder(TestName());
  auto op0_sub2 = builder_sub2.AddInstruction(
          HloInstruction::CreateParameter(0, input_shape, "input"));
  auto op1_sub2 = builder_sub2.AddInstruction(
          HloInstruction::CreateParameter(1, weight_shape, "weights"));

  auto conv2 = builder_sub2.AddInstruction(
          HloInstruction::CreateConvolve(conv2_shape, op0_sub2, op1_sub2,
                                         GetConv1Window(), GetConvDimensions()));
  {
    OpMetadata metadata;
    metadata.set_op_name("Conv2");
    metadata.set_op_type("Conv2DBackpropInput");
    conv2->set_metadata(metadata);
  }

  auto computation_sub2 = builder_sub2.Build();

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto op0 = builder_main.AddInstruction(
          HloInstruction::CreateParameter(0, input_shape, "op0"));
  auto op1 = builder_main.AddInstruction(
          HloInstruction::CreateParameter(1, input_shape, "op1"));
  auto op2 = builder_main.AddInstruction(
          HloInstruction::CreateParameter(2, weight_shape, "op2"));

  auto add = builder_main.AddInstruction(
          HloInstruction::CreateBinary(input_shape, HloOpcode::kAdd, op0, op1));

  auto call1 = builder_main.AddInstruction(
          HloInstruction::CreateCall(conv1_shape, {op1, op2},
                                     computation_sub1.get()));

  auto call2 = builder_main.AddInstruction(
          HloInstruction::CreateCall(conv2_shape, {op1, op2},
                                     computation_sub2.get()));

  builder_main.AddInstruction(
          HloInstruction::CreateTuple({add, call1, call2}));

  auto computation_main = builder_main.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEmbeddedComputation(std::move(computation_sub1));
  hlo_module->AddEmbeddedComputation(std::move(computation_sub2));
  hlo_module->AddEntryComputation(std::move(computation_main));

  AllocationFinder finder;
  TF_EXPECT_OK(finder.CreateAllocationMap(hlo_module.get()));

  const HloInstruction* c_conv1 = conv1;
  const HloInstruction* c_conv2 = conv2;

  ASSERT_EQ(finder.tensor_allocation_map.size(), 6);
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op1,0)),
        std::make_pair(c_conv1,0ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op2,0)),
        std::make_pair(c_conv1,1ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op0_sub1,0)),
        std::make_pair(c_conv1,0ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op1_sub1,0)),
        std::make_pair(c_conv1,1ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op0_sub2,0)),
        std::make_pair(c_conv2,0ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op1_sub2,0)),
        std::make_pair(c_conv2,1ll));
}

// Check it works for multiple valid destinations (perferred one second)
TEST_F(AllocationFinderTest, FindMultiCompTensorAllocations2) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {3, 3, 2, 1});

  Shape conv1_shape = ShapeInference::InferConvolveShape(
          input_shape, weight_shape,
          GetConv1Window(), GetConvDimensions()).ConsumeValueOrDie();

  Shape conv2_shape = ShapeInference::InferConvolveShape(
          input_shape, weight_shape,
          GetConv2Window(), GetConvDimensions()).ConsumeValueOrDie();

  /* Create convolution sub-computation 1 */
  auto builder_sub1 = HloComputation::Builder(TestName());
  auto op0_sub1 = builder_sub1.AddInstruction(
          HloInstruction::CreateParameter(0, input_shape, "input"));
  auto op1_sub1 = builder_sub1.AddInstruction(
          HloInstruction::CreateParameter(1, weight_shape, "weights"));

  auto conv1 = builder_sub1.AddInstruction(
          HloInstruction::CreateConvolve(conv1_shape, op0_sub1, op1_sub1,
                                         GetConv1Window(), GetConvDimensions()));
  {
    OpMetadata metadata;
    metadata.set_op_name("Conv1");
    metadata.set_op_type("Conv2DBackpropInput");
    conv1->set_metadata(metadata);
  }


  auto computation_sub1 = builder_sub1.Build();

  /* Create convolution sub-computation 2 */
  auto builder_sub2 = HloComputation::Builder(TestName());
  auto op0_sub2 = builder_sub2.AddInstruction(
          HloInstruction::CreateParameter(0, input_shape, "input"));
  auto op1_sub2 = builder_sub2.AddInstruction(
          HloInstruction::CreateParameter(1, weight_shape, "weights"));

  auto conv2 = builder_sub2.AddInstruction(
          HloInstruction::CreateConvolve(conv2_shape, op0_sub2, op1_sub2,
                                         GetConv1Window(), GetConvDimensions()));
  {
    OpMetadata metadata;
    metadata.set_op_name("Conv2");
    metadata.set_op_type("Conv2D");
    conv2->set_metadata(metadata);
  }

  auto computation_sub2 = builder_sub2.Build();

  /* Create main computation */
  auto builder_main = HloComputation::Builder(TestName());
  auto op0 = builder_main.AddInstruction(
          HloInstruction::CreateParameter(0, input_shape, "op0"));
  auto op1 = builder_main.AddInstruction(
          HloInstruction::CreateParameter(1, input_shape, "op1"));
  auto op2 = builder_main.AddInstruction(
          HloInstruction::CreateParameter(2, weight_shape, "op2"));

  auto add = builder_main.AddInstruction(
          HloInstruction::CreateBinary(input_shape, HloOpcode::kAdd, op0, op1));

  auto call1 = builder_main.AddInstruction(
          HloInstruction::CreateCall(conv1_shape, {op1, op2},
                                     computation_sub1.get()));

  auto call2 = builder_main.AddInstruction(
          HloInstruction::CreateCall(conv2_shape, {op1, op2},
                                     computation_sub2.get()));

  builder_main.AddInstruction(
          HloInstruction::CreateTuple({add, call1, call2}));

  auto computation_main = builder_main.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEmbeddedComputation(std::move(computation_sub1));
  hlo_module->AddEmbeddedComputation(std::move(computation_sub2));
  hlo_module->AddEntryComputation(std::move(computation_main));

  AllocationFinder finder;
  TF_EXPECT_OK(finder.CreateAllocationMap(hlo_module.get()));

  const HloInstruction* c_conv1 = conv1;
  const HloInstruction* c_conv2 = conv2;

  ASSERT_EQ(finder.tensor_allocation_map.size(), 6);
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op1,0)),
          std::make_pair(c_conv2,0ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op2,0)),
          std::make_pair(c_conv2,1ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op0_sub1,0)),
          std::make_pair(c_conv1,0ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op1_sub1,0)),
          std::make_pair(c_conv1,1ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op0_sub2,0)),
          std::make_pair(c_conv2,0ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op1_sub2,0)),
          std::make_pair(c_conv2,1ll));
}

// Check it works for constants
TEST_F(AllocationFinderTest, FindConstantTensorAllocations) {
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 10, 10, 2});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {3, 3, 2, 1});

  Shape conv_shape = ShapeInference::InferConvolveShape(
          input_shape, weight_shape,
          GetConv1Window(), GetConvDimensions()).ConsumeValueOrDie();

  std::unique_ptr<Literal> literal = Literal::CreateFromShape(weight_shape);

  auto builder = HloComputation::Builder(TestName());
  auto op0 = builder.AddInstruction(
        HloInstruction::CreateParameter(0, input_shape, "op0"));
  auto op1 = builder.AddInstruction(
        HloInstruction::CreateParameter(1, input_shape, "op1"));
  auto op2 = builder.AddInstruction(
        HloInstruction::CreateConstant(std::move(literal)));

  auto add = builder.AddInstruction(
        HloInstruction::CreateBinary(input_shape, HloOpcode::kAdd, op0, op1));

  auto conv = builder.AddInstruction(
        HloInstruction::CreateConvolve(conv_shape, op1, op2,
                                       GetConv1Window(), GetConvDimensions()));

  builder.AddInstruction(
        HloInstruction::CreateTuple({add, conv}));

  auto computation = builder.Build();

  auto hlo_module = MakeUnique<HloModule>("test_module");
  hlo_module->AddEntryComputation(std::move(computation));

  AllocationFinder finder;
  TF_EXPECT_OK(finder.CreateAllocationMap(hlo_module.get()));

  const HloInstruction* c_conv = conv;

  ASSERT_EQ(finder.tensor_allocation_map.size(), 2);
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op1,0)),
        std::make_pair(c_conv,0ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(op2,0)),
        std::make_pair(c_conv,1ll));
}

// Check it goes through Tuple/Detuple pairs
TEST_F(AllocationFinderTest, CanTraverseTuples) {
  auto hlo_module = MakeUnique<HloModule>("test_module");

  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 2});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape,rhs_shape});

  auto b = HloComputation::Builder(TestName());
  auto in = b.AddInstruction(
          HloInstruction::CreateParameter(0, lhs_shape, "in"));
  auto w = b.AddInstruction(
          HloInstruction::CreateParameter(1, rhs_shape, "weight"));

  auto tuple = b.AddInstruction(
          HloInstruction::CreateTuple({in, w}));

  auto in1 = b.AddInstruction(HloInstruction::CreateGetTupleElement(
        lhs_shape, tuple, 0));
  auto w1 = b.AddInstruction(HloInstruction::CreateGetTupleElement(
        rhs_shape, tuple, 1));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot_inst = b.AddInstruction(
          HloInstruction::CreateDot(lhs_shape, in1, w1, dot_dnums));

  hlo_module->AddEntryComputation(b.Build());

  AllocationFinder finder;
  TF_EXPECT_OK(finder.CreateAllocationMap(hlo_module.get()));

  const HloInstruction* dot = dot_inst;

  ASSERT_EQ(finder.tensor_allocation_map.size(), 2);
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(in,0)),
        std::make_pair(dot,0ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(w,0)),
        std::make_pair(dot,1ll));

}

// Check it can start from tuple subshapes
TEST_F(AllocationFinderTest, CanStartOnTuples) {
  auto hlo_module = MakeUnique<HloModule>("test_module");

  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {2, 2});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({lhs_shape,rhs_shape});

  auto b = HloComputation::Builder(TestName());
  auto in = b.AddInstruction(
          HloInstruction::CreateParameter(0, tuple_shape, "tuple"));

  auto in1 = b.AddInstruction(HloInstruction::CreateGetTupleElement(
          lhs_shape, in, 0));
  auto w1 = b.AddInstruction(HloInstruction::CreateGetTupleElement(
          rhs_shape, in, 1));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  auto dot_inst = b.AddInstruction(
          HloInstruction::CreateDot(lhs_shape, in1, w1, dot_dnums));

  hlo_module->AddEntryComputation(b.Build());

  AllocationFinder finder;
  TF_EXPECT_OK(finder.CreateAllocationMap(hlo_module.get()));

  const HloInstruction* dot = dot_inst;

  ASSERT_EQ(finder.tensor_allocation_map.size(), 2);
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(in,0)),
        std::make_pair(dot,0ll));
  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(in,1)),
        std::make_pair(dot,1ll));

}

// Check it goes through while instructions
//TEST_F(AllocationFinderTest, FindWhileTensorAllocations) {
//  auto hlo_module = MakeUnique<HloModule>("test_module");
//
//  Shape counter_shape = ShapeUtil::MakeShape(S32, {});
//  Shape input_shape = ShapeUtil::MakeShape(F32, {2});
//  Shape weight_shape = ShapeUtil::MakeShape(F32, {2, 2});
//  Shape tuple_shape = ShapeUtil::MakeTupleShape(
//          {counter_shape,input_shape,weight_shape});
//
//  const HloInstruction* dot_inst;
//
//  /* Create while condition */
//  HloComputation* comp_cond;
//  {
//  auto builder_cond = HloComputation::Builder(TestName());
//  auto tuple = builder_cond.AddInstruction(
//          HloInstruction::CreateParameter(0, tuple_shape, "tuple"));
//  auto limit = builder_cond.AddInstruction(
//          HloInstruction::CreateConstant(Literal::CreateR0<int32>(10)));
//  auto c = builder_cond.AddInstruction(
//          HloInstruction::CreateGetTupleElement(ShapeUtil::MakeShape(S32, {}),
//                                                tuple, 0));
//  builder_cond.AddInstruction(HloInstruction::CreateBinary(
//        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c, limit));
//
//  comp_cond = hlo_module->AddEmbeddedComputation(builder_cond.Build());
//  }
//
//  /* Create while body */
//  HloComputation* comp_body;
//  {
//  auto builder_body = HloComputation::Builder(TestName());
//  auto tuple = builder_body.AddInstruction(
//          HloInstruction::CreateParameter(0, tuple_shape, "tuple"));
//  auto c = builder_body.AddInstruction(HloInstruction::CreateGetTupleElement(
//          counter_shape, tuple, 0));
//  auto in = builder_body.AddInstruction(HloInstruction::CreateGetTupleElement(
//          input_shape, tuple, 1));
//  auto w = builder_body.AddInstruction(HloInstruction::CreateGetTupleElement(
//          weight_shape, tuple, 2));
//  auto one = builder_body.AddInstruction(
//          HloInstruction::CreateConstant(Literal::CreateR0<int32>(1)));
//  auto new_c = builder_body.AddInstruction(HloInstruction::CreateBinary(
//          c->shape(), HloOpcode::kAdd, c, one));
//
//  DotDimensionNumbers dot_dnums;
//  dot_dnums.add_lhs_contracting_dimensions(1);
//  dot_dnums.add_rhs_contracting_dimensions(0);
//  auto new_in = builder_body.AddInstruction(
//          HloInstruction::CreateDot(input_shape, in, w, dot_dnums));
//
//  dot_inst = new_in;
//
//  builder_body.AddInstruction(
//          HloInstruction::CreateTuple({new_c, new_in, w}));
//
//  comp_body = hlo_module->AddEmbeddedComputation(builder_body.Build());
//  }
//
//
//  /* Create main computation */
//  auto builder_main = HloComputation::Builder(TestName());
//  auto c = builder_main.AddInstruction(
//          HloInstruction::CreateParameter(0, counter_shape, "counter"));
//  auto in = builder_main.AddInstruction(
//          HloInstruction::CreateParameter(1, input_shape, "in"));
//  auto w = builder_main.AddInstruction(
//          HloInstruction::CreateParameter(2, weight_shape, "weight"));
//
//  auto init = builder_main.AddInstruction(
//          HloInstruction::CreateTuple({c, in, w}));
//
//  auto main = builder_main.AddInstruction(
//          HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));
//
//  builder_main.AddInstruction(
//          HloInstruction::CreateTuple({main}));
//
//  hlo_module->AddEntryComputation(builder_main.Build());
//
//  AllocationFinder finder;
//  TF_EXPECT_OK(finder.CreateAllocationMap(hlo_module.get()));
//
//  ASSERT_EQ(finder.tensor_allocation_map.size(), 3);
//  EXPECT_EQ(finder.tensor_allocation_map.at(std::make_pair(w,0)),
//        std::make_pair(dot_inst,1ll));
//}

}
}
}
