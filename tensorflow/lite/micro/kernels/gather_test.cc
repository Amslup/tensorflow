/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

const int8_t batchdims1_input_data_i8[] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};

const int8_t batchdims1_golden_data_i8[] = {
    5,  6,  7,  8,  9,  0,  1,  2,  3,  4,  0,  1,  2,  3,  4,  5,
    6,  7,  8,  9,  20, 21, 22, 23, 24, 15, 16, 17, 18, 19, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 35, 36, 37, 38, 39, 30, 31, 32,
    33, 34, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 50, 51, 52, 53,
    54, 45, 46, 47, 48, 49, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54};

template <typename InType, typename PosType>
void TestGather(const int* input_dims, const InType* input_data,
                const int* positions_dims, const PosType* positions_data,
                int* output_dims, InType* output_data,
                const int* expected_output_dims,
                const InType* expected_output_data, const int axis = 0,
                const int batch_dims = 0) {
  TfLiteIntArray* in_dims = IntArrayFromInts(input_dims);
  TfLiteIntArray* pos_dims = IntArrayFromInts(positions_dims);
  TfLiteIntArray* out_dims = IntArrayFromInts(output_dims);
  TfLiteGatherParams params = {axis, batch_dims};

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, in_dims),
      CreateTensor(positions_data, pos_dims),
      CreateTensor(output_data, out_dims, true),
  };
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = Register_GATHER();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, &params);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  // The output tensor's data and shape have been updated by the kernel.
  TfLiteTensor* actual_output_tensor = &tensors[2];
  TfLiteIntArray* actual_output_dims = actual_output_tensor->dims;
  const int actual_output_dims_size = actual_output_dims->size;
  const int output_size = ElementCount(*actual_output_dims);
  for (int i = 0; i < output_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }

  // Compare output tensor's shape if expected_output_dims[] is provided.
  if (expected_output_dims != nullptr) {
    for (int i = 0; i < actual_output_dims_size; ++i) {
      TF_LITE_MICRO_EXPECT_EQ(expected_output_dims[i],
                              actual_output_dims->data[i]);
    }
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

// For all test functions below, dims[0] is the dimension count.
TF_LITE_MICRO_TEST(GatherOpTestShuffle) {
  const int input_dims[] = {2, 2, 2};
  const int* golden_dims = nullptr;
  const int positions_dims[] = {1, 2};
  const float input_data[] = {-2.0, 0.2, 0.7, 0.8};
  const float golden_data[] = {0.7, 0.8, -2, 0.2};
  const int32_t positions_data[] = {1, 0};
  int output_dims[] = {2, 0, 0};
  float output_data[4];
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(GatherOpTest0DIndex) {
  const int input_dims[] = {2, 2, 2};
  const int golden_dims[] = {2};
  const int positions_dims[] = {0};
  const float input_data[] = {-2.0, 0.2, 0.7, 0.8};
  const float golden_data[] = {0.7, 0.8};
  const int32_t positions_data[] = {1};
  int output_dims[] = {1, 0};
  float output_data[4];
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(GatherOpTest0DIndexWith0DResult) {
  const int input_dims[] = {1, 3};
  const int golden_dims[] = {0};
  const int positions_dims[] = {0};
  const float input_data[] = {1.0, 2.0, 3.0};
  const float golden_data[] = {2.0};
  const int32_t positions_data[] = {1};
  int output_dims[] = {1, 0};
  float output_data[3];
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(GatherOpTest1DInput1DIndex) {
  const int input_dims[] = {1, 3};
  const int golden_dims[] = {1};
  const int positions_dims[] = {1, 1};
  const float input_data[] = {1.0, 3.0, 5.0};
  const float golden_data[] = {3.0};
  const int32_t positions_data[] = {1};
  int output_dims[] = {1, 0};
  float output_data[3];
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(GatherOpTest2DIndexWith2DResult) {
  const int input_dims[] = {1, 3};
  const int golden_dims[] = {1, 2};
  const int positions_dims[] = {2, 1, 2};
  const float input_data[] = {1.0, 2.0, 3.0};
  const float golden_data[] = {2.0, 1.0};
  const int32_t positions_data[] = {1, 0};
  int output_dims[] = {2, 0, 0};
  float output_data[2];
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(FloatGatherOpTestDuplicate) {
  const int input_dims[] = {3, 1, 2, 2};
  const int* golden_dims = nullptr;
  const int positions_dims[] = {1, 2};
  const float input_data[] = {-2.0, 0.2, 0.7, 0.8};
  const float golden_data[] = {-2, 0.2, 0.7, 0.8, -2, 0.2, 0.7, 0.8};
  const int32_t positions_data[] = {0, 0};
  int output_dims[] = {3, 0, 0, 0};
  float output_data[8];
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(FloatGatherOpTestSlice) {
  const int input_dims[] = {2, 4, 1};
  const int* golden_dims = nullptr;
  const int positions_dims[] = {1, 2};
  const float input_data[] = {-2.0, 0.2, 0.7, 0.8};
  const float golden_data[] = {0.2, 0.8};
  const int32_t positions_data[] = {1, 3};
  int output_dims[] = {2, 0, 0};
  float output_data[2];
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(FloatGatherOpTestAxis1) {
  const int axis = 1;
  const int input_dims[] = {3, 1, 2, 3};
  const int golden_dims[] = {1, 2, 3};
  const int positions_dims[] = {1, 2};
  const float input_data[] = {1, 2, 3, 4, 5, 6};
  const float golden_data[] = {4, 5, 6, 1, 2, 3};
  const int32_t positions_data[] = {1, 0};
  int output_dims[] = {3, 0, 0, 0};
  float output_data[6];
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data, axis);
}

TF_LITE_MICRO_TEST(FloatGatherOpTestAxis10DIndex) {
  const int axis = 1;
  const int input_dims[] = {3, 1, 3, 2};
  const int golden_dims[] = {1, 2};
  const int positions_dims[] = {0};
  const float input_data[] = {1, 2, 3, 4, 5, 6};
  const float golden_data[] = {3, 4};
  const int32_t positions_data[] = {1};
  int output_dims[] = {2, 0, 0};
  float output_data[2];
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data, axis);
}

TF_LITE_MICRO_TEST(FloatGatherOpTestAxis1Slice) {
  const int axis = 1;
  const int input_dims[] = {3, 1, 4, 2};
  const int golden_dims[] = {1, 2, 2};
  const int positions_dims[] = {1, 2};
  const float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const float golden_data[] = {7, 8, 3, 4};
  const int32_t positions_data[] = {3, 1};
  int output_dims[] = {3, 0, 0, 0};
  float output_data[4];
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data, axis);
}

TF_LITE_MICRO_TEST(FloatGatherOpTestLastAxis) {
  const int axis = -1;
  const int input_dims[] = {3, 1, 2, 3};
  const int golden_dims[] = {1, 2, 2};
  const int positions_dims[] = {1, 2};
  const float input_data[] = {1, 2, 3, 4, 5, 6};
  const float golden_data[] = {3, 1, 6, 4};
  const int32_t positions_data[] = {2, 0};
  int output_dims[] = {3, 0, 0, 0};
  float output_data[4];
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data, axis);
}

TF_LITE_MICRO_TEST(FloatGatherOpTestLastAxis0DIndex) {
  const int axis = -1;
  const int input_dims[] = {3, 1, 2, 3};
  const int golden_dims[] = {1, 2};
  const int positions_dims[] = {0};
  const float input_data[] = {1, 2, 3, 4, 5, 6};
  const float golden_data[] = {3, 6};
  const int32_t positions_data[] = {2};
  int output_dims[] = {2, 0, 0};
  float output_data[2];
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data, axis);
}

TF_LITE_MICRO_TEST(GatherOpTestFloat32Int32) {
  const int input_dims[] = {2, 2, 2};
  const int* golden_dims = nullptr;
  const int positions_dims[] = {1, 2};
  const float input_data[] = {13.3, -13.4, -1.4, 1.5};
  const float golden_data[] = {-1.4, 1.5, 13.3, -13.4};
  const int32_t positions_data[] = {1, 0};
  int output_dims[] = {2, 0, 0};
  float output_data[4];
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(GatherOpTestInt8Int32) {
  const int input_dims[] = {2, 2, 2};
  const int* golden_dims = nullptr;
  const int positions_dims[] = {1, 2};
  const int8_t input_data[] = {-13, -120, 14, 15};
  const int8_t golden_data[] = {14, 15, -13, -120};
  const int32_t positions_data[] = {1, 0};
  int output_dims[] = {2, 0, 0};
  int8_t output_data[4];
  tflite::testing::TestGather<int8_t, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data);
}

TF_LITE_MICRO_TEST(GatherOpTestBatchDims2) {
  const int axis = 2;
  const int batch_dims = 2;
  const int input_dims[] = {4, 2, 2, 3, 5};
  const int golden_dims[] = {2, 2, 2, 5};
  const int positions_dims[] = {3, 2, 2, 2};
  const float input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                              12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                              24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                              36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                              48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
  const float golden_data[] = {5,  6,  7,  8,  9,  0,  1,  2,  3,  4,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                               35, 36, 37, 38, 39, 30, 31, 32, 33, 34,
                               45, 46, 47, 48, 49, 50, 51, 52, 53, 54};
  const int32_t positions_data[] = {1, 0, 0, 1, 1, 0, 0, 1};
  int output_dims[] = {4, 0, 0, 0, 0};
  float output_data[40];
  tflite::testing::TestGather<float, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data, axis, batch_dims);
}

TF_LITE_MICRO_TEST(GatherOpTestBatchDims1) {
  const int axis = 2;
  const int batch_dims = 1;
  const int input_dims[] = {4, 2, 2, 3, 5};
  const int golden_dims[] = {2, 2, 2, 2, 5};
  const int positions_dims[] = {3, 2, 2, 2};
  const int32_t positions_data[] = {1, 0, 0, 1, 1, 0, 0, 1};
  int output_dims[] = {5, 0, 0, 0, 0, 0};
  int8_t output_data[80];
  tflite::testing::TestGather<int8_t, int32_t>(
      input_dims, tflite::testing::batchdims1_input_data_i8, positions_dims,
      positions_data, output_dims, output_data, golden_dims,
      tflite::testing::batchdims1_golden_data_i8, axis, batch_dims);
}

TF_LITE_MICRO_TEST(GatherOpTestNegativeBatchDims) {
  const int axis = 2;
  const int batch_dims = -2;
  const int input_dims[] = {4, 2, 2, 3, 5};
  const int golden_dims[] = {2, 2, 2, 2, 5};
  const int positions_dims[] = {3, 2, 2, 2};
  const int32_t positions_data[] = {1, 0, 0, 1, 1, 0, 0, 1};
  int output_dims[] = {5, 0, 0, 0, 0};
  int8_t output_data[80];
  tflite::testing::TestGather<int8_t, int32_t>(
      input_dims, tflite::testing::batchdims1_input_data_i8, positions_dims,
      positions_data, output_dims, output_data, golden_dims,
      tflite::testing::batchdims1_golden_data_i8, axis, batch_dims);
}

TF_LITE_MICRO_TEST(GatherOpTestBatchDimsEqualIndiceDims) {
  const int axis = 3;
  const int batch_dims = 3;
  const int input_dims[] = {4, 2, 2, 2, 5};
  const int golden_dims[] = {2, 2, 2};
  const int positions_dims[] = {3, 2, 2, 2};
  const int8_t input_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                               10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                               30, 31, 32, 33, 34, 35, 36, 37, 38, 39};
  const int8_t golden_data[] = {1, 5, 10, 16, 21, 25, 30, 36};
  const int32_t positions_data[] = {1, 0, 0, 1, 1, 0, 0, 1};
  int output_dims[] = {3, 0, 0, 0};
  int8_t output_data[8];
  tflite::testing::TestGather<int8_t, int32_t>(
      input_dims, input_data, positions_dims, positions_data, output_dims,
      output_data, golden_dims, golden_data, axis, batch_dims);
}

TF_LITE_MICRO_TESTS_END
