#ifndef XCORE_OPS_H_
#define XCORE_OPS_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "flatbuffers/flexbuffers.h"

extern "C" {
    #include "nn_operator.h"
    #include "nn_types.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

TfLiteRegistration* Register_Conv_SIDO();
TfLiteRegistration* Register_Conv_DIDO();
TfLiteRegistration* Register_FullyConnected_AOI();
TfLiteRegistration* Register_FullyConnected_AOF();
TfLiteRegistration* Register_ArgMax_16();
TfLiteRegistration* Register_MaxPool();
TfLiteRegistration* Register_AvgPool();

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_OPS_H_
