/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("BatchNormalizeTraining")
    .Input("input: T")
    .Input("scale: T")
    .Input("bias: T")
    .Input("running_mean: Ref(T)")
    .Input("running_inv_var: Ref(T)")
    .Output("out: T")
    .Output("save_mean: T")
    .Output("save_inv_var: T")
    .Attr("T: {float}")
    .Attr("epsilon: float")
    .Attr("exponential_average_factor: float")
    .Doc(R"doc(
Perform batch normalization using batch statistics.

input: A 4D input Tensor.
scale: A 1D Tensor with size equal to the number of channels.
  This is the learned scale value multiplied post normalization.
  Also known as gamma.
bias: A 1D Tensor with size equal to the number of channels.
  This is the learned bias value added post normalization.
  Also known as beta.
running_mean: A 1D Tensor reference with size equal to the number of channels.
  During training this reference is updated via a rolling average from the current
  batch mean.
running_inv_var: A 1D Tensor reference with size equal to the number of channels.
  During training this reference is updated via a rolling average from 1 over the
  current batch variance.
out: A 4D output Tensor. The input after applying batch normalization.
save_mean: A 1D Tensor. Computed means. To be used in the backward pass.
save_inv_var: A 1D Tensor. Computed inverse variance. To be used in the backward pass.
epsilon: float
exponential_average_factor: float. Exponential decay on rolling average.
)doc");

REGISTER_OP("BatchNormalizeTrainingGrad")
    .Input("input: T")
    .Input("output_grad: T")
    .Input("scale: T")
    .Input("saved_mean: T")
    .Input("saved_var: T")
    .Output("input_grad: T")
    .Output("scale_grad: T")
    .Output("bias_grad: T")
    .Attr("T: {float}")
    .Attr("epsilon: float")
    .Doc(R"doc(
Perform the backward pass for batch normalization using batch statistics.

input: A 4D input Tensor.
scale: A 1D Tensor with size equal to the number of channels.
  This is the learned scale value multiplied post normalization.
  Also known as gamma.
save_mean: A 1D Tensor. Computed means. From output of forward pass.
save_inv_var: A 1D Tensor. Computed inverse variance. From output of forward pass
input_grad: A 4D output tensor. The gradient with respect to the input.
scale_grad: A 1D output tensor. The gradient with respect to the scale.
bias_grad: A 1D output tensor. The gradient with respect to the bias.
epsilon: float. Same epsilon used in forward pass.
)doc");
}  // namespace tensorflow
