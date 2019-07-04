# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for regularization with Distribution Strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.framework import ops
from tensorflow.python.layers import base as base_layers
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class RegularizationWithDistributionStrategyTest(
    test.TestCase, parameterized.TestCase):

  def model_fn(self):
    regularizer = lambda x: math_ops.reduce_sum(x) * 1e-3
    layer = base_layers.Layer(name='my_layer', dtype='float32')
    layer.add_variable('my_var', [2, 2], regularizer=regularizer)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.default_strategy,
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.mirrored_strategy_with_two_gpus,
          ],
          mode=["graph"]))
  def testWithLayers(self, distribution):
    with distribution.scope():
      distribution.extended.call_for_each_replica(self.model_fn)
      reg_list = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
      self.assertEqual(len(reg_list), 1)
      for reg_loss in reg_list:
        if distribution.num_replicas_in_sync > 1:
          self.assertTrue(isinstance(reg_loss, value_lib.Mirrored))
        else:
          self.assertTrue(isinstance(reg_loss, ops.Tensor))


if __name__ == "__main__":
  test.main()
