# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Kafka Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.kafka.python.ops import kafka_op_loader  # pylint: disable=unused-import
from tensorflow.contrib.kafka.python.ops import gen_dataset_ops
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape


class KafkaDataset(Dataset):
  """A Kafka Dataset that consumes the message.
  """

  def __init__(self):
    """Create a KafkaReader.
    """
    super(KafkaDataset, self).__init__()

  def _as_variant_tensor(self):
    return gen_dataset_ops.kafka_dataset()

  @property
  def output_classes(self):
    return {
	'a' : ops.Tensor,
	'b' : {
	    'b1' : ops.Tensor,
	    'b2' : ops.Tensor
	}
    }

  @property
  def output_shapes(self):
    return {
	'a' : tensor_shape.scalar(),
        'b' : {
            'b1' : tensor_shape.TensorShape([4]),
	    'b2' : tensor_shape.scalar()
        }
    }

  @property
  def output_types(self):
    return {
    	'a' : dtypes.string,
        'b' : {
           'b1' : dtypes.int32,
           'b2' : dtypes.int64
        }
    }
