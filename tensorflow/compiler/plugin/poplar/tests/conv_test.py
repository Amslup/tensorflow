# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import test_utils as tu

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

import numpy as np


class IpuXlaConvTest(test_util.TensorFlowTestCase):

  def testConv1x1_Stride2x1_In1x5(self):
    with tf.device("/device:IPU:0"):
      pa = tf.placeholder(tf.float32, [1,1,5,1], name="a")
      pb = tf.placeholder(tf.float32, [1,1,1,1], name="b")
      output = nn_ops.convolution(pa, pb, strides=[1,2], padding="VALID")

    with tf.Session() as sess:
      fd = {
        pa: [[[[1], [2], [3], [4], [5]]]],
        pb: [[[[10]]]]
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, [[[[10], [30], [50]]]])

  def testConv3x3_Pad1x1(self):
    with tf.device("/device:IPU:0"):
      pa = tf.placeholder(tf.float32, [1,14,14,64], name="a")
      pb = tf.placeholder(tf.float32, [3,3,64,128], name="b")
      output = nn_ops.convolution(pa, pb, padding="SAME")

    with tf.Session() as sess:
      fd = {
        pa: np.zeros([1,14,14,64]),
        pb: np.zeros([3,3,64,128])
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([1,14,14,128]))

  def testConv3x3_WithBias(self):
    with tf.device("/device:IPU:0"):
      pa = tf.placeholder(tf.float32, [1,14,14,64], name="a")
      pb = tf.placeholder(tf.float32, [3,3,64,128], name="b")
      bi = tf.placeholder(tf.float32, [128], name="b")
      output = nn_ops.convolution(pa, pb, padding="SAME")
      output = nn_ops.bias_add(output, bi)

    with tf.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        pa: np.zeros([1,14,14,64]),
        pb: np.zeros([3,3,64,128]),
        bi: np.zeros([128]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([1,14,14,128]))

      result = sess.run(report)
      self.assertTrue(len(result) == 2)
      cs_list = tu.get_compute_sets_from_report(result[1])

      ok = ['convolution.clone/Conv_3x3',
            'Copy_{<const>,arg0_input}',
            'call/addToChannel']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testConv8x8_WithBias(self):
    with tf.device("/device:IPU:0"):
      inp = tf.placeholder(tf.float32, [1,84,84,4], name="inp")
      wei = tf.placeholder(tf.float32, [8,8,4,16], name="wei")
      bia = tf.placeholder(tf.float32, [16], name="bia")
      output = nn_ops.conv2d(inp, wei, strides=[1,4,4,1], padding="VALID")
      output = nn_ops.bias_add(output, bia)

    with tf.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        inp: np.zeros([1,84,84,4]),
        wei: np.zeros([8,8,4,16]),
        bia: np.zeros([16]),
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, np.zeros([1, 20, 20, 16]))

      result = sess.run(report)
      self.assertTrue(len(result) == 2)
      cs_list = tu.get_compute_sets_from_report(result[1])

      ok = ['convolution.clone/Conv_8x8_stride4x4',
            'call/addToChannel']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))


  def testDepthwiseConv3x2(self):
    with tf.device("/device:IPU:0"):
      pa = tf.placeholder(tf.float32, [1,2,2,3], name="a")
      pb = tf.placeholder(tf.float32, [1,1,3,2], name="b")
      pc = tf.placeholder(tf.float32, [1,2,2,6], name="c")
      c = tf.nn.depthwise_conv2d(pa, pb, strides=[1,1,1,1], padding="SAME")
      output = c + pc

    with tf.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        pa: [[[[1,2,3],
               [4,5,6]],
              [[7,8,9],
               [10,11,12]]]],
        pb: [[[[6,5],
               [4,3],
               [2,1]]]],
        pc: [[[[1,1,1,1,1,1],
               [1,1,1,1,1,1]],
              [[1,1,1,1,1,1],
               [1,1,1,1,1,1]]]]
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, [[[[7, 6, 9, 7, 7, 4],
                                     [25, 21, 21, 16, 13, 7]],
                                    [[43, 36, 33, 25, 19, 10],
                                     [61, 51, 45, 34, 25, 13]]]])

      result = sess.run(report)
      self.assertTrue(len(result) == 2)
      cs_list = tu.get_compute_sets_from_report(result[1])

      ok = ['call.1.clone/Conv_1x1',
            'Copy_partials_to_call',
            'call/addToChannel']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testDepthwiseConv3x1(self):
    with tf.device("/device:IPU:0"):
      pa = tf.placeholder(tf.float32, [1,2,2,3], name="a")
      pb = tf.placeholder(tf.float32, [1,1,3,1], name="b")
      pc = tf.placeholder(tf.float32, [1,2,2,3], name="c")
      c = tf.nn.depthwise_conv2d(pa, pb, strides=[1,1,1,1], padding="SAME")
      output = c + pc

    with tf.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session() as sess:
      fd = {
        pa: [[[[1,2,3],
               [4,5,6]],
              [[7,8,9],
               [10,11,12]]]],
        pb: [[[[6],
               [4],
               [2]]]],
        pc: [[[[1,1,1],
               [1,1,1]],
              [[1,1,1],
               [1,1,1]]]]
      }
      result = sess.run(output, fd)
      self.assertAllClose(result, [[[[7, 9, 7],
                                     [25, 21, 13]],
                                    [[43, 33, 19],
                                     [61, 45, 25]]]])

      result = sess.run(report)
      self.assertTrue(len(result) == 2)
      cs_list = tu.get_compute_sets_from_report(result[1])

      ok = ['call.1.clone/Conv_1x1',
            'Copy_partials_to_call',
            'call/addToChannel']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))


if __name__ == "__main__":
    googletest.main()
