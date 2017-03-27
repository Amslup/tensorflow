# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

class IpuXlaSimpleNetworkTest(test_util.TensorFlowTestCase):

    def testAdd(self):
        with tf.device("/device:XLA_IPU:0"):
            with tf.Session() as sess:
                pa = tf.placeholder(tf.float32, [2,2], name="a")
                pb = tf.placeholder(tf.float32, [2,2], name="b")
                output = pa + pb

                fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
                result = sess.run(output, fd)
                self.assertAllClose(result, [[1.,2.],[6.,8.]])

                fd = {pa: [[0.,0.],[1.,1.]], pb: [[2.,1.],[4.,5.]]}
                result = sess.run(output, fd)
                self.assertAllClose(result, [[2.,1.],[5.,6.]])

    def testAddVariable(self):
        with tf.device("/device:XLA_IPU:0"):
            with tf.Session() as sess:
                pa = tf.placeholder(tf.float32, [2,2], name="a")
                v = tf.Variable(tf.zeros([2,2]))
                a = tf.assign(v, pa + v)

                sess.run(tf.global_variables_initializer())

                fd = {pa: [[0.,0.],[1.,1.]]}
                result = sess.run(a, fd)
                self.assertAllClose(result, [[0.,0.],[1.,1.]])

    def testTransposeNegate(self):
        with tf.device("/device:XLA_IPU:0"):
            with tf.Session() as sess:
                pa = tf.placeholder(tf.float32, [2,2,3], name="a")
                a = array_ops.transpose(pa, [2, 1, 0])
                b = math_ops.negative(a)

                sess.run(tf.global_variables_initializer())

                fd = {
                    pa: [[[1, 2, 3], [3, 4, 5]],[[5,6,7],[7,8,9]]]
                }
                result = sess.run(b, fd)
                self.assertAllClose(result,
                                    [[[-1,-5],[-3,-7]],
                                     [[-2,-6],[-4,-8]],
                                     [[-3,-7],[-5,-9]]])

    def testTransposeNegate2(self):
        with tf.device("/device:XLA_IPU:0"):
            with tf.Session() as sess:
                pa = tf.placeholder(tf.float32, [2,2,3], name="a")
                a = array_ops.transpose(pa, [1, 2, 0])
                b = math_ops.negative(a)

                sess.run(tf.global_variables_initializer())

                fd = {
                    pa: [[[1, 2, 3], [3, 4, 5]],[[5,6,7],[7,8,9]]]
                }
                result = sess.run(b, fd)
                self.assertAllClose(result,
                                    [[[-1,-5],[-2,-6],[-3,-7]],
                                     [[-3,-7],[-4,-8],[-5,-9]]])

    def testReshape(self):
        with tf.device("/device:XLA_IPU:0"):
            with tf.Session() as sess:
                pa = tf.placeholder(tf.float32, [2,1,3], name="a")
                a = array_ops.reshape(pa, [1,3,2])

                sess.run(tf.global_variables_initializer())

                fd = {
                    pa: [[[1,2,3]],[[5,6,7]]]
                }
                result = sess.run(a, fd)
                self.assertAllClose(result,
                                    [[[1,2],[3,5],[6,7]]])

if __name__ == "__main__":
    googletest.main()
