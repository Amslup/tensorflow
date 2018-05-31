# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re

import test_utils as tu

from tensorflow.python.platform import googletest
from tensorflow.python.framework import test_util
from tensorflow.core.protobuf import config_pb2
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

class IpuIpuModelTest(test_util.TensorFlowTestCase):

    def testIpuModelDevice(self):
        with tf.device("/device:IPU:0"):
            pa = tf.placeholder(tf.float32, [2,2], name="a")
            pb = tf.placeholder(tf.float32, [2,2], name="b")
            output = pa + pb

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL

        with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:

            fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
            result = sess.run(output, fd)
            self.assertAllClose(result, [[1.,2.],[6.,8.]])

    def testIpuModelDeviceWithNoReport(self):
        with tf.device("/device:IPU:0"):
            pa = tf.placeholder(tf.float32, [2,2], name="a")
            pb = tf.placeholder(tf.float32, [2,2], name="b")
            output = pa + pb

        with tf.device('cpu'):
            with tf.control_dependencies([output]):
                report = gen_ipu_ops.ipu_event_trace()

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL

        with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:

            fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
            sess.run(report, fd)

            result, rep = sess.run([output, report], fd)
            self.assertAllClose(result, [[1.,2.],[6.,8.]])
            self.assertTrue(len(rep) == 0)

    def testIpuModelDeviceWithReport(self):
        with tf.device("/device:IPU:0"):
            pa = tf.placeholder(tf.float32, [2,2], name="a")
            pb = tf.placeholder(tf.float32, [2,2], name="b")
            output = pa + pb

        with tf.device('cpu'):
            with tf.control_dependencies([output]):
                report = gen_ipu_ops.ipu_event_trace()

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
        dev.profiling.enable_compilation_trace = True
        dev.profiling.enable_io_trace = False
        dev.profiling.enable_execution_trace = True

        with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:

            fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
            sess.run(report, fd)

            result, rep = sess.run([output, report], fd)
            self.assertAllClose(result, [[1.,2.],[6.,8.]])
            self.assertTrue(len(rep) == 2)

    def testIpuModelDeviceWithMultipleReport(self):
        with tf.device("/device:IPU:0"):
            pa = tf.placeholder(tf.float32, [2,2], name="a")
            pb = tf.placeholder(tf.float32, [2,2], name="b")
            out1 = pa + pb
            out2 = pa - pb

        with tf.device('cpu'):
            with tf.control_dependencies([out1, out2]):
                report = gen_ipu_ops.ipu_event_trace()

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
        dev.profiling.enable_compilation_trace = True
        dev.profiling.enable_io_trace = False
        dev.profiling.enable_execution_trace = True

        with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:

            fd = {pa: [[1.,1.],[2.,3.]], pb: [[0.,1.],[4.,5.]]}
            sess.run(report, fd)

            result = sess.run(out1, fd)
            self.assertAllClose(result, [[1.,2.],[6.,8.]])

            result, rep = sess.run([out2, report], fd)
            self.assertAllClose(result, [[1.,0.],[-2.,-2.]])

            # 2x compile, 2x load engine
            self.assertTrue(len(rep) == 4)

    def testIpuModelDeviceMultipleIPUs(self):
        with tf.device("/device:IPU:0"):
            pa = tf.placeholder(tf.float32, [480], name="a")
            pb = tf.placeholder(tf.float32, [480], name="b")
            output = pa + pb

        with tf.device('cpu'):
            with tf.control_dependencies([output]):
                report = gen_ipu_ops.ipu_event_trace()

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
        dev.profiling.enable_compilation_trace = True
        dev.profiling.enable_io_trace = False
        dev.profiling.enable_execution_trace = True
        dev.ipu_model_config.num_ipus = 2
        dev.ipu_model_config.tiles_per_ipu = 4

        with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:

            fd = {pa: np.zeros([480]), pb: np.zeros([480])}
            sess.run(report, fd)

            result, rep = sess.run([output, report], fd)
            self.assertAllClose(result, np.zeros([480]))
            self.assertTrue(len(rep) == 2)

            s = tu.extract_all_strings_from_event_trace(rep)
            l = s.split("\n")
            l = [x for x in l if re.search(" *Num tiles computing *: *8", x)]
            self.assertTrue(len(l) == 1)

    def testIpu(self):
        with tf.device("/device:IPU:0"):
            pa = tf.placeholder(tf.float32, [480], name="a")
            pb = tf.placeholder(tf.float32, [480], name="b")
            output = pa + pb

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU
        dev.profiling.enable_compilation_trace = True
        dev.profiling.enable_io_trace = False
        dev.profiling.enable_execution_trace = True
        dev.ipu_model_config.num_ipus = 2

        try:
            with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:
                fd = {pa: np.zeros([480]), pb: np.zeros([480])}
                sess.run(output, fd)
        except tf.errors.InternalError:
            pass

    def testEngineCompilationOptions(self):
        with tf.device("/device:IPU:0"):
            pa = tf.placeholder(tf.float32, [480], name="a")
            pb = tf.placeholder(tf.float32, [480], name="b")
            output = pa + pb

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
        dev.profiling.enable_compilation_trace = True
        dev.profiling.enable_io_trace = False
        dev.profiling.enable_execution_trace = True
        dev.ipu_model_config.num_ipus = 2

        opt = dev.compilation_options.add()
        opt.option = "some_option"
        opt.value = "some_value"

        try:
            with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:
                fd = {pa: np.zeros([480]), pb: np.zeros([480])}
                sess.run(output, fd)
        except tf.errors.UnknownError:
            pass

    def testIpuSimulatorDevice(self):
        with tf.device("/device:IPU:0"):
            pa = tf.placeholder(tf.float32, [16], name="a")
            pb = tf.placeholder(tf.float32, [16], name="b")
            output = pa + pb

        opts = config_pb2.IPUOptions()
        dev = opts.device_config.add()
        dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_SIMULATOR

        try:
            with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as sess:
                fd = {pa: np.zeros([16]), pb: np.zeros([16])}
                res = sess.run(output, fd)

                self.assertAllClose(res, np.zeros([16]))
        except tf.errors.InternalError as e:
            self.assertTrue(e.message.startswith("Failed to create session"))

if __name__ == "__main__":
    googletest.main()
