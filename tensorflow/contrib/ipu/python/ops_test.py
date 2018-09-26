from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.contrib.compiler import xla
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import googletest
from tensorflow.contrib import ipu
from tensorflow.python.training import gradient_descent

def count_event_type(events, type, payload=0):
  return sum(map((lambda x: 1 if x.type==type and len(x.data_str) > payload
                              else 0), events))

class ContribIpuOpsTest(test_util.TensorFlowTestCase):

  def testSummary(self):
    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, [1], name="a")
      b = array_ops.placeholder(np.float32, [1], name="b")
      out = a + b

    summary = ipu.ops.ipu_compile_summary('comp', out)

    cfg = ipu.utils.create_ipu_config(profiling=True, type='IPU_MODEL')
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      fd = {
        a: [1.0],
        b: [2.0],
      }
      result, s = sess.run([out, summary], fd)
      self.assertAllClose(result, [3.0])
      self.assertTrue(len(s) > 100)

  def testCreateConfig(self):
    cfg = ipu.utils.create_ipu_config(type='IPU')
    self.assertTrue(isinstance(cfg, config_pb2.IPUOptions))

    cfg = ipu.utils.create_ipu_config(type='IPU_MODEL')
    self.assertTrue(isinstance(cfg, config_pb2.IPUOptions))

    cfg = ipu.utils.create_ipu_config(type='CPU')
    self.assertTrue(isinstance(cfg, config_pb2.IPUOptions))

    cfg = ipu.utils.create_ipu_config(type='CPU', num_devices=2)
    self.assertTrue(isinstance(cfg, config_pb2.IPUOptions))
    self.assertTrue(len(cfg.device_config), 2)

    cfg = ipu.utils.create_ipu_config(type='CPU', num_devices=2, num_ipus=4)
    self.assertTrue(isinstance(cfg, config_pb2.IPUOptions))
    self.assertTrue(len(cfg.device_config), 2)
    self.assertTrue(cfg.device_config[0].ipu_model_config.num_ipus, 4)
    self.assertTrue(cfg.device_config[1].ipu_model_config.num_ipus, 4)

    with self.assertRaises(Exception):
      cfg = ipu.utils.create_ipu_config(type='Other')

    with self.assertRaises(Exception):
      ipu.utils.create_ipu_config(num_ipus=[1,2,3])

    with self.assertRaises(Exception):
      ipu.utils.create_ipu_config(num_devices=1, ipu_device_config_index=[1,2])

    with self.assertRaises(Exception):
      ipu.utils.create_ipu_config(num_devices=2, ipu_device_config_index=1)

    with self.assertRaises(Exception):
      ipu.utils.create_ipu_config(num_ipus=2, ipu_device_config_index=0)

    with self.assertRaises(Exception):
      ipu.utils.create_ipu_config(num_devices=5)

  def testEventFetchAndStringDecode(self):
    with ops.device("/device:IPU:0"):
      a = array_ops.placeholder(np.float32, [1], name="a")
      b = array_ops.placeholder(np.float32, [1], name="b")
      out = a + b

    events = gen_ipu_ops.ipu_event_trace()

    cfg = ipu.utils.create_ipu_config(profiling=True, type='IPU_MODEL')
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      # Discard any existing events
      sess.run(events)

      fd = {
        a: [1.0],
        b: [2.0],
      }
      result = sess.run(out, fd)
      self.assertAllClose(result, [3.0])

      # 1x compile begin, 1x compile end, 1x load engine, 1x execute
      e = sess.run(events)
      self.assertEqual(len(e), 4)

      dump = ipu.utils.extract_all_strings_from_event_trace(e);
      self.assertTrue(len(dump) > 100)

  def testIpuSimpleScope(self):
    def my_net(a, b):
      c = a + b
      return [c]

    with ops.device('cpu'):
      a = array_ops.placeholder(np.float32, [1], name="a")
      b = array_ops.placeholder(np.float32, [1], name="b")
      events = gen_ipu_ops.ipu_event_trace()

    with ipu.ops.ipu_scope("/device:IPU:0"):
      r = xla.compile(my_net, inputs=[a, b])

    cfg = ipu.utils.create_ipu_config(profiling=True, type='IPU_MODEL')
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:

      fd = {
        a: [1],
        b: [2],
      }

      sess.run(events)

      res = sess.run(r[0], fd)
      self.assertEqual(res, [3])

      e = sess.run(events)
      evts = ipu.utils.extract_all_events(e)
      self.assertEqual(count_event_type(evts, IpuTraceEvent.COMPILE_END, 10), 1)

  def testIpuWhileScope(self):
    # 1: design is targetted at the device
    # 2: variables are resource variables
    # 3: training a while_loop is possible
    def my_net(a, b):
      c = variable_scope.get_variable('c', initializer=[1.0])
      self.assertTrue("ResourceVariable" in str(type(c)))

      lstm_cell = rnn_cell.LSTMCell(1, forget_bias=1.0)
      outputs, states = rnn.dynamic_rnn(lstm_cell, a, dtype=np.float32)

      logits = outputs[-1] * c
      self.assertEqual(logits.device, "/device:IPU:0")

      res = array_ops.reshape(logits, [1,8,1])

      l = losses.mean_squared_error(res, b)

      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(l)

      return [l, train]

    with ops.device('cpu'):
      a = array_ops.placeholder(np.float32, [1,8,1], name="a")
      b = array_ops.placeholder(np.float32, [1,8,1], name="b")

    with ipu.ops.ipu_scope("/device:IPU:0"):

      l = xla.compile(my_net, inputs=[a, b])

    cfg = ipu.utils.create_ipu_config(profiling=False, type='IPU_MODEL')
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      # Initialize and then discard events relating to initialization
      sess.run(variables.global_variables_initializer())

      fd = {
        a: [[[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]]],
        b: [[[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.]]],
      }

      l_initial = sess.run([l], fd)

      for _ in range(100):
        _ = sess.run([l], fd)

      l_final = sess.run([l], fd)

      self.assertTrue(l_initial > l_final)

  def testInitializerDeviceChange(self):

    inp = array_ops.placeholder(np.float32, [1,8,8,4])
    with ipu.ops.ipu_scope("/device:IPU:0"):
      out = convolutional.conv2d(inp, 8, 1)

    events = gen_ipu_ops.ipu_event_trace()

    ipu.utils.move_variable_initialization_to_cpu()

    cfg = ipu.utils.create_ipu_config(profiling=True, type='IPU_MODEL')
    with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
      # Discard any pending events
      sess.run(events)

      # Run initializer (should be on CPU)
      sess.run(variables.global_variables_initializer())

      e = sess.run(events)
      self.assertEqual(len(e), 2) # compile begin/end, no load/execute

  def testDeviceConfigIndex(self):
    # We only allow 1 device config for IPU_MODEL, so if one is requested and it
    # is not 0, then an exception should occur.
    try:
      raised = False
      with ops.device("/device:IPU:0"):
        a = array_ops.placeholder(np.float32, [1], name="a")
        b = array_ops.placeholder(np.float32, [1], name="b")
        out = a + b

      cfg = ipu.utils.create_ipu_config(profiling=True, type='IPU_MODEL',
                                        ipu_device_config_index=1)
      with sl.Session(config=config_pb2.ConfigProto(ipu_options=cfg)) as sess:
        fd = {
          a: [1.0],
          b: [2.0],
        }
        result = sess.run(out, fd)
    except Exception as e:
      self.assertEqual(type(e).__name__, 'InvalidArgumentError')
      self.assertEqual(
        e.message,
        'Requested device configuration index 1, but 1 configuration was available.'
      )
      raised = True
    self.assertTrue(raised)

if __name__ == "__main__":
    googletest.main()
