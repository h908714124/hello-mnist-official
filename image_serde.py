#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np

import argparse
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order
from tensorflow.python.training import session_run_hook

from official.mnist import dataset
from official.utils.arg_parsers import parsers

import matplotlib.pyplot as plt

LEARNING_RATE = 1e-4


def save(ar, file_name):
  w, h = 28, 28
  data = np.zeros((h, w), dtype=np.uint8)
  for i in range(0, 28):
    for j in range(0, 28):
      data[i, j] = ar[28 * i + j] * 256
  img = Image.fromarray(data, 'L')
  img.save(file_name)


def load():
  img = Image.open('my.png').load()
  m = np.zeros(784, np.float32)
  for i in range(0, 28):
    for j in range(0, 28):
      m[28 * i + j] = 0.00390625 * img[j, i]
  return m


it = dataset.train('mnist_data').batch(3).make_one_shot_iterator()
next = it.get_next()
with tf.Session() as sess:
  # Run the initializer
  sess.run(tf.global_variables_initializer())
  n = sess.run(next)
  save(n[0][2], 'my.png')
  reloaded = load()
  save(reloaded, 'yours.png')
  print(n[1][2])
 
