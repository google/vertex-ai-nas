# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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


# MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# ==============================================================================
"""Train MNIST.

This is a basic working ML program which trains MNIST.
The code is modified from the tf.keras tutorial here:
https://www.tensorflow.org/tutorials/keras/classification

(The tutorial uses Fashion-MNIST,
but we just use "regular" MNIST for these tutorials.)

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from typing import Tuple

import numpy as np
import tensorflow as tf


def create_arg_parser():
  """Creates arg parser."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--job-dir', type=str, help='Job output directory.')
  parser.add_argument(
      '--num_epochs', type=int, default=1, help='Number of epochs to train.')

  return parser


def download_and_prep_data():
  """Download dataset and scale to [0, 1].

  Returns:
    tr_x: Training data.
    tr_y: Training labels.
    te_x: Testing data.
    te_y: Testing labels.
  """
  mnist_dataset = tf.keras.datasets.mnist
  (tr_x, tr_y), (te_x, te_y) = mnist_dataset.load_data()
  tr_x = tr_x / 255.0
  te_x = te_x / 255.0
  return tr_x, tr_y, te_x, te_y


def create_model():
  """Create a model for training.

  Create a simple tf.keras model for training.

  Returns:
    The model to use for training.
  """

  layers = [
      tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 28, 28, 1))),
      tf.keras.layers.Conv2D(
          filters=4, kernel_size=(3, 3), padding='same', activation='relu'),
      tf.keras.layers.Conv2D(
          filters=8, kernel_size=(3, 3), padding='same', activation='relu')
  ]
  layers.append(tf.keras.layers.Flatten())
  layers.append(tf.keras.layers.Dense(10, activation='softmax'))
  return tf.keras.Sequential(layers)


def train_and_eval(argv):
  """Trains and eval the NAS model."""

  tr_x, tr_y, te_x, te_y = download_and_prep_data()

  model = create_model()
  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=argv.job_dir)
  model.fit(
      tr_x, tr_y, epochs=argv.num_epochs, callbacks=[tensorboard_callback])
  model.evaluate(te_x, te_y, verbose=2)


if __name__ == '__main__':
  flags = create_arg_parser().parse_args()
  train_and_eval(flags)
