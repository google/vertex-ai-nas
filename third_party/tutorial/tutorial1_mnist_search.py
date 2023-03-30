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

# copybara-strip-begin
# The MNIST example comes from here:
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb
# So we need to include MIT and Apache license both.
# copybara-strip-end

"""NAS on MNIST.

This is a basic working ML program which does NAS on MNIST.
The code is modified from the tf.keras tutorial here:
https://www.tensorflow.org/tutorials/keras/classification

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
from typing import Tuple

import cloud_nas_utils
import metrics_reporter
import tf_utils
from gcs_utils import gcs_path_utils
import numpy as np
import tensorflow as tf


_LOCAL_DATA_DIR = '/test_data'


def create_arg_parser():
  """Creates arg parser."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # Nas-service related flags.
  parser.add_argument(
      '--nas_params_str', type=str, help='Nas args serialized in JSON string.')
  # NOTE: Although this flag is injected as "job-dir", it is consumed
  # in the code as "argv.job_dir" instead of the hyphen.
  parser.add_argument(
      '--job-dir', type=str, default='tmp', help='Job output directory.')
  parser.add_argument(
      '--retrain_search_job_dir',
      type=str,
      help='The job dir of the NAS search job to retrain.')
  parser.add_argument(
      '--retrain_search_job_trials',
      type=str,
      help='A list of trial IDs of the NAS search job to retrain, '
      'separated by comma.')

  # Flags specific to this trainer.
  parser.add_argument(
      '--num_epochs', type=int, default=1, help='Number of epochs to train.')
  parser.add_argument(
      '--dummy_input_file', type=str, default='',
      help='A file name. This flag is only for demonstrating how to access '
      'local files from local run.')
  parser.add_argument(
      '--dummy_gcs_bucket', type=str, default='',
      help='A GCS bucket. This flag is only for demonstrating how to access '
      'GCS from either local or cloud run.')

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

  # Print cloud config.
  logging.info(os.system('gcloud config list'))

  # Demonstrating how to read file from local data dir.
  if argv.dummy_input_file:
    with open(os.path.join(_LOCAL_DATA_DIR, argv.dummy_input_file), 'r') as f:
      logging.info(f.read())

  argv.job_dir = cloud_nas_utils.get_job_dir_from_environment_if_exist(
      current_job_dir=argv.job_dir
  )
  # Demonstrating how to access GCS buckets via GCSFuse.
  # NOTE: This code is not meant for accessing GCS buckets with local
  # docker run. Please read the tutorial documentation for alternatives.
  argv.job_dir = gcs_path_utils.gcs_fuse_path(argv.job_dir)
  logging.info('Job output directory: %s', argv.job_dir)

  if argv.dummy_gcs_bucket:
    argv.dummy_gcs_bucket = gcs_path_utils.gcs_fuse_path(argv.dummy_gcs_bucket)
    contents = os.listdir(argv.dummy_gcs_bucket)
    logging.info('Found %d items under %s.',
                 len(contents), argv.dummy_gcs_bucket)

  # Create job-dir if it does not exist.
  if not os.path.exists(argv.job_dir):
    os.makedirs(argv.job_dir)

  tr_x, tr_y, te_x, te_y = download_and_prep_data()

  model = create_model()
  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=argv.job_dir)
  model.fit(
      tr_x, tr_y, epochs=argv.num_epochs, callbacks=[tensorboard_callback])
  _, test_acc = model.evaluate(te_x, te_y, verbose=2)

  # Since this trainer does not write output files, we will save out
  # dummy files here to illustrate GCS file I/O.
  with open(os.path.join(argv.job_dir, 'dummy_output.txt'), 'w') as fp:
    fp.write('Finished training.')

  # Reporting metrics back to the NAS_service.
  metric_tag = os.environ.get('CLOUD_ML_HP_METRIC_TAG', '')
  if metric_tag:
    nas_metrics_reporter = metrics_reporter.NasMetricsReporter()
    nas_metrics_reporter.report_metrics(
        hyperparameter_metric_tag=metric_tag,
        metric_value=test_acc,
        global_step=argv.num_epochs,
        other_metrics={})


if __name__ == '__main__':
  tf_utils.suppress_tf_logging()
  cloud_nas_utils.setup_logging()
  flags = create_arg_parser().parse_args()
  train_and_eval(flags)
