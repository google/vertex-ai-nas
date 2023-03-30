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
import os
from typing import Tuple

import cloud_nas_utils
import metrics_reporter
import tf_utils
from gcs_utils import gcs_path_utils
from third_party.tutorial import search_spaces
import numpy as np
import tensorflow as tf


def create_arg_parser():
  """Creates arg parser."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # Nas-service related flags.
  parser.add_argument(
      '--nas_params_str', type=str, help='Nas args serialized in JSON string.')
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


def create_model(model_spec):
  """Create a model for training.

  Args:
    model_spec: A PyGlove-based search-space-sample.

  Returns:
    The model to use for training.
  """
  # pylint: disable=g-complex-comprehension
  layers = [tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 28, 28, 1)))
           ] + [
               tf.keras.layers.Conv2D(
                   filters=layer_spec.filter_size,
                   kernel_size=(layer_spec.kernel_size, layer_spec.kernel_size),
                   padding='same',
                   activation='relu') for layer_spec in model_spec
           ]
  # pylint: enable=g-complex-comprehension

  layers.append(tf.keras.layers.Flatten())
  layers.append(tf.keras.layers.Dense(10, activation='softmax'))
  return tf.keras.Sequential(layers)


def train_and_eval(argv):
  """Trains and eval the NAS model."""

  argv.job_dir = cloud_nas_utils.get_job_dir_from_environment_if_exist(
      current_job_dir=argv.job_dir)
  argv.job_dir = gcs_path_utils.gcs_fuse_path(argv.job_dir)
  # Create job-dir if it does not exist.
  if not os.path.exists(argv.job_dir):
    os.makedirs(argv.job_dir)

  if argv.retrain_search_job_trials:
    # Resets `nas_params_str` if this job is to retrain a previous NAS trial.
    argv.nas_params_str = cloud_nas_utils.get_finetune_nas_params_str(
        retrain_search_job_trials=argv.retrain_search_job_trials,
        retrain_search_job_dir=argv.retrain_search_job_dir)

  # Process nas_params_str passed by the NAS-service.
  # This gives one instance of the search-space to be used for this trial.
  tunable_object = cloud_nas_utils.parse_and_save_nas_params_str(
      search_space=search_spaces.mnist_list_of_dictionary_search_space(),
      nas_params_str=argv.nas_params_str,
      model_dir=argv.job_dir,
  )
  unused_serialized_tunable_object = (
      cloud_nas_utils.serialize_and_save_tunable_object(
          tunable_object=tunable_object, model_dir=argv.job_dir
      )
  )
  model_spec = tunable_object

  # Run training.
  tr_x, tr_y, te_x, te_y = download_and_prep_data()

  model = create_model(model_spec=model_spec)
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

  # Reporting the model metadata.
  other_metrics = {
      'filter_size_0': model_spec[0].filter_size,
      'kernel_size_0': model_spec[0].kernel_size,
      'filter_size_1': model_spec[1].filter_size,
      'kernel_size_1': model_spec[1].kernel_size
  }

  # Reporting metrics back to the NAS_service.
  metric_tag = os.environ.get('CLOUD_ML_HP_METRIC_TAG', '')
  if argv.retrain_search_job_trials:
    other_metrics[
        'nas_trial_id'] = cloud_nas_utils.get_search_trial_id_to_finetune(
            argv.retrain_search_job_trials)
  if metric_tag:
    nas_metrics_reporter = metrics_reporter.NasMetricsReporter()
    nas_metrics_reporter.report_metrics(
        hyperparameter_metric_tag=metric_tag,
        metric_value=test_acc,
        global_step=argv.num_epochs,
        other_metrics=other_metrics)


if __name__ == '__main__':
  tf_utils.suppress_tf_logging()
  cloud_nas_utils.setup_logging()
  flags = create_arg_parser().parse_args()
  train_and_eval(flags)
