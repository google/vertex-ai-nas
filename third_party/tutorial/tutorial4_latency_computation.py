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
import timeit

import cloud_nas_utils
import model_metrics_evaluator
import tf_utils

import numpy as np
import tensorflow as tf


class LatencyEvaluator(model_metrics_evaluator.ModelMetricsEvaluator):
  """Implements the process which evaluates and saves model-latency."""

  def __init__(self,
               service_endpoint,
               project_id,
               nas_job_id,
               latency_worker_id = 0,
               num_latency_workers = 1):
    super(LatencyEvaluator, self).__init__(
        service_endpoint=service_endpoint,
        project_id=project_id,
        nas_job_id=nas_job_id,
        latency_worker_id=latency_worker_id,
        num_latency_workers=num_latency_workers)

  def evaluate_saved_model(self, trial_id, saved_model_path):
    """Returns model latency."""
    logging.info("Job output directory is %s", self.job_output_dir)
    model = tf.keras.models.load_model(saved_model_path)
    my_input = np.random.rand(1, 28, 28)

    def _run_prediction():
      model(my_input)

    num_iter_warm_up = 50
    avg_latency_in_secs_warm_up = timeit.timeit(
        _run_prediction, number=num_iter_warm_up) / float(num_iter_warm_up)
    logging.info("warm-up latency is %f", avg_latency_in_secs_warm_up)
    num_iter = 100
    avg_latency_in_secs = timeit.timeit(
        _run_prediction, number=num_iter) / float(num_iter)
    logging.info("latency is %f", avg_latency_in_secs)
    return {
        "latency_in_milliseconds": avg_latency_in_secs * 1000.0,
        "model_memory": 0
    }


def create_arg_parser():
  """Creates arg parser."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # FLAGS for multi-device latency calculation.
  parser.add_argument(
      "--dummy_flag",
      type=str,
      default="",
      required=False,
      help="A dummy flag for the demonstration purpose.")
  parser.add_argument(
      "--latency_worker_id",
      type=int,
      default=0,
      required=False,
      help="Latency calculation worker ID to start. Should be an integer in "
      "[0, num_latency_workers - 1]. If num_latency_workers > 1, each worker "
      "will only handle a subset of the parallel training trials based on "
      "their trial-ids. For cloud, this will be set automatically. For on-prem "
      "devices, the user have to pass this flag.")
  parser.add_argument(
      "--num_latency_workers",
      type=int,
      default=1,
      required=False,
      help="The total number of parallel latency calculator workers. If "
      "num_latency_workers > 1, it is used to select a subset of the parallel "
      "training trials based on their trial-ids. For cloud, this will be set "
      "automatically. For on-prem devices, the user have to pass the flag.")
  ######################################################
  ######## These FLAGS are set automatically by the nas-client.
  ######################################################
  parser.add_argument(
      "--project_id",
      type=str,
      default="",
      help="The project ID to check for NAS job.")
  parser.add_argument(
      "--nas_job_id", type=str, default="", help="The NAS job id.")
  parser.add_argument(
      "--service_endpoint",
      type=str,
      default="https://ml.googleapis.com/v1",
      help="The end point of the service. Default is https://ml.googleapis.com/v1."
  )
  ######################################################
  ######################################################

  return parser


def compute_latency(argv):
  """Computes latency."""
  latency_evaluator = LatencyEvaluator(
      service_endpoint=argv.service_endpoint,
      project_id=argv.project_id,
      nas_job_id=argv.nas_job_id,
      latency_worker_id=argv.latency_worker_id,
      num_latency_workers=argv.num_latency_workers)
  latency_evaluator.run_continuous_evaluation_loop()


if __name__ == "__main__":
  tf_utils.suppress_tf_logging()
  cloud_nas_utils.setup_logging()
  flags = create_arg_parser().parse_args()
  compute_latency(flags)
