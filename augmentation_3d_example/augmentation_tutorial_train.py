# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
r"""Example code to show how 3D-augmentation string from NAS-service is consumed.

NOTE: This example is only for stage1 search job.

Here is how to run this simple example:

  python3 nas_cli.py build --project_id=${PROJECT} \
  --trainer_docker_id=${AUG_TUTORIAL_DOCKER_ID} \
  --trainer_docker_file=augmentation_tutorial/augmentation_tutorial.Dockerfile

  python3 nas_cli.py search \
  --project_id=${PROJECT} \
  --job_id="${JOB_ID}" \
  --trainer_docker_id=${AUG_TUTORIAL_DOCKER_ID} \
  --prebuilt_search_space="augment_3d_basic" \
  --accelerator_type="" \
  --nas_target_reward_metric="top_1_accuracy" \
  --root_output_dir=${GCS_ROOT_DIR} \
  --use_prebuilt_trainer=False \
  --max_nas_trial=1 \
  --max_parallel_nas_trial=1 \
  --max_failed_nas_trial=1

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os

import cloud_nas_utils
import metrics_reporter
import search_spaces
import tf_utils
import pyglove as pg
import tensorflow.compat.v1 as tf

from nas_lib.augmentation_3d import policies

# Number of points.
P = 10000
# Number of features per point.
K = 3
# Numebr of objects.
L = 15
# Scaling constant.
W = 100


def create_rand_features():
  """Create random 3D features to process."""
  # Make a Tensor with Random Data in the following fields:
  # - features['lasers']['points_xyz'] of shape [..., 3]
  # - features['labels']['bboxes_3d'] of shape [..., 7]
  # - features['lasers']['points_xyz'] of shape [P, 3]
  # - features['lasers']['points_feature'] of shape [P, K]
  padding = [1] * P
  padding[-1] = 0  # Set one of the points_padding to 0
  bboxes_mask = [1] * L
  bboxes_mask[-1] = 0  # Set of the bboxes_mask to 0

  features = dict(
      lasers=dict(
          points_xyz=tf.random.uniform([P, 3]) *
          tf.constant(W, dtype=tf.float32),
          points_feature=tf.random.uniform([P, K]),
          points_padding=tf.convert_to_tensor(padding)
      ),
      labels=dict(
          bboxes_3d=tf.random.uniform([L, 7]) *
          tf.constant(W, dtype=tf.float32),
          bboxes_3d_mask=tf.convert_to_tensor(bboxes_mask)))
  return features


def apply_augmentation(flags_):
  """Trains and eval the NAS model."""
  # An example of how to consume the policy-spec "nas_params_str"
  # inside the trainer to augment the features.
  nas_params_dict = json.loads(flags_.nas_params_str)
  policy_spec = pg.materialize(
      search_spaces.get_search_space('augment_3d_basic'), nas_params_dict)
  policy = policies.get_policy_from_str(pg.to_json_str(policy_spec))

  # Instead of using policy suggested by the NAS service,
  # you may also manually play with simpler policies by uncommenting
  # on of the `policies` below:

  # # Single Operation Policies
  # policy = policies.RandomFlip()
  # policy = policies.WorldScaling(0.5, 1.5)
  # policy = policies.RandomRotation(0.5)
  # policy = policies.FrustumDropout()
  # policy = policies.RandomBBoxTransform(0.5)

  # # Example of using Sequential to string together multiple Ops
  # policy = policies.Sequential([
  #   policies.RandomFlip(),
  #   policies.WorldScaling(0.5, 1.5),
  #   policies.RandomRotation(0.5)
  # ])

  # Apply augmentation policy to features.
  features = create_rand_features()
  data_aug = policy(features)
  with tf.Session() as sess:
    augmentation_result = sess.run(data_aug)
  logging.info('Augmentation result: %s', augmentation_result)

  # Reporting dummy metrics back to the NAS-service.
  metric_tag = os.environ.get('CLOUD_ML_HP_METRIC_TAG', '')
  dummy_val = 0.0
  if metric_tag:
    nas_metrics_reporter = metrics_reporter.NasMetricsReporter()
    nas_metrics_reporter.report_metrics(
        hyperparameter_metric_tag=metric_tag,
        metric_value=dummy_val,
        global_step=1,
        other_metrics={})


def create_arg_parser():
  """Creates arg parser."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--job-dir', type=str, help='Job output directory.')
  parser.add_argument('--nas_params_str', type=str, help='Sampled search-space '
                      'string passed from NAS-service for current trial.')

  return parser


if __name__ == '__main__':
  tf_utils.suppress_tf_logging()
  cloud_nas_utils.setup_logging()
  flags = create_arg_parser().parse_args()
  apply_augmentation(flags)
