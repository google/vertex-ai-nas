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
"""Main function to launch Neural Architecture Search (NAS) in cloud."""

from __future__ import absolute_import

import argparse
import logging
import os
import sys

import cloud_nas_utils
import search_spaces
import tf_utils
from tf_vision import config_utils
from tf_vision import registry_imports  # pylint: disable=unused-import
from tf_vision import train_lib
from tf_vision import utils
import pyglove as pg
import six
import tensorflow as tf
from nas_architecture import tunable_autoaugment_tf2 as tunable_autoaugment


def create_arg_parser():
  """Creates arg parser."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      "--training_data_path",
      type=str,
      required=False,
      help=("The GCS path pattern for training data (tfrecord)."))
  parser.add_argument(
      "--validation_data_path",
      type=str,
      required=False,
      help=("The GCS path pattern for validation data (tfrecord)."))
  parser.add_argument(
      "--nas_params_str",
      type=str,
      help="Nas params str passed in by NAS service. "
      "It will be used to build model with `pg.materialize` function.")
  parser.add_argument(
      "--config_file", type=str, help="Configuration file path.")
  parser.add_argument(
      "--params_override", type=str, help="Configuration file path.")
  parser.add_argument("--num_cores", type=int, help="Number of TPU cores.")
  parser.add_argument("--job-dir", type=str, help="Job output directory.")
  parser.add_argument(
      "--retrain_search_job_dir",
      type=str,
      help="The job dir of the NAS search job to retrain.")
  parser.add_argument(
      "--retrain_search_job_trials",
      type=str,
      help="A list of trial IDs of the NAS search job to retrain, separated by comma."
  )
  parser.add_argument(
      "--retrain_use_search_job_checkpoint",
      type=cloud_nas_utils.str_2_bool,
      default=False,
      help="True to use previous NAS search job checkpoint."
  )
  parser.add_argument(
      "--use_tpu",
      type=cloud_nas_utils.str_2_bool,
      default=False,
      help="True to use TPU.")
  parser.add_argument(
      "--target_device_latency_ms",
      type=float,
      default=0.0,
      help="The targeting device latency in milli-seconds used for "
      "device-aware Neural Architecture Search. If it is 0, model latency is "
      "not considered in search.")
  parser.add_argument(
      "--target_memory_mb",
      type=int,
      default=0,
      help="The target model memory in megabytes.")
  parser.add_argument(
      "--target_flops_multi_adds_billion",
      type=float,
      default=0.0,
      help="The targeting FLOPS in multi-adds-billions used for "
      "device-aware Neural Architecture Search. If it is 0, model latency is "
      "not considered in search.")
  parser.add_argument(
      "--model",
      type=str,
      default="retinanet",
      choices=[
          "retinanet", "segmentation", "classification", "mask_rcnn",
          "pointpillars", "pointpillars_baseline",
      ],
      help="Which model to specify.")
  parser.add_argument(
      "--job_mode",
      type=str,
      default="train",
      choices=["train", "train_and_eval"],
      help="Mode to run: `train` or `train_and_eval`.")
  parser.add_argument(
      "--skip_nan_error",
      type=cloud_nas_utils.str_2_bool,
      default=False,
      help="True to ignore NAN error during training and this trial will not "
      "be counted for `maxFailedNasTrials`.")
  parser.add_argument(
      "--skip_eval",
      type=cloud_nas_utils.str_2_bool,
      default=False,
      help="True to skip eval.")
  parser.add_argument(
      "--multiple_eval_during_search",
      type=cloud_nas_utils.str_2_bool,
      default=False,
      help="True to run multiple evaluations "
      "for each trial during search, using the eval.num_steps_per_eval config "
      "parameter to determine the interval. In this case, only the best "
      "metric over all evaluations is reported to the controller.")
  parser.add_argument(
      "--search_space",
      type=str,
      default="nasfpn",
      choices=[
          "nasfpn",
          "spinenet",
          "spinenet_v2",
          "spinenet_mbconv",
          "mnasnet",
          "efficientnet_v2",
          "pointpillars",
          "randaugment_detection",
          "randaugment_segmentation",
          "autoaugment_detection",
          "autoaugment_segmentation",
          "spinenet_scaling",
      ],
      help="The choice of NAS search space, e.g., nasfpn.",
  )
  return parser


def main(FLAGS):  

  FLAGS.job_dir = cloud_nas_utils.get_job_dir_from_environment_if_exist(
      current_job_dir=FLAGS.job_dir)

  trial_id = cloud_nas_utils.get_trial_id_from_environment()
  hp_metric_tag = os.environ.get("CLOUD_ML_HP_METRIC_TAG", "")
  logging.info("Starting trial %s for hypertuning metric %s", trial_id,
               hp_metric_tag)

  # Set up constraints if any.
  constraint_type, target_latency_or_flops = utils.set_up_constraints(
      FLAGS.target_device_latency_ms, FLAGS.target_flops_multi_adds_billion)

  if not FLAGS.nas_params_str:
    raise RuntimeError("FLAG nas_params_str cannot be empty.")

  if FLAGS.retrain_search_job_trials:
    # Resets `nas_params_str` if this job is to retrain a previous NAS trial.
    FLAGS.nas_params_str = cloud_nas_utils.get_finetune_nas_params_str(
        retrain_search_job_trials=FLAGS.retrain_search_job_trials,
        retrain_search_job_dir=FLAGS.retrain_search_job_dir)

  logging.info("nas_params_str is %s", FLAGS.nas_params_str)

  if not tf.io.gfile.exists(FLAGS.job_dir):
    tf.io.gfile.makedirs(FLAGS.job_dir)

  tunable_functor_or_object = cloud_nas_utils.parse_and_save_nas_params_str(
      search_spaces.get_search_space(FLAGS.search_space), FLAGS.nas_params_str,
      FLAGS.job_dir)

  if FLAGS.search_space in [
      "spinenet_scaling", "randaugment_detection", "randaugment_segmentation"
  ]:
    # Scaling and augmentation search-spaces are defined as pg.Object.
    tunable_object = tunable_functor_or_object
  elif FLAGS.search_space == "autoaugment_detection":
    tunable_object = tunable_autoaugment.AutoAugmentDetectionBuilder(
        tunable_functor_or_object)()
  elif FLAGS.search_space == "autoaugment_segmentation":
    tunable_object = tunable_autoaugment.AutoAugmentSegmentationBuilder(
        tunable_functor_or_object)()
  else:
    # Other search spaces (like `tunable_spinenet.build_tunable_block_specs()``)
    # are defined as a `pg.functor`. Calling it to return a pg.Object.
    tunable_object = tunable_functor_or_object()

  # Creates a serialized `tunable_object` to pass it down through to the
  # layer builder.
  serialized_tunable_object = pg.to_json_str(
      tunable_object, json_indent=2, hide_default_values=False)

  # Outputs `serialized_tunable_object_file` so that it can be re-used to create
  # static NAS searched static component for custom trainer, e.g.,
  # `tunable_spinenet.tunable_spinenet_builder()`. This provides the second
  # option (in addition to `FLAGS.nas_params_str`) to rebuild the searched model
  # for finetuning or exporting to SavedModel.
  serialized_tunable_object_file = os.path.join(
      FLAGS.job_dir,
      "{}_serialized_tunable_object.json".format(FLAGS.search_space))
  if not tf.io.gfile.exists(serialized_tunable_object_file):
    with tf.io.gfile.GFile(serialized_tunable_object_file, "w") as f:
      f.write(serialized_tunable_object)

  # Get cloud-TPU master address.
  if FLAGS.use_tpu:
    tpu_cluster_resolver = cloud_nas_utils.wait_for_tpu_cluster_resolver_ready(
        tf.distribute.cluster_resolver.TPUClusterResolver)
    tpu_address = tpu_cluster_resolver.get_master()
  else:
    tpu_address = None

  params = config_utils.create_params(
      FLAGS,
      FLAGS.search_space,
      serialized_tunable_object,
      tpu_address=tpu_address)
  if FLAGS.retrain_use_search_job_checkpoint:
    prev_checkpoint_dir = cloud_nas_utils.get_retrain_search_job_model_dir(
        retrain_search_job_trials=FLAGS.retrain_search_job_trials,
        retrain_search_job_dir=FLAGS.retrain_search_job_dir)
    logging.info("Setting checkpoint to %s.", prev_checkpoint_dir)
    params.task.init_checkpoint = prev_checkpoint_dir
    params.task.init_checkpoint_modules = "all"

  session = cloud_nas_utils.CloudSession(model_dir=FLAGS.job_dir)
  try:
    train_lib.run_training(
        params=params,
        session=session,
        mode=FLAGS.job_mode,
        model=FLAGS.model,
        model_dir=FLAGS.job_dir,
        constraint_type=constraint_type,
        target_latency_or_flops=target_latency_or_flops,
        target_memory=FLAGS.target_memory_mb,
        skip_eval=FLAGS.skip_eval,
        multiple_eval_during_search=FLAGS.multiple_eval_during_search,
        retrain_search_job_trials=FLAGS.retrain_search_job_trials)
    cloud_nas_utils.write_job_status(FLAGS.job_dir,
                                     cloud_nas_utils.JOB_STATUS_SUCCESS)
  except Exception as e:  # pylint: disable=broad-except
    if "NaN" in str(e):
      cloud_nas_utils.write_job_status(
          FLAGS.job_dir, cloud_nas_utils.JOB_STATUS_FAILED_WITH_NAN_ERROR)
      if FLAGS.skip_nan_error:
        # In case of `skip_nan_error`, do not raise NaN to fail the trial, so it
        # will not be counted toward `maxFailedNasTrials` setting in the API.
        logging.warning(
            ("Trial failed due to NaN error, however the NaN error does not",
             " count for `maxFailedNasTrials`."))
      else:
        six.reraise(*sys.exc_info())
    else:
      cloud_nas_utils.write_job_status(FLAGS.job_dir,
                                       cloud_nas_utils.JOB_STATUS_FAILED)
      six.reraise(*sys.exc_info())


if __name__ == "__main__":
  tf_utils.suppress_tf_logging()
  cloud_nas_utils.setup_logging()
  flags = create_arg_parser().parse_args()
  main(flags)
