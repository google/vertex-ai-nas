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
"""Training script for medical-3D."""

from __future__ import absolute_import

import argparse
import importlib
import logging
import os
import sys

import cloud_nas_utils
import metrics_reporter
from gcs_utils import gcs_path_utils
from pytorch import logging_utils
from third_party.medical_3d import medical_3d_search_spaces
from third_party.medical_3d import static_unet
from third_party.medical_3d import tunable_unet
from third_party.medical_3d import tunable_unet_nasfpn
from monai.utils import set_determinism
from third_party.medical_3d import monai_trainer
import pyglove as pg
import six
import tensorflow.compat.v1 as tf
import yaml

# Fixed seed to get repeatable training results. Randomness in training result
# can give noisy signal to the NAS controller.
_DETERMINISM_SEED = 0


def str_2_bool(v):
  """Auxiliary function to support boolean command-line arguments."""
  if not isinstance(v, str):
    raise ValueError("{} is not string type".format(v))
  return v.lower() == "true"


def create_arg_parser():
  """Creates arg parser."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # Nas-service related flags.
  parser.add_argument(
      "--nas_params_str", type=str, help="Nas args serialized in JSON string.")
  parser.add_argument(
      "--job-dir", type=str, default="tmp", help="Job output directory.")
  parser.add_argument("--search_space", type=str, help="Search space tag.")
  parser.add_argument(
      "--retrain_search_job_dir",
      type=str,
      help="The job dir of the NAS search job to retrain.")
  parser.add_argument(
      "--retrain_search_job_trials",
      type=str,
      help="A list of trial IDs of the NAS search job to retrain, "
      "separated by comma.")

  # Flags specific to this docker.
  parser.add_argument(
      "--num_epochs", type=int, help="Number of training epochs.")
  parser.add_argument(
      "--val_interval",
      type=int,
      default=15,
      help="Number of epochs after which a validation is run. "
      "NOTE: Validation will be run by deafult for the last epoch.")
  parser.add_argument(
      "--num_gpus", type=int, default=0, help="Number of GPUs for training.")
  parser.add_argument(
      "--config_file", type=str, help="Configuration file path for the model.")
  parser.add_argument(
      "--dataset_module",
      type=str,
      help="Dataset module and its optional arguments passed as a string. "
      "For ex: --dataset_module='msd_brats_dataset.fast_dataset; arg_string'. "
      "The dataset module will internally handle the argument string.")
  parser.add_argument("--batch_size", type=int, help="Batch-size.")
  parser.add_argument(
      "--moving_average_decay",
      type=float,
      default=0.998,
      help="Moving-average-decay parameter used for "
      "stochastic-weighted-averaging. Using this functionality reduces the "
      "noise in the training curve. Setting this parameter to 0.0 will shut "
      "down the use of stochastic-weighted-averaging.")
  parser.add_argument(
      "--metric_scaling_factor",
      type=float,
      default=10.0,
      help="The scaling factor multiplied with the final metric before "
      "it gets reported to the NAS controller. For example, a dice score of "
      "0.682 will get reported as 6.82 if the scaling factor is 10. This may "
      "help the controller if the difference in the metrics is low. The final "
      "scaled value reported tot eh controller should be in the range "
      "1e-3 and 10.")
  parser.add_argument(
      "--init_learning_rate",
      type=float,
      default=1e-4,
      help="Initial learning rate.")
  parser.add_argument(
      "--learning_rate_scheduler",
      type=str,
      choices=["none", "linear_decay"],
      default="none",
      help="Choice of learning rate scheduler. NOTE: choice of 'none' means "
      "that the learning rate will be controlled by the default optimizer.")
  return parser


def call_module(module_arg):
  """Calls module given its name and optional arg-string."""
  # First split the input to see if we have an argument to pass.
  module_items = module_arg.split(";", 1)
  module_path = module_items[0].strip()
  module_arg = module_items[1].strip() if len(module_items) > 1 else None
  module_file, module_method_name = module_path.rsplit(".", 1)
  module = importlib.import_module(module_file)
  method = getattr(module, module_method_name)
  if module_arg:
    return method(module_arg)
  else:
    return method()


def main(FLAGS):  
  # Set determinism via MONAI.
  set_determinism(seed=_DETERMINISM_SEED)

  trial_id = cloud_nas_utils.get_trial_id_from_environment()
  logging.info("Starting trial %s.", trial_id)

  FLAGS.job_dir = cloud_nas_utils.get_job_dir_from_environment_if_exist(
      current_job_dir=FLAGS.job_dir)
  model_dir = gcs_path_utils.gcs_fuse_path(FLAGS.job_dir)
  # Create job dir.
  logging.info("Job dir is %s.", FLAGS.job_dir)
  if not tf.gfile.Exists(FLAGS.job_dir):
    tf.gfile.MakeDirs(FLAGS.job_dir)
  FLAGS.job_dir = gcs_path_utils.gcs_fuse_path(FLAGS.job_dir)

  if FLAGS.retrain_search_job_trials:
    # Resets `nas_params_str` if this job is to retrain a previous NAS trial.
    FLAGS.nas_params_str = cloud_nas_utils.get_finetune_nas_params_str(
        retrain_search_job_trials=FLAGS.retrain_search_job_trials,
        retrain_search_job_dir=FLAGS.retrain_search_job_dir)

  # Process nas_params_str passed by the NAS-service.
  # This gives one instance of the search-space to be used for this trial.
  logging.info("search_space is %s", FLAGS.search_space)
  logging.info("nas_params_str passed by NAS-service is %s",
               FLAGS.nas_params_str)
  tunable_functor_or_object = cloud_nas_utils.parse_and_save_nas_params_str(
      medical_3d_search_spaces.get_search_space(FLAGS.search_space),
      FLAGS.nas_params_str, model_dir)
  if FLAGS.search_space == "tunable_unet_nasfpn":
    tunable_object = tunable_functor_or_object()
  else:
    tunable_object = tunable_functor_or_object
  serialized_tunable_object = pg.to_json_str(
      tunable_object, json_indent=2, hide_default_values=False)
  logging.info("serialized_tunable_object for this trial is %s",
               serialized_tunable_object)
  serialized_tunable_object_file = os.path.join(
      model_dir, "{}_serialized_tunable_object.json".format(FLAGS.search_space))
  with open(serialized_tunable_object_file, "w") as f:
    f.write(serialized_tunable_object)

  # Now train.
  try:
    #################################################
    # Add your training code here.
    # Use the parameters in tunable_object passed by the NAS service
    # to build your model for this trial.
    train_ds, val_ds, in_channels, out_channels, val_roi_size = call_module(
        FLAGS.dataset_module)
    with open(gcs_path_utils.gcs_fuse_path(FLAGS.config_file), "r") as f:
      params = yaml.load(f, Loader=yaml.FullLoader)
      params["in_channels"] = in_channels
      params["out_channels"] = out_channels
    if FLAGS.search_space == "static_unet":
      model = static_unet.unet(params)
    elif FLAGS.search_space == "tunable_unet_nasfpn":
      params["block_specs_json"] = serialized_tunable_object
      model = tunable_unet_nasfpn.tunable_unet_nasfpn(params)
    elif FLAGS.search_space == "tunable_unet_encoder":
      params["block_specs_json"] = serialized_tunable_object
      model = tunable_unet.tunable_unet(params)
    else:
      raise ValueError("Unknown search-space: %s" % FLAGS.search_space)
    # Save config file to trial-folder.
    with open(
        os.path.join(FLAGS.job_dir, "params.yaml"), "w"
    ) as f:
      params = yaml.dump(params, f)
    # Train.
    metric_values = monai_trainer.train(
        train_ds=train_ds,
        val_ds=val_ds,
        sliding_window_val_roi_size=val_roi_size,
        output_dir=FLAGS.job_dir,
        model=model,
        num_gpus=FLAGS.num_gpus,
        epoch_num=FLAGS.num_epochs,
        val_interval=FLAGS.val_interval,
        batch_size=FLAGS.batch_size,
        moving_average_decay=FLAGS.moving_average_decay,
        learning_rate=FLAGS.init_learning_rate,
        learning_rate_scheduler=FLAGS.learning_rate_scheduler)
    #################################################
    cloud_nas_utils.write_job_status(model_dir,
                                     cloud_nas_utils.JOB_STATUS_SUCCESS)
  except Exception as e:  # pylint: disable=broad-except
    if "NaN" in str(e):
      cloud_nas_utils.write_job_status(
          model_dir, cloud_nas_utils.JOB_STATUS_FAILED_WITH_NAN_ERROR)
      if FLAGS.skip_nan_error:
        # In case of `skip_nan_error`, do not raise NaN to fail the trial, so it
        # will not be counted toward `maxFailedNasTrials` setting in the API.
        print(("Trial failed due to NaN error, however the NaN error does not",
               " count for `maxFailedNasTrials`."))
      else:
        six.reraise(*sys.exc_info())
    else:
      cloud_nas_utils.write_job_status(model_dir,
                                       cloud_nas_utils.JOB_STATUS_FAILED)
      six.reraise(*sys.exc_info())

  # Report metrics back to NAS-service after training.
  # NOTE: Reporting maximum metric value back to controller.
  metric_tag = os.environ.get("CLOUD_ML_HP_METRIC_TAG", "")
  max_metric_val = max([metric["val"] for metric in metric_values])
  reported_metric = max_metric_val * FLAGS.metric_scaling_factor
  if (
      reported_metric < cloud_nas_utils.MIN_ALLOWED_METRIC
      or reported_metric > cloud_nas_utils.MAX_ALLOWED_METRIC
  ):
    logging.warning(
        "The metric %f reported to NAS controller is outside of allowed "
        "range of [%f, %f].",
        reported_metric, cloud_nas_utils.MIN_ALLOWED_METRIC,
        cloud_nas_utils.MAX_ALLOWED_METRIC)
  num_epochs = metric_values[-1]["epoch"]
  other_metrics = {}
  if FLAGS.retrain_search_job_trials:
    other_metrics[
        "nas_trial_id"] = cloud_nas_utils.get_search_trial_id_to_finetune(
            FLAGS.retrain_search_job_trials)
  if metric_tag:
    nas_metrics_reporter = metrics_reporter.NasMetricsReporter()
    nas_metrics_reporter.report_metrics(
        hyperparameter_metric_tag=metric_tag,
        metric_value=reported_metric,
        global_step=num_epochs,
        other_metrics=other_metrics)


if __name__ == "__main__":
  logging_utils.setup_logging()
  flags = create_arg_parser().parse_args()
  main(flags)
