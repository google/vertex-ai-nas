# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
import enum
import json
import logging
import os
import pprint
import sys

import cloud_nas_utils
import search_spaces
import tf_utils
import pyglove as pg
import six
import tensorflow.compat.v1 as tf
import yaml

from tf1 import export_saved_model
from tf1.detection.configs import factory
from tf1.detection.dataloader import input_reader
from tf1.detection.dataloader import mode_keys as ModeKeys
from tf1.detection.executor import tpu_executor
from tf1.detection.modeling import model_builder
from nas_architecture import tunable_autoaugment
from tf1.hyperparameters import params_dict
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver


# Keys used in model analysis file.
_MODEL_LATENCY_MILLI = "latency_milli_seconds"
_MODEL_FLOPS = "multi_add_flops_billion"
_MODEL_MEMORY = "model_memory"
_MODEL_MAP = "AP"
_MODEL_TOP1_ACCURACY = "top_1_accuracy"
_MODEL_MIOU = "miou"


class LatencyConstraintType(enum.Enum):
  NONE = 1
  DEVICE_MILLI_SEC = 2
  FLOPS_HALF_ADDS_BILLION = 3


def _save_config(params, model_dir):
  if model_dir:
    if not tf.gfile.Exists(model_dir):
      tf.gfile.MakeDirs(model_dir)
    params_dict.save_params_dict_to_yaml(params,
                                         os.path.join(model_dir, "params.yaml"))


@pg.members([
    ("step", pg.typing.Int(), "At which step the result is reported."),
    ("metrics",
     pg.typing.Dict([(pg.typing.StrKey(), pg.typing.Float(), "Metric item.")
                    ]).noneable(), "Metric in key/value pairs (optional)."),
])
class Measurement(pg.Object):
  """Measurement of a trial at certain step."""


def create_arg_parser():
  """Creates arg parser."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      "--training_data_path",
      type=str,
      required=False,
      help=("Comma separated path patterns for training data (tfrecord)."))
  parser.add_argument(
      "--validation_data_path",
      type=str,
      required=False,
      help=(
          "Comma separated path patterns for validation data (tfrecord)."))
  parser.add_argument(
      "--nas_params_str",
      type=str,
      help="Nas params str passed in by NAS service. "
      "It will be used to build model with `pg.materialize` function.")
  parser.add_argument(
      "--config_file", type=str, help="Configuration file path.")
  parser.add_argument(
      "--params_override", type=str, help="Configuration file path.")
  parser.add_argument(
      "--tpu",
      type=str,
      help="The Cloud TPU to use for training. This should be either the name "
      "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
      " url.")
  parser.add_argument(
      "--gcp_project",
      type=str,
      help="Project name for the Cloud TPU-enabled project. If not specified, "
      "we will attempt to automatically detect the GCE project from metadata.")
  parser.add_argument(
      "--tpu_zone",
      type=str,
      help="GCE zone where the Cloud TPU is located in. If not specified, we "
      "will attempt to automatically detect the GCE project from metadata.")
  parser.add_argument(
      "--eval_master",
      type=str,
      help="GRPC URL of the eval master. Set to an appropiate value when  "
      "running on CPU/GPU.")
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
      choices=["retinanet", "segmentation", "classification", "mask_rcnn"],
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
          "nasfpn", "spinenet", "spinenet_v2", "spinenet_mbconv", "mnasnet",
          "efficientnet_v2",
          "randaugment_detection", "randaugment_segmentation",
          "autoaugment_detection", "autoaugment_segmentation",
          "spinenet_scaling"
      ],
      help="The choice of NAS search space, e.g., nasfpn.")
  return parser


def _get_file_pattern_list(file_patterns):
  """Returns a list of file-patterns partitioned over comma."""
  return [pattern.strip() for pattern in file_patterns.split(",")]


def create_params(args, search_space, serialized_tunable_object):
  """Returns `ParamsDict` based on `args` and `serialized_tunable_object`."""

  params = factory.config_generator(args.model)

  if args.config_file:
    params = params_dict.override_params_dict(
        params, args.config_file, is_strict=True)

  params = params_dict.override_params_dict(
      params, args.params_override, is_strict=True)

  if search_space == "nasfpn":
    params.override({
        "architecture": {
            "multilevel_features": "tunable_nasfpn_v2",
        },
        "tunable_nasfpn_v2": {
            "block_specs": serialized_tunable_object
        }
    })
  elif search_space in ["spinenet", "spinenet_v2"]:
    params.override({
        "architecture": {
            "backbone": "tunable_spinenet",
            "multilevel_features": "identity"
        },
        "tunable_spinenet": {
            "block_specs": serialized_tunable_object
        },
    })
  elif search_space == "spinenet_mbconv":
    params.override({
        "architecture": {
            "backbone": "tunable_spinenet_mbconv",
            "multilevel_features": "identity"
        },
        "tunable_spinenet_mbconv": {
            "block_specs": serialized_tunable_object,
        }
    })
  elif search_space == "mnasnet":
    params.override({
        "architecture": {
            "backbone": "tunable_mnasnet",
        },
        "tunable_mnasnet": {
            "block_specs": serialized_tunable_object
        },
    })
  elif search_space == "efficientnet_v2":
    params.override({
        "architecture": {
            "backbone": "tunable_efficientnet_v2",
        },
        "tunable_efficientnet_v2": {
            "block_specs": serialized_tunable_object
        },
    })
  elif search_space == "randaugment_detection":
    params.override(
        {"retinanet_parser": {
            "aug_policy": serialized_tunable_object,
        }})
  elif search_space == "randaugment_segmentation":
    params.override(
        {"segmentation_parser": {
            "aug_policy": serialized_tunable_object,
        }})
  elif search_space == "autoaugment_detection":
    params.override(
        {"retinanet_parser": {
            "aug_policy": serialized_tunable_object,
        }})
  elif search_space == "autoaugment_segmentation":
    params.override(
        {"segmentation_parser": {
            "aug_policy": serialized_tunable_object,
        }})
  elif search_space == "spinenet_scaling":
    scaling_config = pg.from_json_str(serialized_tunable_object)
    scaling_model_spec_params = {
        "retinanet_parser": {
            "output_size": [
                scaling_config.image_size, scaling_config.image_size
            ],
        },
        "architecture": {
            "backbone": "tunable_spinenet",
            "multilevel_features": "identity",
        },
        "tunable_spinenet": {
            "endpoints_num_filters": scaling_config.endpoints_num_filters,
            "resample_alpha": scaling_config.resample_alpha,
            "block_repeats": scaling_config.block_repeats,
            "filter_size_scale": scaling_config.filter_size_scale,
        },
        "retinanet_head": {
            "num_convs": scaling_config.head_num_convs,
            "num_filters": scaling_config.head_num_filters,
        }
    }

    # Write scaling parameters as yaml to job_dir, so that it can be reused
    # later (with --params_override flag).
    scaling_params_file = os.path.join(args.job_dir, "scaling_params.yaml")
    with tf.gfile.GFile(scaling_params_file, "w") as f:
      yaml.dump(scaling_model_spec_params, f)

    params.override(scaling_model_spec_params)
  else:
    raise ValueError("Unexpected search_space {}".format(search_space))

  params.override(
      {
          "isolate_session_state": True,
          "architecture": {
              # Use bfloat16 for TPU as default.
              "use_bfloat16": True,
          },
          "platform": {
              "eval_master": args.eval_master,
              "tpu": args.tpu,
              "tpu_zone": args.tpu_zone,
              "gcp_project": args.gcp_project,
          },
          "tpu_job_name": "",
          "use_tpu": args.use_tpu,
          "model_dir": args.job_dir,
          "batch_norm_activation": {
              # Use sync batchnorm for TPU as default.
              "use_sync_bn": True,
          },
          "train": {
              "num_shards":
                  args.num_cores,
              "train_file_pattern":
                  _get_file_pattern_list(args.training_data_path),
              # You can adjust the `gradient_clip_norm` for your own use case.
              # Training may get NaN if its value is too big.
              "gradient_clip_norm": 1,
          },
          "eval": {
              "eval_file_pattern":
                  _get_file_pattern_list(args.validation_data_path),
              "num_steps_per_eval":
                  3000,
              "use_json_file":
                  False,
          },
          "retinanet_parser": {
              # To ensure that source_id is integer format for TPU.
              "regenerate_source_id": True,
          },
          "enable_summary": True,
      },
      is_strict=False)

  if not args.use_tpu:
    params.override(
        {
            "architecture": {
                "use_bfloat16": False,
            },
            "batch_norm_activation": {
                "use_sync_bn": False,
            },
        },
        is_strict=True)

  # Only run spatial partitioning in training mode.
  params.train.input_partition_dims = None
  params.train.num_cores_per_replica = None

  params.validate()
  params.lock()
  pp = pprint.PrettyPrinter()
  params_str = pp.pformat(params.as_dict())
  logging.info("Model Parameters: %s", params_str)
  return params


def _get_model_latency(params, latest_checkpoint):
  """Returns the latency of generated SavedModel."""
  model = params.type
  params_for_saved_model = params_dict.ParamsDict(params)
  params_for_saved_model.batch_norm_activation.use_sync_bn = False
  if model == "retinanet":
    # Use batched non-max-suppression to speed up.
    params_for_saved_model.postprocess.nms_version = "batched"

  saved_model_path = cloud_nas_utils.get_saved_model_dir(
      trial_job_dir=params_for_saved_model.model_dir)

  # Set export model size using model params.
  export_model_size = []
  if model == "segmentation":
    export_model_size = params_for_saved_model.segmentation_parser.output_size
  elif model == "retinanet":
    export_model_size = params_for_saved_model.retinanet_parser.output_size
  elif model == "classification":
    export_model_size = params_for_saved_model.classification_parser.output_size
  else:
    logging.error("Could not find export_model_size.")
    raise ValueError("Invalid model: %s" % model)
  logging.info("export_model_size: %s", export_model_size)

  export_saved_model.export(
      export_dir=saved_model_path,
      checkpoint_path=latest_checkpoint,
      model=model,
      config_file="",
      params_override=params_for_saved_model.as_dict(),
      image_size=export_model_size,
      input_type="image_bytes",
      output_normalized_coordinates=True,
      cast_num_detections_to_float=True)
  tf.logging.info("Exported SavedModel at %s", saved_model_path)

  # Wait for another process to compute latency.
  model_stats = cloud_nas_utils.wait_and_retrieve_latency(saved_model_path)
  latency_in_seconds = model_stats["latency_in_seconds"]
  model_memory = model_stats["model_memory"]
  return latency_in_seconds, model_memory


def _run_train_and_eval_in_loop(params, executor, train_input_fn,
                                eval_input_fn):
  """Runs train and eval in for loop and returns the list of all the metrics."""
  step = params.eval.num_steps_per_eval
  total = params.train.total_steps
  # Get a list of current-last-step at each iteration.
  # The last step is always included in the list.
  current_last_step_list = list(range(step, total, step)) + [total]
  logging.info("Running for %s num_cycles.", len(current_last_step_list))
  metrics_list = []
  for cycle, current_last_step in enumerate(current_last_step_list):
    logging.info("Start training cycle %d.", cycle)
    executor.train(train_input_fn, current_last_step)
    eval_times = (
        params.eval.eval_samples //
        params.eval.eval_batch_size if params.eval.eval_samples else None)
    metrics = executor.evaluate(eval_input_fn, eval_times)
    metrics_list.append(metrics)
  return metrics_list


def _get_accuracy_key(task_type):
  """Given the task type, returns the accuracy tag in evaluation metrics."""
  if task_type == "segmentation":
    return _MODEL_MIOU
  elif task_type == "classification":
    return _MODEL_TOP1_ACCURACY
  else:
    return _MODEL_MAP


def run_training(params,
                 session,
                 job_mode,
                 constraint_type=LatencyConstraintType.NONE,
                 target_latency_or_flops=0.0,
                 target_memory=0,
                 skip_eval=False,
                 multiple_eval_during_search=False,
                 retrain_search_job_trials=None):
  """Runs model training."""
  model_fn = model_builder.ModelFn(params)
  executor = tpu_executor.TpuExecutor(
      model_fn,
      params,
      cloud_nas_utils.wait_for_tpu_cluster_resolver_ready(
          contrib_cluster_resolver.TPUClusterResolver),
      # Sets keep_checkpoint_max to None to save all checkpoints.
      keep_checkpoint_max=None)

  train_input_fn = input_reader.InputFn(
      params.train.train_file_pattern, params, mode=ModeKeys.TRAIN)

  if params.type == "segmentation" or params.type == "classification":
    # For classification/segmentation, the eval is done directly.
    eval_input_fn = input_reader.InputFn(
        params.eval.eval_file_pattern, params, mode=ModeKeys.EVAL)
  else:
    # For detection, the evaluation is done after prediction.
    eval_input_fn = input_reader.InputFn(
        params.eval.eval_file_pattern, params, mode=ModeKeys.PREDICT_WITH_GT)

  # Save configurations to trial directory.
  _save_config(params, params.model_dir)

  # Set-up model metrics like mAP, FLOPS, etc.
  model_analysis = {}
  accuracy_key = _get_accuracy_key(task_type=params.type)
  model_flops = 0.0
  if job_mode == "train":
    if skip_eval:
      # This is only used during debugging.
      executor.train(train_input_fn, params.train.total_steps)
      return

    if multiple_eval_during_search or retrain_search_job_trials:
      # We want to enable multiple evaluations when either the flag
      # '--multiple_eval_during_search' is explicitly set to true, or it's a
      # stage-2 train job (which also gets triggered similar to a search job
      # with multiple trials).
      # For proxy task with large number of epochs (for ex: augmentation-search)
      # we need to see eval-etrics in Tensorboard. So we need to run
      # multiple evaluations and choose the best metric to report.
      metrics_list = _run_train_and_eval_in_loop(
          params=params,
          executor=executor,
          train_input_fn=train_input_fn,
          eval_input_fn=eval_input_fn)
      tf.logging.info("Metrics list is: %s.", metrics_list)
      metrics = max(metrics_list, key=lambda x: x[accuracy_key])
      tf.logging.info("Best metrics is: %s.", metrics)
    else:
      # For model-search, the proxy task is a small training job which should
      # run fast. So we just run one evaluation.
      executor.train(train_input_fn, params.train.total_steps)
      eval_times = (
          params.eval.eval_samples //
          params.eval.eval_batch_size if params.eval.eval_samples else None)
      metrics = executor.evaluate(eval_input_fn, eval_times)
    model_analysis[accuracy_key] = float(metrics[accuracy_key])

    # Get Model Flops.
    with tf.gfile.Open(
        os.path.join(params.model_dir, "train_model_stats.json"), "r") as fp:
      model_stats_dict = json.load(fp)
    tf.logging.info("Loaded train model: %s multi-adds billion FLOPS.",
                    model_stats_dict["multi_add_flops_billion"])
    model_flops = model_stats_dict["multi_add_flops_billion"]
    model_memory = 0

    # Get inference latency.
    if constraint_type == LatencyConstraintType.DEVICE_MILLI_SEC:
      latency_sec, model_memory = _get_model_latency(
          params, executor._estimator.latest_checkpoint()  # pylint: disable=protected-access
          )
      measured_latency_or_flops = latency_sec * 1000
      model_analysis[_MODEL_LATENCY_MILLI] = measured_latency_or_flops
      model_analysis[_MODEL_MEMORY] = model_memory
      tf.logging.info("Model latency is %s milli-seconds.",
                      measured_latency_or_flops)
    elif constraint_type == LatencyConstraintType.FLOPS_HALF_ADDS_BILLION:
      measured_latency_or_flops = model_stats_dict["multi_add_flops_billion"]
    # Update metrics-reward with latency/FLOPS before reporting to controller.
    if constraint_type != LatencyConstraintType.NONE:
      original_accuracy = float(metrics[accuracy_key])
      metrics[accuracy_key] = cloud_nas_utils.compute_reward(
          accuracy=original_accuracy,
          target_latency_or_flops=target_latency_or_flops,
          measured_latency_or_flops=measured_latency_or_flops,
          target_memory=target_memory,
          measured_memory=model_memory)
      tf.logging.info(
          ("Original accuracy [%s], accuracy_with_constraint [%s] with "
           "target_latency_or_flops [%s] and measured_latency_or_flops [%s]"),
          original_accuracy, metrics[accuracy_key], target_latency_or_flops,
          measured_latency_or_flops)

    # Report measurement to the controller.
    measurement = Measurement(
        step=params.train.total_steps,
        metrics={k: float(v) for k, v in six.iteritems(metrics)})

    if retrain_search_job_trials:
      model_analysis[
          "nas_trial_id"] = cloud_nas_utils.get_search_trial_id_to_finetune(
              retrain_search_job_trials)

    # Other metrics can also be reported for analysis,
    session.report(
        measurement, model_flops=model_flops, other_metrics=model_analysis)

    # Save model-analysis to disk.
    tf.logging.info("Writing model_analysis.json: %s", model_analysis)
    with tf.gfile.Open(
        os.path.join(params.model_dir, "model_analysis.json"), "w") as fp:
      json.dump(model_analysis, fp, indent=2, sort_keys=True)

  elif job_mode == "train_and_eval":
    # In NAS stage 2, we need to
    # evaluate continuously so that Tensorboard can show the curves of eval
    # metrics. Besides, we do not need to report rewards for `train_and_eval`
    # mode since `reward` is only needed by NAS controller in NAS stage 1 jobs.
    _ = _run_train_and_eval_in_loop(
        params=params,
        executor=executor,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn)
  else:
    raise ValueError("Unsupported job_mode: {}".format(job_mode))


def _set_up_constraints(target_device_latency_ms,
                        target_flops_multi_adds_billion):
  """Sets up constarint based on the user flags."""
  constraint_type = LatencyConstraintType.NONE
  target_constraint = 0.0
  if target_device_latency_ms > 0.0 and target_flops_multi_adds_billion > 0.0:
    raise RuntimeError("Can not use both device-latency and FLOPS as the "
                       "latency constraint. Please choose one.")
  elif target_device_latency_ms > 0.0:
    constraint_type = LatencyConstraintType.DEVICE_MILLI_SEC
    target_constraint = target_device_latency_ms
  elif target_flops_multi_adds_billion > 0.0:
    constraint_type = LatencyConstraintType.FLOPS_HALF_ADDS_BILLION
    target_constraint = target_flops_multi_adds_billion
  tf.logging.info("Constraint is (%s, %s).", constraint_type, target_constraint)
  return constraint_type, target_constraint


def main(FLAGS):  

  FLAGS.job_dir = cloud_nas_utils.get_job_dir_from_environment_if_exist(
      current_job_dir=FLAGS.job_dir)

  # Set the default to be trial 0.
  trial_id = os.environ.get("CLOUD_ML_TRIAL_ID", 0)
  hp_metric_tag = os.environ.get("CLOUD_ML_HP_METRIC_TAG", "")
  tf.logging.info("Starting trial %s for hypertuning metric %s", trial_id,
                  hp_metric_tag)

  # Set up constraints if any.
  constraint_type, target_latency_or_flops = _set_up_constraints(
      FLAGS.target_device_latency_ms, FLAGS.target_flops_multi_adds_billion)

  if not FLAGS.nas_params_str:
    raise RuntimeError("FLAG nas_params_str cannot be empty.")

  if FLAGS.retrain_search_job_trials:
    # Resets `nas_params_str` if this job is to retrain a previous NAS trial.
    FLAGS.nas_params_str = cloud_nas_utils.get_finetune_nas_params_str(
        retrain_search_job_trials=FLAGS.retrain_search_job_trials,
        retrain_search_job_dir=FLAGS.retrain_search_job_dir)

  tf.logging.info("nas_params_str is %s", FLAGS.nas_params_str)

  if not tf.gfile.Exists(FLAGS.job_dir):
    tf.gfile.MakeDirs(FLAGS.job_dir)

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
  # layer builder. Then `models/detection/modeling/architecture/factory.py` will
  # create model based on `serialized_tunable_object`, e.g.,
  # `tunable_spinenet.tunable_spinenet_builder()`.
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
  if not tf.gfile.Exists(serialized_tunable_object_file):
    with tf.gfile.GFile(serialized_tunable_object_file, "w") as f:
      f.write(serialized_tunable_object)

  params = create_params(FLAGS, FLAGS.search_space, serialized_tunable_object)

  session = cloud_nas_utils.CloudSession(model_dir=params.model_dir)
  try:
    run_training(
        params,
        session,
        FLAGS.job_mode,
        constraint_type=constraint_type,
        target_latency_or_flops=target_latency_or_flops,
        target_memory=FLAGS.target_memory_mb,
        skip_eval=FLAGS.skip_eval,
        multiple_eval_during_search=FLAGS.multiple_eval_during_search,
        retrain_search_job_trials=FLAGS.retrain_search_job_trials)
    cloud_nas_utils.write_job_status(params.model_dir,
                                     cloud_nas_utils.JOB_STATUS_SUCCESS)
  except Exception as e:  # pylint: disable=broad-except
    if "NaN" in str(e):
      cloud_nas_utils.write_job_status(
          params.model_dir, cloud_nas_utils.JOB_STATUS_FAILED_WITH_NAN_ERROR)
      if FLAGS.skip_nan_error:
        # In case of `skip_nan_error`, do not raise NaN to fail the trial, so it
        # will not be counted toward `maxFailedNasTrials` setting in the API.
        tf.logging.warning(
            ("Trial failed due to NaN error, however the NaN error does not",
             " count for `maxFailedNasTrials`."))
      else:
        six.reraise(*sys.exc_info())
    else:
      cloud_nas_utils.write_job_status(params.model_dir,
                                       cloud_nas_utils.JOB_STATUS_FAILED)
      six.reraise(*sys.exc_info())


if __name__ == "__main__":
  tf_utils.suppress_tf_logging()
  cloud_nas_utils.setup_logging()
  flags = create_arg_parser().parse_args()
  main(flags)
