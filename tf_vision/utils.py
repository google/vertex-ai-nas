# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Utilities related to Cloud NAS."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import json
from typing import Any, Dict, Optional

from absl import logging
from tf_vision import configs
import pyglove as pg
import tensorflow as tf

from official.core import config_definitions
from official.core import train_utils
from tf_vision.pointpillars.configs import pointpillars as pointpillars_cfg
from official.vision.beta import configs as base_configs

# Keys used in model analysis file.
MODEL_LATENCY_MILLI = "latency_milli_seconds"
MODEL_FLOPS = "multi_add_flops_billion"
MODEL_MEMORY = "model_memory"
MODEL_MAP = "AP"
MODEL_MAPH = "APH"
MODEL_TOP1_ACCURACY = "accuracy"
MODEL_MIOU = "miou"


class LatencyConstraintType(enum.Enum):
  NONE = 1
  DEVICE_MILLI_SEC = 2
  FLOPS_HALF_ADDS_BILLION = 3


@pg.members([
    ("step", pg.typing.Int(), "At which step the result is reported."),
    ("metrics",
     pg.typing.Dict([(pg.typing.StrKey(), pg.typing.Float(), "Metric item.")
                    ]).noneable(), "Metric in key/value pairs (optional)."),
])
class Measurement(pg.Object):
  """Measurement of a trial at certain step."""


def read_tunable_block_specs(block_specs_json):
  """Reads and returns tunable block_specs from a JSON file."""
  if block_specs_json.endswith(".json"):
    with tf.io.gfile.GFile(block_specs_json) as f:
      block_specs_json = f.read()
  return pg.from_json_str(block_specs_json)


def get_accuracy_key(task):
  """Given the task type, returns the accuracy tag in evaluation metrics."""
  if isinstance(task, configs.semantic_segmentation.SemanticSegmentationTask):
    return MODEL_MIOU
  elif isinstance(task,
                  base_configs.image_classification.ImageClassificationTask):
    return MODEL_TOP1_ACCURACY
  elif isinstance(task, configs.retinanet.RetinaNetTask):
    return MODEL_MAP
  elif isinstance(task, pointpillars_cfg.PointPillarsTask):
    return MODEL_MAP
  else:
    raise ValueError("Task is not supported.")


def set_up_constraints(target_device_latency_ms,
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
  logging.info("Constraint is (%s, %s).", constraint_type, target_constraint)
  return constraint_type, target_constraint


def get_model_flops_and_params(
    model,
    params,
    json_file_path = None):
  """Profiles the given model and returns FLOPS and number of parameters.

  Args:
    model: A model instance.
    params: The model config.
    json_file_path: An optional string specifying the JSON file path to save
      model statistics.

  Returns:
    A dictionary of model statistics, including FLOPS and number of parameters.
  """

  # Because we have mix use of functional API and subclassing of tf.keras.Model,
  # we need to create inputs_kwargs argument to specify the input shapes for
  # subclass model that overrides model.call to take multiple inputs,
  # e.g., RetinaNet model.
  inputs_kwargs = None
  if isinstance(params.task, base_configs.retinanet.RetinaNetTask):
    inputs_kwargs = {
        "images": tf.TensorSpec([1] + params.task.model.input_size, tf.float32),
        "image_shape": tf.TensorSpec([1, 2], tf.float32)
    }
    dummy_inputs = {
        k: tf.ones(v.shape.as_list(), tf.float32)
        for k, v in inputs_kwargs.items()
    }
    # Must do forward pass to build the model.
    model(**dummy_inputs)
  elif isinstance(params.task, pointpillars_cfg.PointPillarsTask):
    pillars_config = params.task.model.pillars
    batch_size = 1
    inputs_kwargs = {
        "pillars":
            tf.TensorSpec([
                batch_size,
                pillars_config.num_pillars,
                pillars_config.num_points_per_pillar,
                pillars_config.num_features_per_point
            ], tf.float32),
        "indices":
            tf.TensorSpec([
                batch_size,
                pillars_config.num_pillars,
                2
            ], tf.int32),
        "image_shape": tf.TensorSpec([batch_size, 2], tf.int32)
    }
    dummy_inputs = {
        k: tf.ones(v.shape.as_list(), v.dtype)
        for k, v in inputs_kwargs.items()
    }
    model(**dummy_inputs)

  # Measure FLOPs.
  flops = train_utils.try_count_flops(model, inputs_kwargs)
  multi_add_flops_billion = 0
  if flops:
    multi_add_flops_billion = flops * 1. / 10**9 / 2
    logging.info("number of FLOPS (multi-adds): %f B.", multi_add_flops_billion)
  else:
    logging.warn("Failed to compute FLOPS (multi-adds). Set it to 0.")

  # Measure parameter count.
  num_params = train_utils.try_count_params(model)
  num_params_million = 0
  if num_params:
    num_params_million = num_params * 1. / 10**6
    logging.info("number of params: %f M.", num_params_million)
  else:
    logging.warn("Failed to compute number of parameters. Set it to 0.")

  model_stats = {
      "multi_add_flops_billion": float(multi_add_flops_billion),
      "num_params_million": float(num_params_million)
  }
  if json_file_path:
    with tf.io.gfile.GFile(json_file_path, "w") as fp:
      json.dump(model_stats, fp)

  return model_stats
