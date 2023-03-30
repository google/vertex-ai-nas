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

"""Utilities related to compute TF Vision models' latency."""
import copy
from typing import Tuple

from absl import logging
import cloud_nas_utils
from tf_vision.configs import retinanet as retinanet_cfg
from tf_vision.configs import semantic_segmentation as segmentation_cfg
from tf_vision.modeling import tunable_pointpillars

from official.core import config_definitions
from official.modeling import performance
from tf_vision.pointpillars.configs import pointpillars as pointpillars_cfg
from official.vision.beta.configs import image_classification as classification_cfg
from official.vision.beta.serving import export_saved_model_lib


# Keys used in model export.
_INPUT_TYPE_IMAGE_BYTES = "image_bytes"


def get_model_latency(params,
                      model_dir,
                      latest_checkpoint):
  """Returns the latency of generated SavedModel.

  Args:
    params: An ExperimentConfig that specifies the model config.
    model_dir: The path to the saved model.
    latest_checkpoint: The path to the latest saved model checkpoint.

  Returns:
    A tuple of (model latency in milliseconds, model memory).

  Raises:
    Exception: when latency results are missing after 600 seconds.
  """
  params_for_saved_model = copy.deepcopy(params)

  # Update TPU specific parameters as model export operation is done on CPU and
  # latency is measured on a single GPU with float32 dtype.
  params_for_saved_model.task.model.norm_activation.use_sync_bn = False
  params_for_saved_model.runtime.mixed_precision_dtype = "float32"
  params_for_saved_model.runtime.distribution_strategy = "one_device"
  params_for_saved_model.runtime.num_gpus = 0

  if isinstance(params_for_saved_model.task, retinanet_cfg.RetinaNetTask):
    params_for_saved_model.task.model.detection_generator.nms_version = "batched"

  saved_model_path = cloud_nas_utils.get_saved_model_dir(
      trial_job_dir=model_dir)

  logging.info("Model params for export: %s.", params_for_saved_model.as_dict())

  # 2D model export.
  if isinstance(params_for_saved_model.task,
                (classification_cfg.ImageClassificationTask,
                 segmentation_cfg.SemanticSegmentationTask,
                 retinanet_cfg.RetinaNetTask)):
    export_model_image_size = params_for_saved_model.task.model.input_size[0:2]
    logging.info("export_model_image_size: %s.", export_model_image_size)
    performance.set_mixed_precision_policy(
        params_for_saved_model.runtime.mixed_precision_dtype)
    export_saved_model_lib.export_inference_graph(
        input_type=_INPUT_TYPE_IMAGE_BYTES,
        batch_size=1,
        input_image_size=export_model_image_size,
        params=params_for_saved_model,
        checkpoint_path=latest_checkpoint,
        export_dir=saved_model_path)

  # 3D model export.
  if isinstance(params_for_saved_model.task, pointpillars_cfg.PointPillarsTask):
    tunable_pointpillars.export_inference_graph(
        params=params_for_saved_model,
        checkpoint_path=latest_checkpoint,
        export_dir=saved_model_path,
    )

  logging.info("Exported SavedModel at %s.", saved_model_path)

  # Set mixed precision policy back to original value from params in case there
  # are additional steps to be run on TPU workers after model export.
  performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype)

  # Wait for another process to compute latency.
  logging.info("Waiting for latency computation.")
  model_stats = cloud_nas_utils.wait_and_retrieve_latency(saved_model_path)
  latency_in_milliseconds = model_stats["latency_in_milliseconds"]
  model_memory = model_stats["model_memory"]
  return latency_in_milliseconds, model_memory
