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
"""Cloud NAS util libraries."""
import contextlib
import json
import logging
import os
import re
import sys
import time
import traceback

from typing import Mapping
import metrics_reporter
from gcs_utils import gcs_utils_using_cloud_storage as gcs_utils

import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
import pyglove as pg

JOB_STATUS_KEY = "job_status"
JOB_STATUS_SUCCESS = "success"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_FAILED_WITH_NAN_ERROR = "failed_with_nan_error"

# The allowed-range for the metric reported back to the NAS controller.
MIN_ALLOWED_METRIC = 0.001
MAX_ALLOWED_METRIC = 10

# The number of times to wait for TPU-cluster.
_TPU_RESOLVER_RETRY = 40

# Environment variable for Vertex NAS job directory.
ENVIRONMENT_VARIABLE_FOR_MODEL_DIR = "AIP_MODEL_DIR"

# Environment variable for trial id.
ENVIRONMENT_VARIABLE_FOR_TRIAL_ID = "CLOUD_ML_TRIAL_ID"

# Vertex NAS search trial job directory regex.
# This regular expression is used to infer the
# Vertex NAS search job directory from the environment
# variable for the model-directory.
VERTEX_SEARCH_TRIAL_DIR_REGEX = r".*nas/search/\d+"
# Vertex NAS train trial job directory regex.
# This regular expression is used to infer the
# Vertex NAS train job directory from the environment
# variable for the model directory.
VERTEX_TRAIN_TRIAL_DIR_REGEX = r".*nas/train/\d+"

# Saved-model sub-directory.
SAVED_MODEL_SUB_DIR = "saved_model"


class CloudSession(object):
  """An implementation of tuner.Session for cloud run."""

  def __init__(self, model_dir):
    self._model_dir = model_dir

  @contextlib.contextmanager
  def fail_on_exceptions(self, exceptions):
    """Yield failing on exceptions."""
    try:
      yield
    except exceptions:
      self.fail(traceback.format_exc())

  def report(self, measurement, model_flops=0.0, other_metrics={}):  # pylint: disable=dangerous-default-value
    """Report results to cloud tuner.

    Args:
      measurement: a pair of metrics for a step.
      model_flops: Number of FLOPS for the model.
      other_metrics: Dictionary of key value pair of metrics. Only a maximum of
        5 custom metrics are allowed to report. The key should be a string with
        less than 65 characters and value should be a number.
    """
    # `hp_metric_tag` is the value specified in `nasTargetRewardMetric`.
    # We choose it out of `measurement.metrics` to report.
    hp_metric_tag = str(os.environ.get("CLOUD_ML_HP_METRIC_TAG", ""))
    if hp_metric_tag:
      if hp_metric_tag not in measurement.metrics:
        raise ValueError(
            "hp_metric_tag [{}] is missing in the metrics {}".format(
                hp_metric_tag, measurement.metrics))
      nas_metrics_reporter = metrics_reporter.NasMetricsReporter()

      metric_value = float(measurement.metrics[hp_metric_tag])
      nas_metrics_reporter.report_metrics(
          hyperparameter_metric_tag=hp_metric_tag,
          metric_value=metric_value,
          global_step=measurement.step,
          model_flops=model_flops,
          other_metrics=other_metrics)
      logging.info(
          ("Reported metrics at step %s for metric %s: metric_value [%s], "
           "model_flops [%s], other_metrics [%s]"), measurement.step,
          hp_metric_tag, metric_value, model_flops, other_metrics)


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin


def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.01):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
        25)
  except IOError:
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      if class_names[i] == b"mic":
        print("{} is at: {}_{}_{}_{}".format(class_names[i], ymin, xmin, ymax,
                                             xmax))
      display_str = "{}: {}%".format(class_names[i], int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image


def str_2_bool(v):
  """Auxiliary function to support boolean command-line arguments."""
  if not isinstance(v, str):
    raise ValueError("{} is not string type".format(v))
  return v.lower() == "true"


def get_trial_id_from_environment():
  if ENVIRONMENT_VARIABLE_FOR_TRIAL_ID not in os.environ:
    logging.warning("Environment variable %s not found, "
                    "return 0 as default trial id.",
                    ENVIRONMENT_VARIABLE_FOR_TRIAL_ID)
  return os.environ.get(ENVIRONMENT_VARIABLE_FOR_TRIAL_ID, 0)


def parse_and_save_nas_params_str(search_space, nas_params_str, model_dir=None):
  """Returns the tunable object for `search_space` based on `nas_params_str`.

  Args:
    search_space: a pg.Object or pg.functor for the search space definition.
    nas_params_str:  a string of the model suggestion from the NAS service.
    model_dir: a string. The model dir of the training job. If provided, the
      value of `nas_params_str` is saved as "nas_params_str.json" in
      `model_dir`.  It also saves `nas_params_str` as `nas_params_str.json` in
      the model_dir, which can be used to rebuild the model for stage-2
      finetuning or create model for deployment.
  Returns: The tunable object (pg.functor or pg.Object) that can be used to
    create model.
  """

  if nas_params_str.endswith(".json"):
    # In stage-2 finetuning, read `nas_params_str` from a json file created in
    # stage-1 search.
    logging.info("Reading %s for nas_params_str value.", nas_params_str)
    with gcs_utils.file_open(nas_params_str, "r") as f:
      nas_params_str = f.read()

  logging.info("nas_params_str is %s", nas_params_str)
  nas_params_dict = json.loads(nas_params_str)

  # The materialized object can be a `pg.functor` or `pg.Object`.
  # For example, search_spaces.SpineNetScalingSpecBuilder is a pg.Object.
  # tunable_spinenet.build_tunable_block_specs() is a pg.functor.
  # If it is a pg.functor, invoking `functor_or_object()` can return
  # pg.Objects. `pg.functor` is normally used to convert `its args` to
  # `pg.Objects` with some transformations (see
  # `tunable_spinenet.build_tunable_block_specs()` as an example for details).
  pg_functor_or_object = pg.materialize(search_space, nas_params_dict)

  if model_dir:
    # Outputs `nas_params_str` so that stage-2 job can reuse `the JSON file` as
    # the flag `--nas_params_str`.
    nas_params_file = os.path.join(model_dir, "nas_params_str.json")
    gcs_utils.save_json(filepath=nas_params_file, json_data=nas_params_dict)
  return pg_functor_or_object


def serialize_and_save_tunable_object(tunable_object, model_dir):
  serialized_tunable_object = pg.to_json_str(
      tunable_object, json_indent=2, hide_default_values=False
  )
  logging.info("serialized_tunable_object is %s", serialized_tunable_object)
  serialized_tunable_object_file = os.path.join(
      model_dir, "serialized_tunable_object.json"
  )
  with gcs_utils.file_open(serialized_tunable_object_file, "w") as f:
    f.write(serialized_tunable_object)
  return serialized_tunable_object


def write_job_status(model_dir, status):
  """Writes job status file in root dir."""
  job_status_file = os.path.join(model_dir, "job_status.json")
  gcs_utils.save_json(
      filepath=job_status_file, json_data={JOB_STATUS_KEY: status})


def get_map_finetune_trial_id_to_search_trial_id(
    retrain_search_job_trials):
  """Returns a mapping from finetune trial-id to previous search trial-id."""
  # Given a list of search-trial ids to finetune such as '123, 453, 1025', the
  # nas-cli will launch 3 finetune trials corresponding to them. This function
  # will return a map (finetune-trial-id -> search-trial-id):
  # "1"->"123"
  # "2"->"453"
  # "3"->"1025".
  search_trial_id_list = [
      search_trial_id.strip()
      for search_trial_id in retrain_search_job_trials.split(",")
  ]
  map_finetune_trial_idx_to_search_trial_id = {}
  for idx, search_trial_id in enumerate(search_trial_id_list):
    finetune_trial_id = str(idx + 1)
    map_finetune_trial_idx_to_search_trial_id[
        finetune_trial_id] = search_trial_id
  logging.info("map_finetune_trial_idx_to_search_trial_id is %s",
               map_finetune_trial_idx_to_search_trial_id)
  return map_finetune_trial_idx_to_search_trial_id


def get_search_trial_id_to_finetune(retrain_search_job_trials):
  """Returns the search-trial-id corresponding to the current finetuning job calling this function.

  Given a list of search-trial ids to finetune such as '123, 453, 1025', the
  nas-cli will launch 3 finetune trials corresponding to them. This function
  will first map (finetune-trial-idx -> search-trial-id):
  1->123
  2->453
  3->1025
  Then based on the current finetune-trial-idx, it will return the cuurent
  search-trial-id.

  Args:
    retrain_search_job_trials: A comma separated string of search-trial-ids
      to finetune. For example, '123, 453, 1025'.

  Returns:
    The search-trial-id corresponding to the current stage2 CALLER job. For
    example, if the current stage2 was launched for the trial 453, then 453 will
    be returned.
  """
  map_finetune_trial_idx_to_search_trial_id = (
      get_map_finetune_trial_id_to_search_trial_id(retrain_search_job_trials)
  )
  current_finetune_trial_id = get_trial_id_from_environment()
  current_search_trial_id = map_finetune_trial_idx_to_search_trial_id[
      current_finetune_trial_id]
  return current_search_trial_id


def get_retrain_search_job_model_dir(retrain_search_job_trials,
                                     retrain_search_job_dir):
  """Returns the previous search-job model dir corresponding to the current trial.

  Args:
    retrain_search_job_trials: A comma separated string of search-trial-ids
      to finetune. For example, '123, 453, 1025'.
    retrain_search_job_dir: The previous search job directory.

  Returns:
    The previous search job previous search-job model dir
    corresponding to the current trial.
  """
  current_search_trial_id = get_search_trial_id_to_finetune(
      retrain_search_job_trials)
  return os.path.join(retrain_search_job_dir, current_search_trial_id)


def get_finetune_nas_params_str(retrain_search_job_trials,
                                retrain_search_job_dir):
  """Returns the nas_params_str for the CURRENT finetuning job calling this function."""
  if not retrain_search_job_trials:
    raise ValueError("retrain_search_job_trials is not set.")
  if not retrain_search_job_dir:
    raise ValueError("retrain_search_job_dir is not set.")

  current_search_trial_id = get_search_trial_id_to_finetune(
      retrain_search_job_trials)
  nas_params_str = os.path.join(retrain_search_job_dir, current_search_trial_id,
                                "nas_params_str.json")
  return nas_params_str


def setup_logging():
  """Sets up logging for the main file.

  Sets INFO level for logging, and ensures that ERROR logs display
  different than the INFO logs on cloud.
  """
  root_logger = logging.getLogger()
  root_logger_previous_handlers = list(root_logger.handlers)
  for h in root_logger_previous_handlers:
    root_logger.removeHandler(h)
  root_logger.setLevel(logging.INFO)
  root_logger.propagate = False

  # Set tf logging to avoid duplicate logging. If the handlers are not removed,
  # then we will have duplicate logging :
  # From tf loggging written to stderr stream, and
  # From python logger written to stdout stream.
  tf_logger = logging.getLogger("tensorflow")
  while tf_logger.handlers:
    tf_logger.removeHandler(tf_logger.handlers[0])

  # Redirect INFO logs to stdout
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging.INFO)
  root_logger.addHandler(stdout_handler)


def compute_reward(accuracy,
                   target_latency_or_flops,
                   measured_latency_or_flops,
                   target_memory=0,
                   measured_memory=0,
                   weight=1,
                   use_hard_limit=True):
  """Compute reward from accuracy and other constraint-value.

  It supports Hard or Soft exponential reward scheme described in the MNAS paper
  https://arxiv.org/pdf/1807.11626.pdf

  Args:
    accuracy: model accuracy, e.g., Top-1 accuracy or mAP.
    target_latency_or_flops: the target latency or FLOPS. For example, 100
      (milliseconds) for latency or 50 (billion) FLOPS.
    measured_latency_or_flops: the measured model latency or FLOPS value.
    target_memory: the target model memory, e.g., 5000 (megabytes).
    measured_memory: the measured model memory.
    weight: the weight to compute the reward. Larger value would penalize more
      for the models with larger latency/memory.
    use_hard_limit: bool - True to use the hard expontential reward else use
      soft exponential reward above.

  Returns:
    The computed reward.
  """
  latency_or_flops_ratio = float(target_latency_or_flops) / float(
      measured_latency_or_flops)

  # Use memory_ratio if `target_memory` is provided.
  if target_memory == 0:
    memory_ratio = 1
  else:
    if not measured_memory:
      raise ValueError("measured_memory should not be 0.")
    memory_ratio = float(target_memory) / float(measured_memory)
  logging.info("latency_or_flops_ratio: %s, memory_ratio: %s",
               latency_or_flops_ratio, memory_ratio)

  latency_weight = latency_or_flops_ratio**weight
  memory_weight = memory_ratio**weight

  if use_hard_limit:
    latency_weight = min(latency_weight, 1.0)
    memory_weight = min(memory_weight, 1.0)

  return accuracy * latency_weight * memory_weight


def wait_and_retrieve_latency(saved_model_path, sleep_time=30):
  """Waits and retrieves model-latency generated by another process.

  Args:
    saved_model_path: GCS path to the saved-model.
    sleep_time: Time to sleep in seconds before checking for the latency file.

  Raises:
    Exception: When does not find latency-file even after a time-out
    period.

  Returns:
    A dictionary of model-latency stats loaded from the latency file.
  """
  # Wait in a while loop for another
  # process to compute and save the model_latency.json file. Once the file gets
  # generated, we will load and return the latency values.
  start_time = time.time()
  latency_calculation_timeout_seconds = 60 * 10
  while True:
    if time.time() - start_time > latency_calculation_timeout_seconds:
      raise Exception(
          "Missing latency calculation results after {} seconds.".format(
              latency_calculation_timeout_seconds))
    time.sleep(sleep_time)
    model_latency_file = os.path.join(saved_model_path, "model_latency.json")
    if gcs_utils.exists(model_latency_file):
      model_stats = gcs_utils.load_json(model_latency_file)
      return model_stats


def wait_for_tpu_cluster_resolver_ready(tpu_cluster_resolver_module):
  """Waits for `TPUClusterResolver` to be ready and then returns it.

  Args:
    tpu_cluster_resolver_module: A TF1 or TF2 `TPUClusterResolver` module.

  Returns:
    A TPUClusterResolver if there is TPU machine (in TPU_CONFIG). Otherwise,
    return None.
  """
  tpu_config_env = os.environ.get("TPU_CONFIG")
  if not tpu_config_env:
    logging.info("Missing TPU_CONFIG, use CPU/GPU for training.")
    return None

  tpu_node = json.loads(tpu_config_env)
  logging.info("Waiting for TPU to be ready: \n%s.", tpu_node)

  num_retries = _TPU_RESOLVER_RETRY
  for i in range(num_retries):
    try:
      tpu_cluster_resolver = (
          tpu_cluster_resolver_module(
              tpu=[tpu_node["tpu_node_name"]],
              zone=tpu_node["zone"],
              project=tpu_node["project"],
              job_name="worker"))
      tpu_cluster_resolver_dict = tpu_cluster_resolver.cluster_spec().as_dict()
      if "worker" in tpu_cluster_resolver_dict:
        logging.info("Found TPU worker: %s", tpu_cluster_resolver_dict)
        return tpu_cluster_resolver
    except Exception as e:  # pylint: disable=broad-except
      if i < num_retries - 1:
        logging.info("Still waiting for provisioning of TPU VM instance.")
      else:
        # Preserves the traceback.
        logging.info(
            "Failed for wait_for_tpu_cluster_resolver_ready after retry %s",
            num_retries)
        raise e
    time.sleep(10)

  # Raise error when failed to get TPUClusterResolver after retry.
  raise ValueError("Cannot get TPUClusterResolver: {}".format(tpu_node))


def get_job_dir_from_environment_if_exist(current_job_dir):
  """Checks environment for trial job-dir."""
  vertex_model_dir = os.environ.get(ENVIRONMENT_VARIABLE_FOR_MODEL_DIR)
  if vertex_model_dir:
    # For vertex, the job-dir is of the form:
    # gs://<bucket-name>/<job-name-with-timestamp>/nas/search/<trial-id>
    # or
    # gs://<bucket-name>/<job-name-with-timestamp>/nas/train/<trial-id>
    #
    # However, the environment-variable path also contains a sub-folder
    # inside the job-dir. So we need to extract the job-dir
    # from the environment-variable.
    search_trial_match = re.match(VERTEX_SEARCH_TRIAL_DIR_REGEX,
                                  vertex_model_dir)
    train_trial_match = re.match(VERTEX_TRAIN_TRIAL_DIR_REGEX, vertex_model_dir)
    if search_trial_match:
      vertex_job_dir = search_trial_match.group(0)
    elif train_trial_match:
      vertex_job_dir = train_trial_match.group(0)
    else:
      raise ValueError("Could not extract vertex_job_dir from %s" %
                       vertex_model_dir)
    logging.info("Overriding job-dir %s to %s set by the environment.",
                 current_job_dir, vertex_job_dir)
    return vertex_job_dir
  else:
    # It is AI Platform job-dir so no change is needed.
    logging.info("Keeping current job-dir %s.", current_job_dir)
    return current_job_dir


def get_saved_model_dir(trial_job_dir):
  return os.path.join(trial_job_dir, SAVED_MODEL_SUB_DIR)
