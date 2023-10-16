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
# ==============================================================================
"""Utility functions for proxy-task search."""

import dataclasses
import logging
import os
import time
from typing import Any, Dict, List, Optional

from gcs_utils import gcs_utils_using_cloud_storage as gcs_utils
from proxy_task import proxy_task_controller_utils_constants
import numpy as np

# Filename for trial-training-metrics data.
_TRIAL_TRAINING_METRICS_FILENAME = "TRIAL_TRAINING_METRICS.json"

# Filename for desired-training-pct signal file.
_DESIRED_TRAINING_PCT_FILENAME = "DESIRED_TRAINING_PCT.json"
_DESIRED_TRAINING_PCT_KEY = "desired_training_pct"
_MAX_TRAINING_TIME_FILENAME = "MAX_TRAINING_TIME.json"
_MAX_TRAINING_TIME_KEY = "max_training_time_in_hrs"

_ONE_HOUR_TO_SECS = 3600


@dataclasses.dataclass
class ProxyTaskConfig:
  """Class to define proxy-task configuration.

  Attributes:
    name: A unique proxy-task name set by the user.
    docker_args_map: A dictionary of {'key', val} for the arguments to be passed
      to the proxy-task training docker.
  """
  name: str
  docker_args_map: Dict[str, Any]


@dataclasses.dataclass
class AccuracyAtStep:
  """Class for training accuracy alongwith training step percentage.

  Attributes:
    training_step_pct: The percentage of training steps done, [0.0, 100.0].
    training_step: The absolute training step starting from 0.
    accuracy: The accuracy at this step.
  """
  training_step_pct: float = -1.0
  training_step: int = -1
  accuracy: float = -1.0


@dataclasses.dataclass
class TrainingTime:
  """Class to track training time.

  Attributes:
    total_training_steps: The total training steps.
    total_training_time_in_secs: The total training time in seconds.
  """
  total_training_steps: int = 0
  total_training_time_in_secs: float = 0.0


@dataclasses.dataclass
class TrialTrainingMetrics:
  """Class for training metrics for a trial.

  Attributes:
    latency: The trial latency.
    training_accuracy_over_steps: A sequence of `AccuracyAtStep`, training
      accuracies over steps.
    training_time: `TrainingTime` object.
  """
  training_time: TrainingTime = dataclasses.field(default_factory=TrainingTime)
  latency: float = -1.0
  training_accuracy_over_steps: List[AccuracyAtStep] = dataclasses.field(
      default_factory=list)


def compute_training_step_pct(end_training_cycle_step,
                              total_training_steps):
  """Computes training step percentage."""
  # end_training_cycle_step is the exact step a training cycle ends at, it
  # starts from step-0, so plus 1 to get number of steps.
  training_step_pct = 100.0 * float(end_training_cycle_step +
                                    1) / total_training_steps
  return training_step_pct


def get_accuracy_at_step_from_dict(dict_obj):
  """Loads AccuracyAtStep from a dictionary."""
  return AccuracyAtStep(**dict_obj)


def get_training_time_from_dict(dict_obj):
  """Loads TrainingTime from a dictionary."""
  return TrainingTime(**dict_obj)


def get_trial_training_metrics_from_dict(
    dict_obj):
  """Loads TrialTrainingMetrics from a dictionary."""
  return TrialTrainingMetrics(
      latency=dict_obj["latency"],
      training_accuracy_over_steps=[
          get_accuracy_at_step_from_dict(val)
          for val in dict_obj["training_accuracy_over_steps"]
      ],
      training_time=get_training_time_from_dict(dict_obj["training_time"]))


def get_trial_training_metrics_file(model_dir):
  return os.path.join(model_dir, _TRIAL_TRAINING_METRICS_FILENAME)


def read_trial_training_metrics_file(
    filepath):
  """Reads TrialTrainingMetrics data.

  Args:
    filepath: The file path of TrialTrainingMetrics data.

  Returns:
    TrialTrainingMetrics data if file exists else None.
  """
  json_data = gcs_utils.load_json(filepath)
  if json_data:
    return get_trial_training_metrics_from_dict(json_data)
  else:
    return None


def save_trial_training_metrics_file(
    filepath, trial_training_metrics):
  """Saves TrialTrainingMetrics data.

  Args:
    filepath: The file path of TrialTrainingMetrics data.
    trial_training_metrics: TrialTrainingMetrics data.
  """
  json_data = dataclasses.asdict(trial_training_metrics)
  gcs_utils.save_json(json_data=json_data, filepath=filepath)


def update_trial_training_accuracy_metric(model_dir, accuracy,
                                          begin_training_cycle_step,
                                          end_training_cycle_step,
                                          training_cycle_time_in_secs,
                                          total_training_steps):
  """Updates the trial training accuracy metric to the trial's output GCS location.

  Args:
    model_dir: GCS location for the trial output folder.
    accuracy: Training accurcay at current step.
    begin_training_cycle_step: Begin training-step for this cycle. The
      training-step value is assumed to be 0-based indexing.
    end_training_cycle_step: End training-step for this cycle. The training
      cycle should have included the end-step too. The training-step value is
      assumed to be 0-based indexing.
    training_cycle_time_in_secs: Training only time (excluding validation time)
      in seconds for this cycle.
    total_training_steps: Total number of training steps for entire training.
  """
  # Load or create a new `TrialTrainingMetrics` file for this trial.
  trial_training_metrics_file = get_trial_training_metrics_file(model_dir)
  trial_training_metrics = read_trial_training_metrics_file(
      trial_training_metrics_file)
  if not trial_training_metrics:
    trial_training_metrics = TrialTrainingMetrics()

  # Update the metrics with latest data.
  training_step_pct = compute_training_step_pct(
      end_training_cycle_step=end_training_cycle_step,
      total_training_steps=total_training_steps)
  # Check if we already have data.
  if has_accuracy_at_step(
      trial_training_metrics=trial_training_metrics,
      desired_training_step_pct=training_step_pct):
    logging.warning(
        "Trial training metrics already has data for "
        "training-step-pct %f. Skipping update.", training_step_pct)
    return
  trial_training_metrics.training_accuracy_over_steps.append(
      AccuracyAtStep(
          training_step_pct=training_step_pct,
          training_step=end_training_cycle_step,
          accuracy=accuracy))
  trial_training_metrics.training_time.total_training_steps += (
      end_training_cycle_step - begin_training_cycle_step + 1
  )
  trial_training_metrics.training_time.total_training_time_in_secs += (
      training_cycle_time_in_secs
  )

  # Save it back to the GCS location.
  save_trial_training_metrics_file(
      filepath=trial_training_metrics_file,
      trial_training_metrics=trial_training_metrics)


def update_trial_training_latency_metric(model_dir, latency):
  """Updates the trial training latency metric to the trial's output GCS location.

  Args:
    model_dir: GCS location for the trial output folder.
    latency: The model latency.
  """
  # Load or create a new `TrialTrainingMetrics` file for this trial.
  trial_training_metrics_file = get_trial_training_metrics_file(model_dir)
  trial_training_metrics = read_trial_training_metrics_file(
      trial_training_metrics_file)
  if not trial_training_metrics:
    trial_training_metrics = TrialTrainingMetrics()

  # Update the metrics with latest data.
  trial_training_metrics.latency = latency

  # Save it back to the GCS location.
  save_trial_training_metrics_file(
      filepath=trial_training_metrics_file,
      trial_training_metrics=trial_training_metrics)


def get_stop_training(model_dir, end_training_cycle_step,
                      total_training_steps):
  """Returns True if training should be stopped.

  Args:
    model_dir: GCS location for the trial output folder.
    end_training_cycle_step: End training-step for this cycle. The training
      cycle should have included the end-step too. The training-step value is
      assumed to be 0-based indexing.
    total_training_steps: Total number of training steps for entire training.

  Raises:
    Exception: if the proxy task controller has died.
  """
  # First check if a controller launched this job and the controller
  # is still alive. If a controller exists and is not alive any more
  # then stop the training job.
  timestamp_file = os.path.join(
      model_dir,
      proxy_task_controller_utils_constants.CONTROLLER_ALIVE_TIMESTAMP_FILE)
  timestamp_dict = gcs_utils.load_json(timestamp_file)
  if timestamp_dict:
    controller_timestamp = timestamp_dict[
        proxy_task_controller_utils_constants.CONTROLLER_ALIVE_TIMESTAMP_KEY]
    current_timestamp = time.time()
    if ((current_timestamp - controller_timestamp) >
        proxy_task_controller_utils_constants.CONTROLLER_ALIVE_TIMESTAMP_THRESHOLD):
      raise Exception(
          "The proxy task controller seems to have died. current_timestamp is"
          " {} and proxy task controller_timestamp is {}.".format(
              current_timestamp, controller_timestamp
          )
      )

  # Now check if the desired training percentage has been achieved.
  # If yes, then stop the training job.
  desired_training_pct_file = os.path.join(
      model_dir, _DESIRED_TRAINING_PCT_FILENAME
  )
  logging.info(
      "Looking for desired_training_pct_file: %s.", desired_training_pct_file
  )
  dict_obj = gcs_utils.load_json(desired_training_pct_file)
  if dict_obj:
    logging.info(
        "Found desired_training_pct_file: %s.", desired_training_pct_file
    )
    desired_training_step_pct = dict_obj[_DESIRED_TRAINING_PCT_KEY]
    training_step_pct = compute_training_step_pct(
        end_training_cycle_step=end_training_cycle_step,
        total_training_steps=total_training_steps,
    )
    if training_step_pct >= desired_training_step_pct:
      logging.info(
          "training_step_pct %f is >= desired_training_step_pct %f.",
          training_step_pct,
          desired_training_step_pct,
      )
      return True
    else:
      logging.info(
          "training_step_pct %f is still < desired_training_step_pct %f.",
          training_step_pct,
          desired_training_step_pct,
      )

  # Now check if the max training time has been achieved.
  # If yes, then stop the training job.
  max_training_time_file = os.path.join(model_dir, _MAX_TRAINING_TIME_FILENAME)
  logging.info(
      "Looking for max_training_time_file: %s.", max_training_time_file
  )
  dict_obj = gcs_utils.load_json(max_training_time_file)
  if dict_obj:
    logging.info("Found max_training_time_file: %s.", max_training_time_file)
    max_training_time_in_hrs = dict_obj[_MAX_TRAINING_TIME_KEY]
    trial_training_metrics_file = get_trial_training_metrics_file(model_dir)
    trial_training_metrics = read_trial_training_metrics_file(
        trial_training_metrics_file
    )
    if trial_training_metrics:
      logging.info("Loaded trial_training_metrics for training time.")
      training_time_in_hrs = (
          trial_training_metrics.training_time.total_training_time_in_secs
          / float(_ONE_HOUR_TO_SECS)
      )
      if training_time_in_hrs > max_training_time_in_hrs:
        logging.info(
            "training_time_in_hrs %f is > max_training_time_in_hrs %f.",
            training_time_in_hrs,
            max_training_time_in_hrs,
        )
        return True
      else:
        logging.info(
            "training_time_in_hrs %f is still <= max_training_time_in_hrs %f.",
            training_time_in_hrs,
            max_training_time_in_hrs,
        )
  return False


def set_stop_training(
    model_dir,
    desired_training_step_pct = None,
    max_training_time_in_hrs = None,
):
  """Sets a file to signal that the training should be stopped."""
  # Process desired training step percentage.
  if desired_training_step_pct:
    desired_training_pct_file = os.path.join(
        model_dir, _DESIRED_TRAINING_PCT_FILENAME
    )
    if gcs_utils.exists(desired_training_pct_file):
      logging.info(
          "desired_training_pct_file already exists: %s.",
          desired_training_pct_file,
      )
      return
    json_data = {_DESIRED_TRAINING_PCT_KEY: desired_training_step_pct}
    gcs_utils.save_json(json_data=json_data, filepath=desired_training_pct_file)

  # Process max training time.
  if max_training_time_in_hrs:
    max_training_time_file = os.path.join(
        model_dir, _MAX_TRAINING_TIME_FILENAME
    )
    if gcs_utils.exists(max_training_time_file):
      logging.info(
          "max_training_time_file already exists: %s.", max_training_time_file
      )
      return
    json_data = {_MAX_TRAINING_TIME_KEY: max_training_time_in_hrs}
    gcs_utils.save_json(json_data=json_data, filepath=max_training_time_file)


def has_accuracy_at_step(
    trial_training_metrics,
    desired_training_step_pct,
):
  """Returns True if training-metrics have the desired step."""
  if (
      not trial_training_metrics
      or not trial_training_metrics.training_accuracy_over_steps
  ):
    return False
  latest_training_step_pct = (
      trial_training_metrics.training_accuracy_over_steps[-1].training_step_pct
  )
  return latest_training_step_pct >= desired_training_step_pct


def get_accuracy_at_step(trial_training_metrics,
                         desired_training_step_pct):
  """Returns the accuracy at the desired training step percent."""
  if not has_accuracy_at_step(
      trial_training_metrics=trial_training_metrics,
      desired_training_step_pct=desired_training_step_pct):
    return None
  # Find the location of the 'desired_training_step_pct' using interpolation.
  # Initialize step 0.
  step_pct_list = [0.0]
  accuracy_list = [0.0]
  for accuracy_at_step in trial_training_metrics.training_accuracy_over_steps:
    step_pct_list.append(accuracy_at_step.training_step_pct)
    accuracy_list.append(accuracy_at_step.accuracy)
  return np.interp(desired_training_step_pct, step_pct_list, accuracy_list)


def copy_trial_training_metrics(prev_model_dir, curr_model_dir):
  """Copies trial training metrics to a new location."""
  prev_filepath = get_trial_training_metrics_file(prev_model_dir)
  curr_filepath = get_trial_training_metrics_file(curr_model_dir)
  if gcs_utils.exists(prev_filepath):
    gcs_utils.copy(src_filepath=prev_filepath, dst_filepath=curr_filepath)
  else:
    raise ValueError("Could not find file: %s" % prev_filepath)
