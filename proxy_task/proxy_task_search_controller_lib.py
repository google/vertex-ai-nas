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
"""Proxy-task search controller library functions."""

import copy
import dataclasses
import datetime
import enum
import logging
import math
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

import vertex_client_utils as client_utils
from gcs_utils import gcs_utils_using_cloud_storage as gcs_utils
from gcs_utils import gcs_utils_using_gcsfuse
from proxy_task import proxy_task_controller_utils
from proxy_task import proxy_task_controller_utils_constants
from proxy_task import proxy_task_model_selection_lib
from proxy_task import proxy_task_search_controller_lib_constants as search_controller_constants
from proxy_task import proxy_task_utils
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class ProxyTaskStoppingState(str, enum.Enum):
  """An enum class which defines different proxy-task stopping states."""

  MET_DESIRED_ACCURACY_CORRELATION = "Met desired correlation"
  MET_DESIRED_ACCURACY = "Met desired accuracy"
  EXCEEDED_TRAIN_TIME_LIMIT = "Exceeded training time limit"
  EXCEEDED_BEST_TRAIN_TIME = "Exceeded best training time"
  INVALID_STOPPING_STATE = ""
  FAILED_DESIRED_LATENCY_CORRELATION = "Failed desired latency correlation"
  JOB_FAILED = "Proxy task job failed"
  RAN_TILL_COMPLETION = "Ran till completion"
  EXCESS_TRIAL_FAILURES = "Excess trial failures"


@dataclasses.dataclass
class ProxyTaskStats:
  """Class to store proxy-task scores such as correlation over steps.

  Attributes:
    training_steps: Training-steps over which we have scores.
    accuracy_correlation_over_step: Accuracy correlation score over
      training-steps.
    accuracy_correlation_p_value_over_step: Accuracy correlation p-value over
      training-steps.
    median_accuracy_over_step: Median accuracy over training-steps.
    median_training_time_hrs_over_step: Median training-time in hours over
      training-steps.
    latency_correlation: Latency correlation.
    latency_correlation_p_value: Latency correlation p-value.
    stopping_state: Stopping state for the proxy-task.
    posted_stop_trials_message: True if stop trials message was posted.
    final_training_time_in_hours: If successful, the final training-time in
      hours.
    final_training_steps: If successful, the final training-steps.
  """

  training_steps: List[int] = dataclasses.field(default_factory=list)
  accuracy_correlation_over_step: List[float] = dataclasses.field(
      default_factory=list
  )
  accuracy_correlation_p_value_over_step: List[float] = dataclasses.field(
      default_factory=list
  )
  median_accuracy_over_step: List[float] = dataclasses.field(
      default_factory=list
  )
  median_training_time_hrs_over_step: List[float] = dataclasses.field(
      default_factory=list
  )
  latency_correlation: float = search_controller_constants.INVALID_CORRELATION
  latency_correlation_p_value: float = (
      search_controller_constants.INVALID_P_VALUE
  )
  stopping_state: ProxyTaskStoppingState = (
      ProxyTaskStoppingState.INVALID_STOPPING_STATE
  )
  posted_stop_trials_message: bool = False
  final_training_time_in_hours: float = (
      search_controller_constants.INVALID_TRAINING_TIME
  )
  final_training_steps: int = search_controller_constants.INVALID_TRAINING_STEP


def get_proxy_task_stats_from_dict(
    dict_obj
):
  """Loads ProxyTaskStats from a dictionary."""
  return ProxyTaskStats(**dict_obj)


@dataclasses.dataclass
class ProxyTaskState:
  """Class for Proxy-task state.

  Attributes:
    proxy_task_stats: Proxy-task scores such as correlation over steps.
    proxy_task_name: Proxy-task name.
    search_job_name: Search job name.
    search_job_link: Cloud console link for the search job.
    latency_calculator_job_name: Latency-calculator job name.
    latency_calculator_job_link: Cloud console link for the latency-calculator
      job.
  """

  proxy_task_stats: ProxyTaskStats = dataclasses.field(
      default_factory=ProxyTaskStats
  )
  proxy_task_name: str = ""
  search_job_name: str = ""
  search_job_link: str = ""
  latency_calculator_job_name: str = ""
  latency_calculator_job_link: str = ""


def get_proxy_task_state_from_dict(
    dict_obj
):
  """Loads ProxyTaskState from a dictionary."""
  return ProxyTaskState(
      proxy_task_name=dict_obj["proxy_task_name"],
      search_job_name=dict_obj["search_job_name"],
      search_job_link=dict_obj["search_job_link"],
      latency_calculator_job_name=dict_obj["latency_calculator_job_name"],
      latency_calculator_job_link=dict_obj["latency_calculator_job_link"],
      proxy_task_stats=get_proxy_task_stats_from_dict(
          dict_obj["proxy_task_stats"]
      ),
  )


@dataclasses.dataclass
class SearchControllerState:
  """Class for search controller state.

  Attributes:
    proxy_tasks_map: A map of {proxy_task_name: proxy_task_state}.
    best_proxy_task_name: Name of the best proxy task, one which is successful
      and has lowest training-time.
  """

  proxy_tasks_map: Dict[str, ProxyTaskState] = dataclasses.field(
      default_factory=dict
  )
  best_proxy_task_name: str = ""


def get_search_controller_state_from_dict(
    dict_obj
):
  """Loads SearchControllerState from a dictionary."""
  proxy_tasks_map_dict_obj = dict_obj["proxy_tasks_map"]
  proxy_tasks_map = {}
  for (
      proxy_task_name,
      proxy_task_state_dict_obj,
  ) in proxy_tasks_map_dict_obj.items():
    proxy_tasks_map[proxy_task_name] = get_proxy_task_state_from_dict(
        proxy_task_state_dict_obj
    )
  return SearchControllerState(
      proxy_tasks_map=proxy_tasks_map,
      best_proxy_task_name=dict_obj["best_proxy_task_name"],
  )


def get_search_controller_state_file(search_controller_dir):
  return os.path.join(
      search_controller_dir,
      search_controller_constants.SEARCH_CONTROLLER_STATE_FILENAME,
  )


def save_search_controller_state_file(
    filepath, search_controller_state
):
  """Saves SearchControllerState data.

  Args:
    filepath: The file path of SearchControllerState data.
    search_controller_state: SearchControllerState data.
  """
  json_data = dataclasses.asdict(search_controller_state)
  gcs_utils.save_json(json_data=json_data, filepath=filepath)


def try_load_search_controller_state(
    search_controller_dir,
):
  """Returns search controller state by loading a previously stored file if it exists."""
  filepath = get_search_controller_state_file(search_controller_dir)
  search_controller_state = SearchControllerState()
  json_data = gcs_utils.load_json(filepath=filepath)
  if json_data:
    logging.warning("The search controller has restarted.")
    search_controller_state = get_search_controller_state_from_dict(json_data)
    logging.info("loaded search_controller_state: %s", search_controller_state)
  return search_controller_state


def convert_all_trial_training_data_to_2d_array_of_accuracies(
    all_trial_training_data,
):
  """Converts list of trial-training-data to 2D numpy array of accuracies."""
  # First find the super-set of all recorded training steps.
  training_steps_set = set()
  for trial_training_metrics in all_trial_training_data:
    if trial_training_metrics:
      for (
          accuracy_at_step
      ) in trial_training_metrics.training_accuracy_over_steps:
        training_steps_set.add(accuracy_at_step.training_step)
  if not training_steps_set:
    return None, None

  # Sort the training steps.
  training_steps = list(training_steps_set)
  training_steps.sort()
  map_training_step_to_step_idx = {}
  for step_idx, training_step in enumerate(training_steps):
    map_training_step_to_step_idx[training_step] = step_idx

  # Create 2D numpy array of accuracies.
  # Row-axis will be trial-idx and col-axis will be step-idx.
  num_full_training_trials = len(all_trial_training_data)
  accuracies_2d_array = (
      np.ones((num_full_training_trials, len(training_steps)))
      * search_controller_constants.INVALID_ACCURACY
  )
  for trial_idx in range(num_full_training_trials):
    trial_training_metrics = all_trial_training_data[trial_idx]
    if not trial_training_metrics:
      continue
    for accuracy_at_step in trial_training_metrics.training_accuracy_over_steps:
      step_idx = map_training_step_to_step_idx[accuracy_at_step.training_step]
      accuracies_2d_array[trial_idx, step_idx] = accuracy_at_step.accuracy
  return accuracies_2d_array, training_steps


def get_trials_with_valid_metrics(metric_list):
  """Returns a set of trials with valid metrics."""
  trials_with_valid_metrics = set()
  for trial_idx, val in enumerate(metric_list):
    if not math.isnan(val):
      trials_with_valid_metrics.add(trial_idx_to_id(trial_idx))
  return trials_with_valid_metrics


def compute_rank_correlation(
    metric_list_a,
    metric_list_b,
    min_required_num_correlation_metrics = search_controller_constants.MIN_REQUIRED_NUM_CORRELATION_METRICS,
    min_ratio_of_valid_trials_to_total_trials = search_controller_constants.MIN_RATIO_OF_VALID_TRIALS_TO_TOTAL_TRIALS,
):
  """Returns proxy-task score based on Kendall rank correlation."""
  # Check if they have same length.
  if len(metric_list_a) != len(metric_list_b):
    logging.warning(
        "Can not compute proxy-task score. Metrics do not have same length."
    )
    return math.nan, math.nan, None, None

  # Only valid values are used to calculate the score.
  valid_metric_list_a, valid_metric_list_b = [], []
  for i in range(len(metric_list_a)):
    if math.isnan(metric_list_a[i]) or math.isnan(metric_list_b[i]):
      continue
    valid_metric_list_a.append(metric_list_a[i])
    valid_metric_list_b.append(metric_list_b[i])

  if not has_enough_trials_for_correlation(
      num_valid_trials=len(valid_metric_list_a),
      num_total_trials=len(metric_list_a),
      min_required_num_correlation_metrics=min_required_num_correlation_metrics,
      min_ratio_of_valid_trials_to_total_trials=min_ratio_of_valid_trials_to_total_trials,
  ):
    logging.warning(
        (
            "Can not compute proxy-task score. "
            "Either metric list has too few valid values: %s and %s."
        ),
        valid_metric_list_a,
        valid_metric_list_b,
    )
    return math.nan, math.nan, valid_metric_list_a, valid_metric_list_b

  corr, p_value = stats.kendalltau(valid_metric_list_a, valid_metric_list_b)
  return corr, p_value, valid_metric_list_a, valid_metric_list_b


def compute_accuracy_correlation_over_step(
    proxy_task_accuracies_2d_array,
    full_training_metrics,
    training_steps,
    plot_dir = "",
    proxy_task_name = "",
):
  """Computes proxy-task accuracy correlation and p-values over training steps."""
  num_valid_full_training_metrics = len(
      get_trials_with_valid_metrics(full_training_metrics)
  )
  logging.info(
      "num_valid_full_training_metrics is %d", num_valid_full_training_metrics
  )
  num_training_steps = proxy_task_accuracies_2d_array.shape[1]
  num_trials = proxy_task_accuracies_2d_array.shape[0]
  correlation_over_step = (
      np.ones(num_training_steps)
      * search_controller_constants.INVALID_CORRELATION
  )
  p_value_over_step = (
      np.ones(num_training_steps) * search_controller_constants.INVALID_P_VALUE
  )
  for training_step_idx in range(num_training_steps):
    logging.info("Processing training_step_idx %d", training_step_idx)
    proxy_task_accuracies_at_this_step = proxy_task_accuracies_2d_array[
        :, training_step_idx
    ]
    # Convert accuracies at this step to a list format.
    proxy_task_metrics = [math.nan] * num_trials
    for trial_idx in range(num_trials):
      if (
          proxy_task_accuracies_at_this_step[trial_idx]
          != search_controller_constants.INVALID_ACCURACY
      ):
        proxy_task_metrics[trial_idx] = proxy_task_accuracies_at_this_step[
            trial_idx
        ]
    # Compute correlation if possible.
    logging.info("proxy_task_metrics is %s", proxy_task_metrics)
    (
        proxy_task_score,
        p_value,
        valid_proxy_task_metrics,
        valid_full_training_metrics,
    ) = compute_rank_correlation(proxy_task_metrics, full_training_metrics)
    # Check if we have invalid correlation at this step.
    if math.isnan(proxy_task_score):
      continue
    correlation_over_step[training_step_idx] = proxy_task_score
    p_value_over_step[training_step_idx] = p_value

    # Plot the correlation at this step.
    if plot_dir:
      training_step = training_steps[training_step_idx]
      file_prefix = "proxy_task_" + proxy_task_name
      plot_filename = os.path.join(
          plot_dir,
          file_prefix
          + "_accuracy_correlation_at_step_{}.png".format(training_step),
      )
      with gcs_utils_using_gcsfuse.file_open(plot_filename, "wb") as f:
        plt.figure()
        plt.plot(valid_full_training_metrics, valid_proxy_task_metrics, "rx")
        plt.title("Correlation-score {}".format(proxy_task_score))
        plt.xlabel("Full-training accuracy")
        plt.ylabel("Proxy-task accuracy")
        plt.savefig(f)
        plt.close()
  return correlation_over_step, p_value_over_step


def compute_median_accuracy_and_training_time_over_step(
    proxy_task_all_trial_training_data,
    proxy_task_accuracies_2d_array,
    proxy_task_training_steps,
    correlation_over_step,
):
  """Returns median-accuracy and training-time over steps."""
  logging.info("Computing median-accuracy and training time over step.")
  num_training_steps = proxy_task_accuracies_2d_array.shape[1]
  median_accuracy_over_step = (
      np.ones(num_training_steps) * search_controller_constants.INVALID_ACCURACY
  )
  median_training_time_hrs_over_step = (
      np.ones(num_training_steps)
      * search_controller_constants.INVALID_TRAINING_TIME
  )
  for training_step_idx in range(num_training_steps):
    logging.info("Processing step-idx %d.", training_step_idx)
    if (
        correlation_over_step[training_step_idx]
        == search_controller_constants.INVALID_CORRELATION
    ):
      continue
    # At this step find median valid accuracy and the corresponding trial.
    accuracies_at_this_step = proxy_task_accuracies_2d_array[
        :, training_step_idx
    ]
    valid_accuracies_at_this_step = accuracies_at_this_step[
        accuracies_at_this_step != search_controller_constants.INVALID_ACCURACY
    ]
    # NOTE: Finding median this way guarantees that the value is
    # part of the array.
    median_accuracy = np.sort(valid_accuracies_at_this_step)[
        len(valid_accuracies_at_this_step) // 2
    ]
    median_accuracy_over_step[training_step_idx] = median_accuracy
    median_trial_idx = np.where(accuracies_at_this_step == median_accuracy)[0][
        0
    ]
    # Find training-time corresponding to this trial.
    trial_training_time_data = proxy_task_all_trial_training_data[
        median_trial_idx
    ].training_time
    training_time_per_step = (
        trial_training_time_data.total_training_time_in_secs
        / trial_training_time_data.total_training_steps
    )
    median_training_time_hrs_at_this_step = (
        training_time_per_step * proxy_task_training_steps[training_step_idx]
    ) / search_controller_constants.NUM_SECS_IN_ONE_HOUR
    median_training_time_hrs_over_step[training_step_idx] = (
        median_training_time_hrs_at_this_step
    )
  return median_accuracy_over_step, median_training_time_hrs_over_step


def _create_proxy_task_search_job_spec(
    proxy_task_config,
    base_proxy_task_search_job_spec,
    full_training_job,
):
  """Returns proxy-task job spec."""
  # Get proxy-task-name and docker-flags for this proxy-task-config.
  proxy_task_args_map = proxy_task_config.docker_args_map
  proxy_task_name = proxy_task_config.name

  proxy_task_search_job_spec = copy.deepcopy(base_proxy_task_search_job_spec)

  # Get root-output directory from base-proxy-task spec.
  root_output_dir = client_utils.get_root_output_dir_from_job_dir(
      client_utils.get_job_dir_for_nas_job(base_proxy_task_search_job_spec)
  )

  # Update retrain related info for job-spec.
  # It will use the same models as the full-training job.
  full_training_search_job_dir = client_utils.get_job_dir_for_nas_job(
      full_training_job
  )
  proxy_task_args_map["retrain_search_job_dir"] = full_training_search_job_dir
  full_training_search_trials_count, _, _ = (
      client_utils.get_num_trials_for_nas_job(full_training_job)
  )
  proxy_task_args_map["retrain_search_job_trials"] = ",".join(
      [
          trial_idx_to_id(trial_idx)
          for trial_idx in range(full_training_search_trials_count)
      ]
  )
  retrain_nas_trial_count = full_training_search_trials_count
  proxy_task_search_job_spec = client_utils.set_docker_args_map_for_nas_job(
      nas_job=proxy_task_search_job_spec, docker_args_map=proxy_task_args_map
  )
  proxy_task_search_job_spec = client_utils.set_num_trials_for_nas_job(
      nas_job=proxy_task_search_job_spec,
      max_trial_count=retrain_nas_trial_count,
      max_parallel_trial_count=retrain_nas_trial_count,
      max_failed_trial_count=retrain_nas_trial_count,
  )

  # Set up new job name and output directory.
  base_job_name = client_utils.get_display_name_for_nas_job(
      base_proxy_task_search_job_spec
  )
  proxy_task_job_name = "ProxyTask_{}_{}".format(base_job_name, proxy_task_name)
  dir_name = proxy_task_job_name + datetime.datetime.now().strftime(
      "_%Y%m%d_%H%M%S"
  )
  proxy_task_job_dir = os.path.join(root_output_dir, dir_name, "nas", "search")
  proxy_task_search_job_spec = client_utils.set_display_name_for_nas_job(
      nas_job=proxy_task_search_job_spec, display_name=proxy_task_job_name
  )
  proxy_task_search_job_spec = client_utils.set_job_dir_for_nas_job(
      nas_job=proxy_task_search_job_spec, job_dir=proxy_task_job_dir
  )
  return proxy_task_search_job_spec


def get_interpolated_step_idx(x1, y_list, y):
  """Returns the interpolated x-val corresponding to the desired y-val."""
  # The desired value is between 'x0' and 'x1'.
  if x1 == 0:
    # This is the only value, so can not interpolate.
    return float(x1)
  x0 = float(x1 - 1)
  y0 = float(y_list[x1 - 1])
  y1 = float(y_list[x1])
  # (x-x0)/(y-y0) = (x1-x0)/(y1-y0)
  x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
  return x


def get_interpolated_val(
    x1, y_list, x
):
  """Returns the interpolated y-val corresponding to the x-val."""
  # The desired value is between 'x0' and 'x1'.
  if x1 == 0:
    # This is the only value, so can not interpolate.
    return float(y_list[x1])
  x0 = float(x1 - 1)
  y0 = float(y_list[x1 - 1])
  y1 = float(y_list[x1])
  # (y-y0)/(x-x0) = (y1-y0)/(x1-x0)
  y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
  return y


def check_stopping_condition_and_update_proxy_task_stats(
    proxy_task_stats,
    early_stop_proxy_task_if_not_best,
    desired_accuracy_correlation = None,
    desired_accuracy = None,
    training_time_hrs_limit = None,
    best_training_time_hrs = None,
):
  """Returns True if proxy-task stopping condition is met and updates proxy-task-stats."""

  # NOTE: We need to record the very first instance when the stopping
  # condition is met.
  for step_idx in range(len(proxy_task_stats.training_steps)):
    correlation = proxy_task_stats.accuracy_correlation_over_step[step_idx]
    accuracy = proxy_task_stats.median_accuracy_over_step[step_idx]
    training_time_hrs = proxy_task_stats.median_training_time_hrs_over_step[
        step_idx
    ]
    if (
        desired_accuracy_correlation
        and correlation >= desired_accuracy_correlation
    ):
      logging.info(
          "Found desired-accuracy-correlation score %f >= %f",
          correlation,
          desired_accuracy_correlation,
      )
      interp_step_idx = get_interpolated_step_idx(
          x1=step_idx,
          y_list=proxy_task_stats.accuracy_correlation_over_step,
          y=desired_accuracy_correlation,
      )
      proxy_task_stats.stopping_state = (
          ProxyTaskStoppingState.MET_DESIRED_ACCURACY_CORRELATION
      )
      proxy_task_stats.final_training_time_in_hours = get_interpolated_val(
          x1=step_idx,
          y_list=proxy_task_stats.median_training_time_hrs_over_step,
          x=interp_step_idx,
      )
      proxy_task_stats.final_training_steps = int(
          get_interpolated_val(
              x1=step_idx,
              y_list=proxy_task_stats.training_steps,
              x=interp_step_idx,
          )
      )
      return True
    if desired_accuracy and accuracy >= desired_accuracy:
      logging.info(
          "Found desired-accuracy %f >= %f", accuracy, desired_accuracy
      )
      interp_step_idx = get_interpolated_step_idx(
          x1=step_idx,
          y_list=proxy_task_stats.median_accuracy_over_step,
          y=desired_accuracy,
      )
      proxy_task_stats.stopping_state = (
          ProxyTaskStoppingState.MET_DESIRED_ACCURACY
      )
      proxy_task_stats.final_training_time_in_hours = get_interpolated_val(
          x1=step_idx,
          y_list=proxy_task_stats.median_training_time_hrs_over_step,
          x=interp_step_idx,
      )
      proxy_task_stats.final_training_steps = int(
          get_interpolated_val(
              x1=step_idx,
              y_list=proxy_task_stats.training_steps,
              x=interp_step_idx,
          )
      )
      return True
    if training_time_hrs_limit and training_time_hrs >= training_time_hrs_limit:
      logging.info(
          "Exceeded training-time hours %f >= %f",
          training_time_hrs,
          training_time_hrs_limit,
      )
      proxy_task_stats.stopping_state = (
          ProxyTaskStoppingState.EXCEEDED_TRAIN_TIME_LIMIT
      )
      return True
    # The best_training_time_hrs corrsponds to the best proxy-task so far.
    # The best-proxy-task should have previously either met the
    # desired-accuracy or the desired-accuracy-correlation within
    # the train-time-limit. If no previous proxy-task could meet these
    # criterion, then the best_training_time_hrs is 'None'.
    if (
        early_stop_proxy_task_if_not_best
        and best_training_time_hrs
        and training_time_hrs >= best_training_time_hrs
    ):
      logging.info(
          "Exceeded best training-time hours %f >= %f",
          training_time_hrs,
          best_training_time_hrs,
      )
      proxy_task_stats.stopping_state = (
          ProxyTaskStoppingState.EXCEEDED_BEST_TRAIN_TIME
      )
      return True
  return False


def has_enough_trials_for_correlation(
    num_valid_trials,
    num_total_trials,
    min_required_num_correlation_metrics = search_controller_constants.MIN_REQUIRED_NUM_CORRELATION_METRICS,
    min_ratio_of_valid_trials_to_total_trials = search_controller_constants.MIN_RATIO_OF_VALID_TRIALS_TO_TOTAL_TRIALS,
):
  """Returns True if we have enough trials for correlation computation."""
  if num_valid_trials < min_required_num_correlation_metrics:
    logging.info(
        (
            "num_valid_trials %d is less than "
            "min_required_num_correlation_metrics %d"
        ),
        num_valid_trials,
        min_required_num_correlation_metrics,
    )
    return False
  ratio_valid_to_total_trials = float(num_valid_trials) / num_total_trials
  if ratio_valid_to_total_trials < min_ratio_of_valid_trials_to_total_trials:
    logging.info(
        (
            "ratio_valid_to_total_trials %f is less than "
            "min_ratio_of_valid_trials_to_total_trials %f"
        ),
        ratio_valid_to_total_trials,
        min_ratio_of_valid_trials_to_total_trials,
    )
    return False
  return True


def have_excess_trial_failures(
    running_proxy_task_trials,
    full_training_accuracy_metrics,
):
  """Checks for excess trial failures and signals proxy-task to stop if needed."""
  logging.info("Checking for excess trial failures.")
  # Check if a lot of trials have failed and there is no scope of
  # computing correlation score any further. In this case, post
  # stop message to all trials.
  valid_full_training_trials = get_trials_with_valid_metrics(
      full_training_accuracy_metrics
  )
  common_trials = valid_full_training_trials.intersection(
      running_proxy_task_trials
  )
  if not has_enough_trials_for_correlation(
      num_valid_trials=len(common_trials),
      num_total_trials=len(full_training_accuracy_metrics),
  ):
    logging.info("valid_full_training_trials: %s", valid_full_training_trials)
    logging.info("running_proxy_task_trials: %s", running_proxy_task_trials)
    logging.info(
        "Possible common trials for correlation computation are: %s",
        common_trials,
    )
    logging.info(
        "Do not have enough trials to compute correlation score. "
        "Posting stop-training message."
    )
    return True
  else:
    return False


def get_valid_list_for_plotting(
    x,
    invalid_x,
    y,
    invalid_y,
):
  """Returns valid x and y lists for plotting."""
  if len(x) != len(y):
    raise ValueError(
        "Lists %s and %s should have same length for plotting." % x, y
    )
  ret_x = []
  ret_y = []
  for val_x, val_y in zip(x, y):
    if invalid_x and val_x == invalid_x:
      continue
    if invalid_y and val_y == invalid_y:
      continue
    ret_x.append(val_x)
    ret_y.append(val_y)
  return ret_x, ret_y


def _save_proxy_task_stats_plots(
    proxy_task_name, proxy_task_stats, plot_dir
):
  """Saves proxy-task-stats related plots to disk."""
  file_prefix = "proxy_task_" + proxy_task_name
  # Save accuracy-correlation vs training-steps plot.
  plot_filename = os.path.join(
      plot_dir, file_prefix + "_accuracy_correlation_vs_training_steps.png"
  )
  with gcs_utils_using_gcsfuse.file_open(plot_filename, "wb") as f:
    plt.figure()
    x, y = get_valid_list_for_plotting(
        x=proxy_task_stats.training_steps,
        invalid_x=None,
        y=proxy_task_stats.accuracy_correlation_over_step,
        invalid_y=search_controller_constants.INVALID_CORRELATION,
    )
    plt.plot(x, y)
    plt.title("Accuracy-correlation vs Training-steps")
    plt.xlabel("Training-steps")
    plt.ylabel("Accuracy-correlation")
    plt.savefig(f)
    plt.close()
  # Save accuracy-correlation vs training-time plot.
  plot_filename = os.path.join(
      plot_dir, file_prefix + "_accuracy_correlation_vs_training_time.png"
  )
  with gcs_utils_using_gcsfuse.file_open(plot_filename, "wb") as f:
    plt.figure()
    x, y = get_valid_list_for_plotting(
        x=proxy_task_stats.median_training_time_hrs_over_step,
        invalid_x=search_controller_constants.INVALID_TRAINING_TIME,
        y=proxy_task_stats.accuracy_correlation_over_step,
        invalid_y=search_controller_constants.INVALID_CORRELATION,
    )
    plt.plot(x, y)
    plt.title("Accuracy-correlation vs Training-time")
    plt.xlabel("Training-time (hrs)")
    plt.ylabel("Accuracy-correlation")
    plt.savefig(f)
    plt.close()
  # Save accuracy vs training-steps plot.
  plot_filename = os.path.join(
      plot_dir, file_prefix + "_accuracy_vs_training_steps.png"
  )
  with gcs_utils_using_gcsfuse.file_open(plot_filename, "wb") as f:
    plt.figure()
    x, y = get_valid_list_for_plotting(
        x=proxy_task_stats.training_steps,
        invalid_x=None,
        y=proxy_task_stats.median_accuracy_over_step,
        invalid_y=search_controller_constants.INVALID_ACCURACY,
    )
    plt.plot(x, y)
    plt.title("Accuracy vs Training-steps")
    plt.xlabel("Training-steps")
    plt.ylabel("Accuracy")
    plt.savefig(f)
    plt.close()
  # Save accuracy vs training-time plot.
  plot_filename = os.path.join(
      plot_dir, file_prefix + "_accuracy_vs_training_time.png"
  )
  with gcs_utils_using_gcsfuse.file_open(plot_filename, "wb") as f:
    plt.figure()
    x, y = get_valid_list_for_plotting(
        x=proxy_task_stats.median_training_time_hrs_over_step,
        invalid_x=search_controller_constants.INVALID_TRAINING_TIME,
        y=proxy_task_stats.median_accuracy_over_step,
        invalid_y=search_controller_constants.INVALID_ACCURACY,
    )
    plt.plot(x, y)
    plt.title("Accuracy vs Training-time")
    plt.xlabel("Training-time (hrs)")
    plt.ylabel("Accuracy")
    plt.savefig(f)
    plt.close()


def _update_proxy_task_accuracy_correlation_stats(
    proxy_task_name,
    proxy_task_nas_job,
    full_training_accuracy_metrics,
    search_controller_dir,
    proxy_task_stats,
):
  """Updates proxy-task stats such as correlation and accuracy based on training so far."""
  # Compute accuracy correlation per training-step so far.
  # Also find latest valid correlation score and training-step.
  proxy_task_all_trial_training_data = (
      proxy_task_controller_utils.load_all_trial_training_metrics(
          proxy_task_nas_job
      )
  )
  proxy_task_accuracies_2d_array, proxy_task_training_steps = (
      convert_all_trial_training_data_to_2d_array_of_accuracies(
          proxy_task_all_trial_training_data
      )
  )
  if not proxy_task_training_steps:
    logging.info("Do not have proxy-task accuracies yet.")
    return
  logging.info("proxy_task_training_steps is %s", proxy_task_training_steps)
  logging.info(
      "proxy_task_accuracies_2d_array is %s", proxy_task_accuracies_2d_array
  )
  correlation_over_step, p_value_over_step = (
      compute_accuracy_correlation_over_step(
          proxy_task_accuracies_2d_array=proxy_task_accuracies_2d_array,
          full_training_metrics=full_training_accuracy_metrics,
          training_steps=proxy_task_training_steps,
          proxy_task_name=proxy_task_name,
          plot_dir=search_controller_dir,
      )
  )
  logging.info("correlation_over_step is %s", correlation_over_step)
  logging.info("p_value_over_step is %s", p_value_over_step)

  # Find median-accuracy and median-training-time over steps.
  median_accuracy_over_step, median_training_time_hrs_over_step = (
      compute_median_accuracy_and_training_time_over_step(
          proxy_task_all_trial_training_data=proxy_task_all_trial_training_data,
          proxy_task_accuracies_2d_array=proxy_task_accuracies_2d_array,
          proxy_task_training_steps=proxy_task_training_steps,
          correlation_over_step=correlation_over_step,
      )
  )
  logging.info("median_accuracy_over_step: %s", median_accuracy_over_step)
  logging.info(
      "median_training_time_hrs_over_step: %s",
      median_training_time_hrs_over_step,
  )

  # Update proxy_task_stats and save plots.
  proxy_task_stats.training_steps = proxy_task_training_steps
  proxy_task_stats.accuracy_correlation_over_step = (
      correlation_over_step.tolist()
  )
  proxy_task_stats.accuracy_correlation_p_value_over_step = (
      p_value_over_step.tolist()
  )
  proxy_task_stats.median_accuracy_over_step = (
      median_accuracy_over_step.tolist()
  )
  proxy_task_stats.median_training_time_hrs_over_step = (
      median_training_time_hrs_over_step.tolist()
  )
  _save_proxy_task_stats_plots(
      proxy_task_name=proxy_task_name,
      proxy_task_stats=proxy_task_stats,
      plot_dir=search_controller_dir,
  )


def _update_proxy_task_latency_correlation_stats(
    proxy_task_name,
    proxy_task_nas_job,
    full_training_latency_metrics,
    search_controller_dir,
    proxy_task_stats,
):
  """Updates proxy-task latency correlation stats after training finishes."""
  proxy_task_all_trial_training_data = (
      proxy_task_controller_utils.load_all_trial_training_metrics(
          proxy_task_nas_job
      )
  )
  proxy_task_latency_metrics = (
      proxy_task_controller_utils.get_latency_metrics_for_all_trials(
          proxy_task_all_trial_training_data
      )
  )
  (
      proxy_task_stats.latency_correlation,
      proxy_task_stats.latency_correlation_p_value,
      valid_full_training_latency_metrics,
      valid_proxy_task_latency_metrics,
  ) = compute_rank_correlation(
      metric_list_a=full_training_latency_metrics,
      metric_list_b=proxy_task_latency_metrics,
  )
  if math.isnan(proxy_task_stats.latency_correlation):
    proxy_task_stats.latency_correlation = (
        search_controller_constants.INVALID_CORRELATION
    )
  logging.info(
      "Proxy-task latency correlation is %f",
      proxy_task_stats.latency_correlation,
  )
  file_prefix = "proxy_task_" + proxy_task_name
  plot_filename = os.path.join(
      search_controller_dir, file_prefix + "_latency_correlation.png"
  )
  with gcs_utils_using_gcsfuse.file_open(plot_filename, "wb") as f:
    plt.figure()
    plt.plot(
        valid_full_training_latency_metrics,
        valid_proxy_task_latency_metrics,
        "rx",
    )
    plt.title(
        "Correlation-score {}".format(proxy_task_stats.latency_correlation)
    )
    plt.xlabel("Full-training latency")
    plt.ylabel("Proxy-task latency")
    plt.savefig(f)
    plt.close()


def _monitor_proxy_task_and_stop_trials(
    proxy_task_name,
    search_controller_dir,
    proxy_task_search_job_name,
    service_endpoint,
    project_id,
    region,
    full_training_accuracy_metrics,
    full_training_latency_metrics,
    desired_accuracy_correlation,
    desired_accuracy,
    training_time_hrs_limit,
    desired_latency_correlation,
    best_training_time_hrs,
    early_stop_proxy_task_if_not_best,
):
  """Monitors the proxy-task job progress, stops trials early if needed, and returns proxy-task-stats."""
  logging.info(
      "Starting monitoring of trials for proxy-task: %s.",
      proxy_task_search_job_name,
  )
  proxy_task_nas_job_id = client_utils.nas_job_id_from_nas_job_name(
      proxy_task_search_job_name
  )
  proxy_task_stats = ProxyTaskStats()

  # Post max-training time-limit for proxy-task even before starting to monitor.
  # So that even if the controller dies, the child proxy-task will not exceed
  # its training limit time.
  try:
    proxy_task_nas_job = client_utils.get_job(
        vertex_ai_endpoint=service_endpoint,
        project_id=project_id,
        location=region,
        job_id=proxy_task_nas_job_id,
    )
  except Exception:  # pylint: disable=broad-except
    logging.info("Proxy task job %s failed", proxy_task_search_job_name)
    proxy_task_stats.stopping_state = ProxyTaskStoppingState.JOB_FAILED
    return proxy_task_stats
  proxy_task_controller_utils.set_stop_training_for_all_trials(
      nas_job=proxy_task_nas_job,
      max_training_time_in_hrs=training_time_hrs_limit,
  )

  while True:
    # Get proxy-task nas job status.
    try:
      proxy_task_nas_job = client_utils.get_job(
          vertex_ai_endpoint=service_endpoint,
          project_id=project_id,
          location=region,
          job_id=proxy_task_nas_job_id,
      )
    except Exception:  # pylint: disable=broad-except
      logging.info("Proxy task job %s failed", proxy_task_search_job_name)
      proxy_task_stats.stopping_state = ProxyTaskStoppingState.JOB_FAILED
      return proxy_task_stats

    # Is proxy-task job running?
    proxy_task_controller_utils.post_controller_alive_time_stamp(
        proxy_task_nas_job
    )
    if not client_utils.is_nas_job_active(proxy_task_nas_job):
      logging.info(
          "Stopping monitoring for proxy-task nas job because "
          "it is no longer active."
      )
      break

    # Has the proxy-task been asked to stop already?
    # If yes, then do not do anymore proxy-task-stats updates.
    # Just wait for the job to shut down.
    if proxy_task_stats.posted_stop_trials_message:
      logging.info("Waiting for proxy-task job to shut down completely.")
    else:
      # Update proxy-task stats based on training so far.
      _update_proxy_task_accuracy_correlation_stats(
          proxy_task_name=proxy_task_name,
          proxy_task_nas_job=proxy_task_nas_job,
          full_training_accuracy_metrics=full_training_accuracy_metrics,
          search_controller_dir=search_controller_dir,
          proxy_task_stats=proxy_task_stats,
      )

      # Check for stopping condition.
      if check_stopping_condition_and_update_proxy_task_stats(
          proxy_task_stats=proxy_task_stats,
          desired_accuracy_correlation=desired_accuracy_correlation,
          desired_accuracy=desired_accuracy,
          training_time_hrs_limit=training_time_hrs_limit,
          best_training_time_hrs=best_training_time_hrs,
          early_stop_proxy_task_if_not_best=early_stop_proxy_task_if_not_best,
      ):
        logging.info("Posting stop-training message.")
        proxy_task_controller_utils.set_stop_training_for_all_trials(
            nas_job=proxy_task_nas_job,
            desired_training_step_pct=proxy_task_controller_utils_constants.MIN_TRAINING_STEP_PCT,
        )
        proxy_task_stats.posted_stop_trials_message = True

    # sleep
    logging.info(
        "Sleeping for %d secs",
        proxy_task_controller_utils_constants.TRIAL_MONITOR_SLEEP_SECS,
    )
    time.sleep(proxy_task_controller_utils_constants.TRIAL_MONITOR_SLEEP_SECS)

  ############## The proxy-task job has stopped now. ################
  # Load the job again and compute latency correlation.
  if full_training_latency_metrics:
    proxy_task_nas_job = client_utils.get_job(
        vertex_ai_endpoint=service_endpoint,
        project_id=project_id,
        location=region,
        job_id=proxy_task_nas_job_id,
    )
    _update_proxy_task_latency_correlation_stats(
        proxy_task_name=proxy_task_name,
        proxy_task_nas_job=proxy_task_nas_job,
        full_training_latency_metrics=full_training_latency_metrics,
        search_controller_dir=search_controller_dir,
        proxy_task_stats=proxy_task_stats,
    )
    if (
        desired_latency_correlation
        and proxy_task_stats.latency_correlation < desired_latency_correlation
    ):
      logging.info(
          "The desired_latency_correlation %f was not met.",
          desired_latency_correlation,
      )
      proxy_task_stats.stopping_state = (
          ProxyTaskStoppingState.FAILED_DESIRED_LATENCY_CORRELATION
      )

  # Check if proxy-task ran till completion without running into any
  # accuracy, correlation, or time limits.
  if (
      proxy_task_stats.stopping_state
      == ProxyTaskStoppingState.INVALID_STOPPING_STATE
  ):
    proxy_task_stats.stopping_state = ProxyTaskStoppingState.RAN_TILL_COMPLETION
  return proxy_task_stats


def trial_id_to_idx(trial_id):
  return int(trial_id) - 1


def trial_idx_to_id(trial_idx):
  return str(trial_idx + 1)


def convert_metric_values_map_to_list(
    metric_values_map,
    total_num_trials
):
  """Converts valid-metrics map to a list of valid and invalid metric values (NAN) over all trials."""
  trial_metrics = [math.nan] * total_num_trials
  for trial_id, metric_value in metric_values_map.items():
    trial_metrics[trial_id_to_idx(trial_id)] = metric_value
  return trial_metrics


def _load_full_training_reference_metrics(
    proxy_task_model_selection_job_id,
    proxy_task_model_selection_job_region,
    proxy_task_model_selection_service_endpoint,
    project_id,
    requires_latency,
):
  """Returns a tuple of full-training-job-id, full-training job, its accuracy metrics, and latency metrics (if latency is available)."""
  # Load the proxy-task model selection job.
  model_selection_job = client_utils.get_job(
      vertex_ai_endpoint=proxy_task_model_selection_service_endpoint,
      project_id=project_id,
      location=proxy_task_model_selection_job_region,
      job_id=proxy_task_model_selection_job_id,
      job_type="custom",
  )
  logging.info(
      "Loaded model-selection job %s", proxy_task_model_selection_job_id
  )
  # Load model-selection state.
  model_selection_args_map = client_utils.get_docker_args_map_for_custom_job(
      model_selection_job
  )
  model_selection_dir = model_selection_args_map["model_selection_dir"]
  logging.info("model_selection_dir is %s", model_selection_dir)
  model_selection_state, _ = (
      proxy_task_model_selection_lib.try_load_model_selection_state(
          model_selection_dir
      )
  )
  logging.info("model_selection_state is %s", model_selection_state)
  if not model_selection_state.iterations:
    raise ValueError("model_selection_state has not iterations.")

  # Find the last full-training job iteration for this model-selection job.
  accuracy_metric_id = model_selection_state.accuracy_metric_id
  latency_metric_id = model_selection_state.latency_metric_id
  full_training_job_name = model_selection_state.iterations[-1].search_job_name
  logging.info("full_training_job_name is %s", full_training_job_name)
  full_training_job_id = client_utils.nas_job_id_from_nas_job_name(
      full_training_job_name
  )
  full_training_job = client_utils.get_job(
      vertex_ai_endpoint=proxy_task_model_selection_service_endpoint,
      project_id=project_id,
      location=proxy_task_model_selection_job_region,
      job_id=full_training_job_id,
  )
  total_full_training_trials, _, _ = client_utils.get_num_trials_for_nas_job(
      full_training_job
  )
  valid_full_training_accuracy_metrics_map = (
      client_utils.load_valid_trial_metrics_for_job(
          job=full_training_job, metric_id=accuracy_metric_id
      ))
  full_training_accuracy_metrics = convert_metric_values_map_to_list(
      metric_values_map=valid_full_training_accuracy_metrics_map,
      total_num_trials=total_full_training_trials
  )
  logging.info(
      "Full training accuracy metrics are: %s", full_training_accuracy_metrics
  )
  valid_full_training_trials = get_trials_with_valid_metrics(
      full_training_accuracy_metrics
  )
  if not has_enough_trials_for_correlation(
      num_valid_trials=len(valid_full_training_trials),
      num_total_trials=len(full_training_accuracy_metrics),
  ):
    raise ValueError(
        "Not enough valid full_training_accuracy_metrics %s"
        % full_training_accuracy_metrics
    )
  if requires_latency:
    valid_full_training_latency_metrics_map = (
        client_utils.load_valid_trial_metrics_for_job(
            job=full_training_job, metric_id=latency_metric_id
        ))
    full_training_latency_metrics = convert_metric_values_map_to_list(
        metric_values_map=valid_full_training_latency_metrics_map,
        total_num_trials=total_full_training_trials
    )
    logging.info(
        "Full training latency metrics are: %s", full_training_latency_metrics
    )
  else:
    full_training_latency_metrics = None
  return (
      full_training_job_id,
      full_training_job,
      full_training_accuracy_metrics,
      full_training_latency_metrics,
  )


def get_best_proxy_task_training_time_hrs(
    search_controller_state,
):
  """Returns the training-time for the best proxy-task."""
  if search_controller_state.best_proxy_task_name:
    return search_controller_state.proxy_tasks_map[
        search_controller_state.best_proxy_task_name
    ].proxy_task_stats.final_training_time_in_hours
  return None


def update_best_proxy_task(search_controller_state):
  """Finds and updates the best proxy-task in search-controller state."""

  def _is_successful(stopping_state):
    return stopping_state in [
        ProxyTaskStoppingState.MET_DESIRED_ACCURACY_CORRELATION,
        ProxyTaskStoppingState.MET_DESIRED_ACCURACY,
    ]

  search_controller_state.best_proxy_task_name = ""
  for (
      proxy_task_name,
      proxy_task_state,
  ) in search_controller_state.proxy_tasks_map.items():
    # Check if this is the best proxy-task so far.
    is_curr_proxy_task_the_best = False
    curr_proxy_task_stats = proxy_task_state.proxy_task_stats
    best_proxy_task_training_time_in_hours = (
        get_best_proxy_task_training_time_hrs(search_controller_state)
    )
    if best_proxy_task_training_time_in_hours:
      if (
          _is_successful(curr_proxy_task_stats.stopping_state)
          and curr_proxy_task_stats.final_training_time_in_hours
          < best_proxy_task_training_time_in_hours
      ):
        is_curr_proxy_task_the_best = True
    else:
      # We do not have any other proxy-task to compare against.
      is_curr_proxy_task_the_best = _is_successful(
          curr_proxy_task_stats.stopping_state
      )
    if is_curr_proxy_task_the_best:
      search_controller_state.best_proxy_task_name = proxy_task_name


def run_search_controller_loop(
    proxy_task_model_selection_job_id,
    proxy_task_model_selection_job_region,
    proxy_task_model_selection_service_endpoint,
    proxy_task_config_generator_module,
    search_controller_dir,
    service_endpoint,
    region,
    project_id,
    desired_accuracy_correlation,
    desired_accuracy,
    training_time_hrs_limit,
    desired_latency_correlation,
    early_stop_proxy_task_if_not_best,
):
  """Runs search controller loop to find the best proxy task."""
  base_proxy_task_search_job_spec, latency_calculator_job_spec = (
      proxy_task_controller_utils.load_search_and_latency_job_spec(
          search_controller_dir
      )
  )

  # Load the reference full-training job and its accuracy and latency scores.
  # It will be used later to load reference models and their scores.
  (
      _,
      full_training_job,
      full_training_accuracy_metrics,
      full_training_latency_metrics,
  ) = _load_full_training_reference_metrics(
      proxy_task_model_selection_job_id=proxy_task_model_selection_job_id,
      proxy_task_model_selection_job_region=proxy_task_model_selection_job_region,
      proxy_task_model_selection_service_endpoint=proxy_task_model_selection_service_endpoint,
      project_id=project_id,
      requires_latency=(latency_calculator_job_spec is not None),
  )

  # Get the list of proxy-tasks to process.
  base_proxy_task_search_job_args_map = (
      client_utils.get_docker_args_map_for_nas_job(
          base_proxy_task_search_job_spec
      )
  )
  proxy_task_config_list = client_utils.get_module(
      module_path=proxy_task_config_generator_module
  )(base_proxy_task_search_job_args_map)

  # Load search-controller state so far.
  # The job may have restarted.
  search_controller_state = try_load_search_controller_state(
      search_controller_dir
  )

  # Start loop to process each proxy-task config:
  # 1. Launch the next proxy-task job.
  # 2. Monitor and wait for the proxy task job to finish. Do early stopping
  #    if necessary.
  # 3. Save the proxy-task correlation score.
  # 4. Re-rank all evaluated proxy-tasks till now.
  for proxy_task_config in proxy_task_config_list:
    proxy_task_name = proxy_task_config.name
    # Check proxy-task job state:
    # The search-controller job may have restarted. Check if this proxy-task
    # was processed before.
    skip_proxy_task_job_launch = False
    skip_proxy_task_monitoring = False
    if proxy_task_name in search_controller_state.proxy_tasks_map:
      skip_proxy_task_job_launch = True
      logging.info("Proxy task %s was launched before.", proxy_task_name)
      if search_controller_state.proxy_tasks_map[
          proxy_task_name
      ].proxy_task_stats.training_steps:
        skip_proxy_task_monitoring = True
        logging.info(
            "Proxy task stats for %s are already available.", proxy_task_name
        )

    if not skip_proxy_task_job_launch:
      # Create proxy-task search job spec for this proxy-task-config.
      proxy_task_search_job_spec = _create_proxy_task_search_job_spec(
          proxy_task_config=proxy_task_config,
          base_proxy_task_search_job_spec=base_proxy_task_search_job_spec,
          full_training_job=full_training_job,
      )
      # Launch this proxy-task candidate.
      proxy_task_controller_utils.post_controller_alive_time_stamp(
          proxy_task_search_job_spec
      )
      (
          search_job_name,
          search_job_link,
          latency_calculator_job_name,
          latency_calculator_job_link,
      ) = client_utils.launch_cloud_nas_and_latency_job(
          search_job_spec=proxy_task_search_job_spec,
          latency_calculator_job_spec=latency_calculator_job_spec,
          service_endpoint=service_endpoint,
          region=region,
          project_id=project_id,
      )
      # Update and save search-controller state.
      proxy_task_controller_utils.save_current_nas_job_name(
          nas_job_name=search_job_name, controller_dir=search_controller_dir
      )
      search_controller_state.proxy_tasks_map[proxy_task_name] = ProxyTaskState(
          proxy_task_name=proxy_task_name,
          search_job_name=search_job_name,
          search_job_link=search_job_link,
          latency_calculator_job_name=latency_calculator_job_name,
          latency_calculator_job_link=latency_calculator_job_link,
          proxy_task_stats=ProxyTaskStats(),
      )
      save_search_controller_state_file(
          filepath=get_search_controller_state_file(search_controller_dir),
          search_controller_state=search_controller_state,
      )

    if not skip_proxy_task_monitoring:
      # Monitor this proxy-task job and wait for it to finish.
      best_training_time_hrs = get_best_proxy_task_training_time_hrs(
          search_controller_state
      )
      if best_training_time_hrs:
        logging.info("best_training_time_hrs is %f", best_training_time_hrs)
      proxy_task_stats = _monitor_proxy_task_and_stop_trials(
          proxy_task_name=proxy_task_name,
          search_controller_dir=search_controller_dir,
          proxy_task_search_job_name=search_job_name,
          service_endpoint=service_endpoint,
          project_id=project_id,
          region=region,
          full_training_accuracy_metrics=full_training_accuracy_metrics,
          full_training_latency_metrics=full_training_latency_metrics,
          desired_accuracy_correlation=desired_accuracy_correlation,
          desired_accuracy=desired_accuracy,
          training_time_hrs_limit=training_time_hrs_limit,
          desired_latency_correlation=desired_latency_correlation,
          best_training_time_hrs=best_training_time_hrs,
          early_stop_proxy_task_if_not_best=early_stop_proxy_task_if_not_best,
      )
      # Update and save search-controller state.
      # Also update the best proxy-task state so far.
      search_controller_state.proxy_tasks_map[
          proxy_task_name
      ].proxy_task_stats = proxy_task_stats
      update_best_proxy_task(search_controller_state=search_controller_state)
      save_search_controller_state_file(
          filepath=get_search_controller_state_file(search_controller_dir),
          search_controller_state=search_controller_state,
      )
      # Clear current nas-job name.
      proxy_task_controller_utils.save_current_nas_job_name(
          nas_job_name="", controller_dir=search_controller_dir
      )
