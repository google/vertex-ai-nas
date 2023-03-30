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
"""Proxy-task controller common utility functions."""

import logging
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import vertex_client_utils as client_utils
from gcs_utils import gcs_utils_using_cloud_storage as gcs_utils
from proxy_task import proxy_task_controller_utils_constants
from proxy_task import proxy_task_utils


def set_stop_training_for_all_trials(
    nas_job,
    desired_training_step_pct = None,
    max_training_time_in_hrs = None,
):
  total_num_trials, _, _ = client_utils.get_num_trials_for_nas_job(nas_job)
  for trial_idx in range(total_num_trials):
    trial_id = str(trial_idx + 1)
    trial_dir = client_utils.get_search_trial_dir(
        job=nas_job, trial_id=trial_id
    )
    proxy_task_utils.set_stop_training(
        model_dir=trial_dir,
        desired_training_step_pct=desired_training_step_pct,
        max_training_time_in_hrs=max_training_time_in_hrs,
    )


def load_trial_training_metrics(
    trial_id, nas_job
):
  trial_dir = client_utils.get_search_trial_dir(job=nas_job, trial_id=trial_id)
  trial_training_metrics_file = (
      proxy_task_utils.get_trial_training_metrics_file(trial_dir)
  )
  return proxy_task_utils.read_trial_training_metrics_file(
      trial_training_metrics_file
  )


def load_all_trial_training_metrics(
    proxy_task_nas_job
):
  """Returns a list of all trial-training-metrics for the proxy-task job."""
  num_training_trials, _, _ = client_utils.get_num_trials_for_nas_job(
      proxy_task_nas_job)
  all_trial_training_data = []
  for trial_idx in range(num_training_trials):
    trial_id = str(trial_idx + 1)
    trial_training_metrics = load_trial_training_metrics(
        trial_id=trial_id, nas_job=proxy_task_nas_job)
    all_trial_training_data.append(trial_training_metrics)
  return all_trial_training_data


def get_latency_metrics_for_all_trials(
    all_trial_training_data
):
  """Returns latency metrics for all trials with invalid values set as 'nan'."""
  if not all_trial_training_data:
    return []
  num_trials = len(all_trial_training_data)
  latency_metrics = [math.nan] * num_trials
  for trial_idx in range(num_trials):
    if all_trial_training_data[
        trial_idx] and all_trial_training_data[trial_idx].latency > 0.0:
      latency_metrics[trial_idx] = all_trial_training_data[trial_idx].latency
  return latency_metrics


def load_search_and_latency_job_spec(
    controller_dir):
  """Loads and returns search and latency job specs from controller directory.
  """
  # Load search job spec file
  search_job_spec_json_file = client_utils.get_search_job_spec_file(
      controller_dir)
  search_job_spec = gcs_utils.load_json(search_job_spec_json_file)
  if not search_job_spec:
    raise ValueError("Could not find search job spec file %s" %
                     search_job_spec_json_file)
  logging.info("Loaded search job spec file: %s", search_job_spec)

  # Load latency calculator job spec.
  latency_calculator_job_spec_json_file = (
      client_utils.get_latency_calculator_job_spec_file(controller_dir)
  )
  latency_calculator_job_spec = gcs_utils.load_json(
      latency_calculator_job_spec_json_file)
  if latency_calculator_job_spec:
    logging.info("Loaded latency-calculator job spec file: %s",
                 latency_calculator_job_spec)
  return search_job_spec, latency_calculator_job_spec


def post_controller_alive_time_stamp(nas_job):  # pylint: disable=unused-argument
  """Posts a controller-alive time-stamp to each trial of a NAS job."""
  # TODO: For now will disable this heartbeat check.
  # # NOTE: Post the timestamp to each trial to avoid multiple trials
  # # reading from the same location.
  # total_num_trials, _, _ = client_utils.get_num_trials_for_nas_job(nas_job)
  # for trial_idx in range(total_num_trials):
  #   trial_id = str(trial_idx + 1)
  #   trial_dir = client_utils.get_search_trial_dir(
  #       job=nas_job, trial_id=trial_id)
  #   # Post current time stamp.
  #   timestamp_file = os.path.join(
  #       trial_dir,
  #       proxy_task_controller_utils_constants.CONTROLLER_ALIVE_TIMESTAMP_FILE)
  #   timestamp_dict = {
  #       proxy_task_controller_utils_constants.CONTROLLER_ALIVE_TIMESTAMP_KEY:
  #           time.time()
  #   }
  #   gcs_utils.save_json(filepath=timestamp_file, json_data=timestamp_dict)


def save_current_nas_job_name(nas_job_name, controller_dir):
  """Saves current nas job name to controller directory."""
  filename = os.path.join(
      controller_dir, proxy_task_controller_utils_constants
      .CONTROLLER_CURRENT_NAS_JOB_NAME_FILE)
  json_dict = {
      proxy_task_controller_utils_constants.CONTROLLER_CURRENT_NAS_JOB_NAME_KEY:
          nas_job_name
  }
  gcs_utils.save_json(filepath=filename, json_data=json_dict)
