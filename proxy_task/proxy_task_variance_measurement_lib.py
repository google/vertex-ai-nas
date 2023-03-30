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
"""Library functions to compute proxy-task model variance and smoothness.

Variance measures the repeatability of a model. A model trained for multiple
times with the same setting should yield similar final validation scores.

Smoothness measures the standard-deviation of the validation score towards the
end of the training. If the score frequently goes up and down, it tells a poor
smoothness.
"""

import copy
import dataclasses
import datetime
import logging
import os
from typing import Any, Dict, List, Mapping, Sequence

import global_variables
import vertex_client_utils as client_utils
from gcs_utils import gcs_utils_using_cloud_storage as gcs_utils
from proxy_task import proxy_task_controller_utils
# TODO: This lib should not depend on model_selection lib,
#                    move common functions to a utility lib.
from proxy_task import proxy_task_model_selection_lib as model_selection
from proxy_task import proxy_task_utils
from proxy_task import proxy_task_variance_measurement_lib_constants as variance_measurement_constants
import numpy as np
from scipy import interpolate


@dataclasses.dataclass
class MeasurementState:
  """Class for variance and smoothness measurement job state.

  Attributes:
    workable_model_dir: Directory of found workable model.
    training_job_name: NAS job name of training workable models.
    finished_model_dirs: A list of directories of finished training models.
  """
  workable_model_dir: str = ''
  training_job_name: str = ''
  finished_model_dirs: List[str] = dataclasses.field(default_factory=list)


def get_measurement_state_file(measurement_dir):
  """Get measurement state file."""
  return os.path.join(
      measurement_dir,
      variance_measurement_constants.MEASUREMENT_STATE_FILE_NAME)


def save_measurement_state(measurement_dir,
                           measurement_state):
  """Saves ModelSelectionStates data."""
  json_data = dataclasses.asdict(measurement_state)
  file_path = get_measurement_state_file(measurement_dir)
  gcs_utils.save_json(filepath=file_path, json_data=json_data)


def load_or_create_measurement_state_state(
    measurement_dir):
  """Loads ModelSelectionState from file if exists, otherwise creates one."""
  file_path = get_measurement_state_file(measurement_dir)
  measurement_state = MeasurementState()
  json_data = gcs_utils.load_json(filepath=file_path)
  if json_data:
    measurement_state = MeasurementState(**json_data)
  return measurement_state


def _get_trial_training_metrics(
    model_dir):
  """Get TrialTrainingMetrics data from a model dir."""
  metrics_file = proxy_task_utils.get_trial_training_metrics_file(model_dir)
  metrics = proxy_task_utils.read_trial_training_metrics_file(metrics_file)
  if not metrics:
    raise ValueError(
        'No trail training metrics file found in {}'.format(model_dir))
  if not metrics.training_accuracy_over_steps:
    raise ValueError(
        'Empty training_accuracy_over_steps in {}'.format(model_dir))
  final_step_pct = metrics.training_accuracy_over_steps[-1].training_step_pct
  if final_step_pct != 100.0:
    raise ValueError('Desired 100% training but {} found in {}'.format(
        final_step_pct, model_dir))
  return metrics


def _normalize_to_unit(vector):
  """Normalize an array to unit in range [0.0, 1.0]."""
  min_val = np.min(vector)
  max_val = np.max(vector)
  unit_vector = (vector - min_val) / (max_val - min_val)
  return unit_vector


def _calculate_variance(x):
  """Calculate variance using Coefficient of variation."""
  variance = np.std(x) / np.mean(x)
  return variance


def _compute_variance_from_metrics_list(
    metrics_list):
  """Compute variance with a list of TrialTrainingMetrics data."""
  if len(metrics_list) < 2:
    logging.warning('Less than 2 metrics provided, return 0 for variance.')
    return 0.0

  final_accuracies = []
  for metrics in metrics_list:
    # We assume all models are 100% trained when reading metrics from files.
    # So accuracies read here are at the same steps.
    accuracy_at_step = metrics.training_accuracy_over_steps[-1]
    final_accuracies.append(accuracy_at_step.accuracy)
  logging.info('Computing variance of %s', final_accuracies)
  variance = _calculate_variance(final_accuracies)
  return variance


def compute_variance_from_model_dirs(model_dirs):
  """Compute variance with a list of model output directories."""
  metrics_list = [
      _get_trial_training_metrics(model_dir) for model_dir in model_dirs
  ]
  return _compute_variance_from_metrics_list(metrics_list)


def _compute_smoothness_from_metrics(
    metrics):
  """Compute smoothness with a TrialTrainingMetrics."""
  steps = [
      a.training_step_pct for a in metrics.training_accuracy_over_steps
  ]
  scores = [a.accuracy for a in metrics.training_accuracy_over_steps]
  logging.info('Computing smoothness of steps: %s, scores: %s',
               steps, scores)
  # Normalize steps and scores to [0.0, 1.0].
  steps = _normalize_to_unit(np.array(steps, np.float32))
  scores = _normalize_to_unit(np.array(scores, np.float32))

  # Interpolate step-score points with linear spline.
  spline = interpolate.splrep(steps, scores, k=1, s=0)
  x = []
  # Sample points unevenly from spline, points towards the end section are more
  # than former sections.
  for i in range(
      len(variance_measurement_constants.SMOOTHNESS_SAMPLE_SECTIONS) - 1):
    start = variance_measurement_constants.SMOOTHNESS_SAMPLE_SECTIONS[i]
    end = variance_measurement_constants.SMOOTHNESS_SAMPLE_SECTIONS[i + 1]
    x += np.arange(
        start, end, (end - start) /
        variance_measurement_constants.SMOOTHNESS_POINTS_PER_SECTION).tolist()

  # Compute 1st order derivative for each sampled point.
  derivatives = interpolate.splev(x, spline, der=1)
  # Sum all negative derivatives
  negative_derivatives_sum = np.sum(derivatives[derivatives < 0])
  # Use the absolute value as the smoothness.
  smoothness = np.abs(negative_derivatives_sum)
  return smoothness


def _compute_smoothness_from_metrics_list(
    metrics_list):
  """Compute smoothness with a list of TrialTrainingMetrics data."""
  if not metrics_list:
    logging.warning('No metrics provided, return 0 for smoothness.')
    return 0.0

  smoothness_list = [
      _compute_smoothness_from_metrics(metrics) for metrics in metrics_list
  ]
  smoothness = np.mean(smoothness_list)
  return smoothness


def compute_smoothness_from_model_dirs(model_dirs):
  """Compute smoothness with a list of model output directories."""
  metrics_list = [
      _get_trial_training_metrics(model_dir) for model_dir in model_dirs
  ]
  return _compute_smoothness_from_metrics_list(metrics_list)


def find_workable_model(search_job_spec,
                        service_endpoint,
                        region,
                        project_id):
  """Runs a loop to find a workable model."""
  base_job_name = client_utils.get_display_name_for_nas_job(search_job_spec)
  base_job_dir = client_utils.get_job_dir_for_nas_job(search_job_spec)
  # Update num trial to 1.
  search_job_spec = client_utils.set_num_trials_for_nas_job(
      nas_job=search_job_spec,
      max_trial_count=1,
      max_parallel_trial_count=1,
      max_failed_trial_count=1)

  logging.info('Running find workable model loop.')
  for iteration in range(
      1, variance_measurement_constants.MAX_FAILED_MODELS_TO_ABORT + 1):
    logging.info('Launching nas job for iteration-%d.', iteration)

    # Update display job name and output directory.
    nas_job_name = 'Find_workable_model_{}_{}'.format(iteration, base_job_name)
    search_job_spec = client_utils.set_display_name_for_nas_job(
        nas_job=search_job_spec, display_name=nas_job_name)
    dir_name = nas_job_name + datetime.datetime.now().strftime('_%Y%m%d_%H%M%S')
    root_output_dir = client_utils.get_root_output_dir_from_job_dir(
        base_job_dir)
    nas_job_dir = os.path.join(root_output_dir, dir_name, 'nas', 'search')
    search_job_spec = client_utils.set_job_dir_for_nas_job(
        nas_job=search_job_spec, job_dir=nas_job_dir)

    proxy_task_controller_utils.post_controller_alive_time_stamp(
        search_job_spec)
    nas_job_name, _ = client_utils.launch_cloud_nas_job(service_endpoint,
                                                        project_id, region,
                                                        search_job_spec,
                                                        'Nas Job')
    try:
      # Monitor the job until it trains up to 1%.
      # If the training fails due to like OOM, the API will raise an exception
      # when get job status.
      model_selection.monitor_and_stop_trials(
          nas_job_name=nas_job_name,
          service_endpoint=service_endpoint,
          project_id=project_id,
          region=region,
          desired_training_step_pct=1)
    # pylint: disable=broad-except
    except Exception:
      logging.info('Failed model found for iteration-%d', iteration)
    # pylint: enable=broad-except
    else:
      logging.info('Workable model found for iteration-%d', iteration)
      logging.info('The workable model job dir: %s', nas_job_dir)
      return nas_job_dir
  logging.info('No workable model found after trying %d iterations',
               variance_measurement_constants.MAX_FAILED_MODELS_TO_ABORT)
  return ''


def run_model_training(search_job_spec,
                       workable_job_dir,
                       service_endpoint,
                       region,
                       project_id,
                       measurement_state,
                       measurement_dir):
  """Train a workable model with multiple trials."""
  # Update trainer docker args with retrain trial information.
  trainer_args = client_utils.get_docker_args_map_for_nas_job(
      search_job_spec)
  trainer_args['retrain_search_job_dir'] = workable_job_dir
  # The workable job has only 1 trial, so we need to repeat the trial id '1'
  # NUM_TRIALS_FOR_MEASUREMENT times to make it mappable to retrain job trials.
  retrain_search_job_trials = ','.join(
      ['1'] * variance_measurement_constants.NUM_TRIALS_FOR_MEASUREMENT)
  trainer_args['retrain_search_job_trials'] = retrain_search_job_trials
  search_job_spec = client_utils.set_docker_args_map_for_nas_job(
      nas_job=search_job_spec, docker_args_map=trainer_args)

  # Launch a training job.
  logging.info('Training workable model.')
  if not measurement_state.training_job_name:
    proxy_task_controller_utils.post_controller_alive_time_stamp(
        search_job_spec)
    measurement_state.training_job_name, _ = client_utils.launch_cloud_nas_job(
        service_endpoint=service_endpoint,
        project_id=project_id,
        location=region,
        job_spec=search_job_spec,
        job_type='Nas Job')
    save_measurement_state(measurement_dir, measurement_state)

  # Monitor the training job until 100% step.
  logging.info('Monitoring nas job %s', measurement_state.training_job_name)
  model_selection.monitor_and_stop_trials(
      nas_job_name=measurement_state.training_job_name,
      service_endpoint=service_endpoint,
      project_id=project_id,
      region=region,
      desired_training_step_pct=100)

  # Get trial dirs from the finished job.
  total_num_trials, _, _ = client_utils.get_num_trials_for_nas_job(
      search_job_spec)
  trial_dirs = []
  for trial_idx in range(total_num_trials):
    trial_id = str(trial_idx + 1)
    trial_dirs.append(client_utils.get_search_trial_dir(
        job=search_job_spec, trial_id=trial_id))
  return trial_dirs


def generate_measurement_results(variance,
                                 smoothness):
  """Generate measurement info."""
  # pylint: disable=line-too-long
  if variance > variance_measurement_constants.MAX_VARIANCE_TO_BE_GOOD:
    variance_info = 'Model variance is not good, measured:{} > threshold:{}. '.format(
        variance, variance_measurement_constants.MAX_VARIANCE_TO_BE_GOOD)
    variance_info += 'See {}proxy-task-design#variance_measurement'.format(global_variables.NAS_DOC_PATH)
  else:
    variance_info = 'Model variance is good.'

  if smoothness > variance_measurement_constants.MAX_SMOOTHNESS_TO_BE_GOOD:
    smoothness_info = 'Model smoothness is not good, measured:{} > threshold:{}. '.format(
        smoothness, variance_measurement_constants.MAX_SMOOTHNESS_TO_BE_GOOD)
    smoothness_info += 'See {}proxy-task-design#variance_measurement'.format(global_variables.NAS_DOC_PATH)
  else:
    smoothness_info = 'Model smoothness is good.'
  # pylint: enable=line-too-long

  measurement = {
      'variance_value': variance,
      'variance_info': variance_info,
      'smoothness_value': smoothness,
      'smoothness_info': smoothness_info,
  }
  return measurement


def run_measurement(measurement_dir,
                    service_endpoint,
                    region,
                    project_id):
  """Find workable model and measure variance and smoothness."""
  search_job_spec, _ = (
      proxy_task_controller_utils.load_search_and_latency_job_spec(
          measurement_dir
      )
  )
  measurement_state = load_or_create_measurement_state_state(measurement_dir)

  # Run a loop to find a workable model, which won't abort due to OOM etc.
  if not measurement_state.workable_model_dir:
    measurement_state.workable_model_dir = find_workable_model(
        search_job_spec=copy.deepcopy(search_job_spec),
        service_endpoint=service_endpoint,
        region=region,
        project_id=project_id)
    if not measurement_state.workable_model_dir:
      raise ValueError('Could not find a workable model.')
    save_measurement_state(measurement_dir, measurement_state)

  # Start a job to train the same model NUM_TRIALS_FOR_MEASUREMENT times.
  if not measurement_state.finished_model_dirs:
    measurement_state.finished_model_dirs = run_model_training(
        search_job_spec=copy.deepcopy(search_job_spec),
        workable_job_dir=measurement_state.workable_model_dir,
        service_endpoint=service_endpoint,
        region=region,
        project_id=project_id,
        measurement_state=measurement_state,
        measurement_dir=measurement_dir)
    save_measurement_state(measurement_dir, measurement_state)

  # Compute variance and smoothness.
  variance = compute_variance_from_model_dirs(
      measurement_state.finished_model_dirs)
  smoothness = compute_smoothness_from_model_dirs(
      measurement_state.finished_model_dirs)

  # Generate measurement info.
  measurement = generate_measurement_results(variance, smoothness)
  logging.info(measurement)
  measurement_file = os.path.join(
      measurement_dir,
      variance_measurement_constants.MEASUREMENT_RESULT_FILE_NAME)
  gcs_utils.save_json(filepath=measurement_file, json_data=measurement)
  logging.info('Saved measurement results to: %s', measurement_file)
