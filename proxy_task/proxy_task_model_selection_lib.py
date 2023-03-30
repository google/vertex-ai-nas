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
"""Library functions for proxy-task model selection.

The model-selection consists of following steps:

- Launch an initial nas job.
- Monitor the nas job active trials for the number of steps they finish.
  and send a stop signal when the desired number of steps is done.
- When all trials have stopped, then process the trial scores.
  - If a large percentage of trials show OOM or other error, then report an
    error.
  - Reject 5 models from the pool based on scores. Note: compare scores at the
    same training step.
- Launch another nas retrain job with 5 less models and ensure that
  the previous checkpoints are available to resume training.
- Loop until only the desired number of models remain.
"""

import copy
import dataclasses
import datetime
import logging
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

import cloud_nas_utils
import vertex_client_utils as client_utils
from gcs_utils import gcs_utils_using_cloud_storage as gcs_utils
from gcs_utils import gcs_utils_using_gcsfuse
from proxy_task import proxy_task_controller_utils
from proxy_task import proxy_task_controller_utils_constants
from proxy_task import proxy_task_model_selection_lib_constants as model_selection_constants
from proxy_task import proxy_task_utils
import matplotlib.pyplot as plt
import numpy as np


@dataclasses.dataclass
class ModelSelectionIterationState:
  """Class for model-selection iteration state.

  Attributes:
    num_trials: Number of trials for this nas job.
    trials_to_retrain: A list of trials selected for next iteration.
    search_job_name: Search job name.
    search_job_link: Cloud console link for the search job.
    latency_calculator_job_name: Latency-calculator job name.
    latency_calculator_job_link: Cloud console link for the latency-calculator
      job.
    desired_training_step_pct: Desired training step percentage.
  """

  num_trials: int = 0
  trials_to_retrain: List[str] = dataclasses.field(default_factory=list)
  search_job_name: str = ""
  search_job_link: str = ""
  latency_calculator_job_name: str = ""
  latency_calculator_job_link: str = ""
  desired_training_step_pct: float = 0.0


def get_model_selection_iteration_state_from_dict(
    dict_obj
):
  """Loads ModelSelectionState from a dictionary."""
  return ModelSelectionIterationState(
      num_trials=dict_obj["num_trials"],
      trials_to_retrain=dict_obj["trials_to_retrain"],
      search_job_name=dict_obj["search_job_name"],
      search_job_link=dict_obj["search_job_link"],
      latency_calculator_job_name=dict_obj["latency_calculator_job_name"],
      latency_calculator_job_link=dict_obj["latency_calculator_job_link"],
      desired_training_step_pct=dict_obj["desired_training_step_pct"],
  )


@dataclasses.dataclass
class ModelSelectionState:
  """Class for model selection state.

  Attributes:
    start_num_models: Number of models to begin with.
    final_num_models: Number of models to end with.
    num_models_to_remove_per_iter: Number of models to remove per iteration.
    iterations: A list of model-selection iterations done so far.
    accuracy_metric_id: The accuracy metric-id based on which the models will be
      selected.
    latency_metric_id: The latency metric-id based on which the models will be
      selected. This is not required if the training job does not compute
      latency.
  """

  start_num_models: int = model_selection_constants.START_NUM_MODELS
  final_num_models: int = model_selection_constants.FINAL_NUM_MODELS
  num_models_to_remove_per_iter: int = (
      model_selection_constants.NUM_MODELS_TO_REMOVE_PER_ITER
  )
  iterations: List[ModelSelectionIterationState] = dataclasses.field(
      default_factory=list
  )
  accuracy_metric_id: str = ""
  latency_metric_id: str = ""


def get_model_selection_state_from_dict(
    dict_obj
):
  """Loads ModelSelectionState from a dictionary."""
  return ModelSelectionState(
      start_num_models=dict_obj["start_num_models"],
      final_num_models=dict_obj["final_num_models"],
      num_models_to_remove_per_iter=dict_obj["num_models_to_remove_per_iter"],
      iterations=[
          get_model_selection_iteration_state_from_dict(val)
          for val in dict_obj["iterations"]
      ],
      accuracy_metric_id=dict_obj["accuracy_metric_id"],
      latency_metric_id=dict_obj["latency_metric_id"],
  )


def get_model_selection_state_file(model_selection_dir):
  return os.path.join(
      model_selection_dir,
      model_selection_constants.MODEL_SELECTION_STATE_FILENAME,
  )


def save_model_selection_state_file(
    filepath, model_selection_state
):
  """Saves ModelSelectionStates data.

  Args:
    filepath: The file path of ModelSelectionState data.
    model_selection_state: ModelSelectionState data.
  """
  json_data = dataclasses.asdict(model_selection_state)
  gcs_utils.save_json(filepath=filepath, json_data=json_data)


def try_load_model_selection_state(
    model_selection_dir,
):
  """Returns model selection state and iteration by loading a previously stored file if it exists."""
  filepath = get_model_selection_state_file(model_selection_dir)
  model_selection_state = ModelSelectionState()
  iteration = 0
  json_data = gcs_utils.load_json(filepath)
  if json_data:
    model_selection_state = get_model_selection_state_from_dict(json_data)
    iteration = len(model_selection_state.iterations) - 1
  return model_selection_state, iteration


def monitor_and_stop_trials(
    nas_job_name,
    service_endpoint,
    project_id,
    region,
    desired_training_step_pct,
):
  """Monitors the nas job progress and stops trials once they reach desired steps."""
  logging.info("Starting monitoring of trials.")
  logging.info("desired_training_step_pct is %f.", desired_training_step_pct)

  # Post desired-training-step-pct to all the trials.
  # NOTE: It does not matter if the trials have not started yet.
  # The message will be posted to the trial-directories and they will read it
  # later and stop accordingly.
  nas_job_id = client_utils.nas_job_id_from_nas_job_name(nas_job_name)
  logging.info("nas_job_id is %s", nas_job_id)
  nas_job = client_utils.get_job(
      vertex_ai_endpoint=service_endpoint,
      project_id=project_id,
      location=region,
      job_id=nas_job_id,
  )
  proxy_task_controller_utils.set_stop_training_for_all_trials(
      nas_job=nas_job, desired_training_step_pct=desired_training_step_pct
  )

  # Start monitoring nas-job till it stops.
  while True:
    # Check if nas-job is active.
    nas_job = client_utils.get_job(
        vertex_ai_endpoint=service_endpoint,
        project_id=project_id,
        location=region,
        job_id=nas_job_id,
    )
    logging.info("Got nas job info.")
    proxy_task_controller_utils.post_controller_alive_time_stamp(nas_job)
    logging.info("Posted controller alive timestamp.")
    if not client_utils.is_nas_job_active(nas_job):
      logging.info(
          "Stopping monitoring for nas job because it is no longer active."
      )
      break

    # sleep
    logging.info(
        "Sleeping for %d secs",
        proxy_task_controller_utils_constants.TRIAL_MONITOR_SLEEP_SECS,
    )
    time.sleep(proxy_task_controller_utils_constants.TRIAL_MONITOR_SLEEP_SECS)


def normalize_trial_scores(trial_scores):
  """Normalizes trial scores between 0 and 1."""
  for score_key in (
      model_selection_constants.ACCURACY,
      model_selection_constants.LATENCY,
  ):
    min_val = np.min(trial_scores[score_key])
    max_val = np.max(trial_scores[score_key])
    trial_scores[score_key] = (trial_scores[score_key] - min_val) / (
        max_val - min_val + np.finfo(np.float32).eps
    )
  return trial_scores


def find_trial_to_remove(normalized_trial_scores):
  """Finds the trial-idx which can be removed and still preserve a good score-distribution."""
  number_of_trials = normalized_trial_scores.shape[0]
  if number_of_trials <= 2:
    return None

  # Find first closest and second-closest distance for each trial-score.
  # The numpy array below will store in each row:
  # (first_closest_distance, first_closest_trial, second_closest_distance)
  trial_distances = np.zeros((number_of_trials, 3))
  first_closest_distance_col_idx = 0
  first_closest_trial_col_idx = 1
  second_closest_distance_col_idx = 2

  for trial_idx in range(number_of_trials):
    first_closest_distance = model_selection_constants.MAX_TRIAL_SCORE_DISTANCE
    first_closest_trial_idx = -1
    second_closest_distance = model_selection_constants.MAX_TRIAL_SCORE_DISTANCE
    point1 = np.array([
        normalized_trial_scores[trial_idx][model_selection_constants.ACCURACY],
        normalized_trial_scores[trial_idx][model_selection_constants.LATENCY],
    ])
    for neighbor_trial_idx in range(number_of_trials):
      if neighbor_trial_idx == trial_idx:
        continue
      point2 = np.array([
          normalized_trial_scores[neighbor_trial_idx][
              model_selection_constants.ACCURACY
          ],
          normalized_trial_scores[neighbor_trial_idx][
              model_selection_constants.LATENCY
          ],
      ])
      distance_to_point1 = np.linalg.norm(point1 - point2)
      if distance_to_point1 < first_closest_distance:
        second_closest_distance = first_closest_distance
        first_closest_distance = distance_to_point1
        first_closest_trial_idx = neighbor_trial_idx
      elif distance_to_point1 < second_closest_distance:
        second_closest_distance = distance_to_point1
    # Done computing distances to all other points for point1.
    trial_distances[trial_idx, first_closest_trial_col_idx] = (
        first_closest_trial_idx
    )
    trial_distances[trial_idx, first_closest_distance_col_idx] = (
        first_closest_distance
    )
    trial_distances[trial_idx, second_closest_distance_col_idx] = (
        second_closest_distance
    )

  # Done computing all pair wise distances.
  # Find two closest trial-idxs.
  trial_idx1 = np.argmin(trial_distances[:, first_closest_distance_col_idx])
  trial_idx2 = int(trial_distances[trial_idx1, first_closest_trial_col_idx])

  # Remove the one which has lower second-closest distance.
  if (
      trial_distances[trial_idx1, second_closest_distance_col_idx]
      < trial_distances[trial_idx2, second_closest_distance_col_idx]
  ):
    return trial_idx1
  else:
    return trial_idx2


def save_trial_scores_plot(
    plot_filename, trial_scores, filtered_trials
):
  """Saves a plot for the filtered trial scores."""
  with gcs_utils_using_gcsfuse.file_open(plot_filename, "wb") as f:
    plt.figure()
    # Plot filtered scores as blue circle and
    # removed ones as red cross.
    for trial_idx in range(trial_scores.shape[0]):
      trial_id = trial_scores[trial_idx][model_selection_constants.TRIAL_ID]
      if trial_id in filtered_trials:
        plt.plot(
            trial_scores[trial_idx][model_selection_constants.ACCURACY],
            trial_scores[trial_idx][model_selection_constants.LATENCY],
            "bo",
        )
      else:
        plt.plot(
            trial_scores[trial_idx][model_selection_constants.ACCURACY],
            trial_scores[trial_idx][model_selection_constants.LATENCY],
            "rx",
        )
    plt.title("Filtered trial scores")
    plt.xlabel("Accuracy")
    plt.ylabel("Latency")
    plt.savefig(f)
    plt.close()


def filter_trials_based_on_scores(
    nas_job_name,
    service_endpoint,
    project_id,
    region,
    desired_training_step_pct,
    num_models_to_remove,
    requires_latency,
):
  """Returns a subset of trials based on scores to be used for next model selection iteration."""
  logging.info("Starting filtering of trials based on scores.")
  # Get nas job.
  nas_job_id = client_utils.nas_job_id_from_nas_job_name(nas_job_name)
  logging.info("Fetching job for nas-job-id %s", nas_job_id)
  nas_job = client_utils.get_job(
      vertex_ai_endpoint=service_endpoint,
      project_id=project_id,
      location=region,
      job_id=nas_job_id,
  )
  # Gather all trial-training-profiles for the nas-job.
  total_trials, _, _ = client_utils.get_num_trials_for_nas_job(nas_job)
  logging.info("Total trials are %d", total_trials)
  trial_scores = np.recarray(
      (total_trials,),
      dtype=[
          (model_selection_constants.TRIAL_ID, object),
          (model_selection_constants.ACCURACY, float),
          (model_selection_constants.LATENCY, float),
      ],
  )
  for trial_idx in range(total_trials):
    trial_id = str(trial_idx + 1)
    # Initialize the score to an invalid value.
    trial_scores[trial_idx] = (
        model_selection_constants.INVALID_TRIAL_ID,
        0.0,
        0.0,
    )
    logging.info("Fetching scores for trial-id %s", trial_id)
    trial_training_metrics = (
        proxy_task_controller_utils.load_trial_training_metrics(
            trial_id=trial_id, nas_job=nas_job
        )
    )
    if not trial_training_metrics:
      logging.warning("Could not fetch scores for trial-id %s", trial_id)
      continue
    # Check for valid latency.
    if requires_latency and trial_training_metrics.latency < 0.0:
      logging.info(
          "Required latency is not available for trial id %s", trial_id
      )
      continue
    # Check for valid accuracy.
    accuracy = proxy_task_utils.get_accuracy_at_step(
        trial_training_metrics=trial_training_metrics,
        desired_training_step_pct=desired_training_step_pct,
    )
    if not accuracy:
      logging.info(
          "Required accuracy is not available for trial id %s", trial_id
      )
      continue
    # We have both accuracy and latency.
    logging.info(
        (
            "Accuracy for trial id %s at training-step-pct %f "
            "is %f and latency is %f"
        ),
        trial_id,
        desired_training_step_pct,
        accuracy,
        trial_training_metrics.latency,
    )
    trial_scores[trial_idx] = (
        trial_id,
        accuracy,
        trial_training_metrics.latency,
    )

  # Remove invalid trials.
  logging.info("Trial scores are %s", trial_scores)
  trial_scores = trial_scores[
      trial_scores[model_selection_constants.TRIAL_ID]
      != model_selection_constants.INVALID_TRIAL_ID
  ]
  logging.info(
      "Trial scores after removing invalid trials are %s", trial_scores
  )
  num_invalid_trials = total_trials - trial_scores.shape[0]
  logging.info("Num invalid trials are %d", num_invalid_trials)
  filtered_trials = set(trial_scores[model_selection_constants.TRIAL_ID])
  num_models_to_remove = max(0, num_models_to_remove - num_invalid_trials)

  # Remove trials one by one based on their scores.
  normalized_trial_scores = normalize_trial_scores(trial_scores.copy())
  logging.info(
      "Candidate normalized trial scores before filtering are %s",
      normalized_trial_scores,
  )
  # We need at least _MIN_NUM_FILTERED_TRIALS models to set a
  # min and max limit on set of trials.
  # The 'num_models_to_remove' should not result in final number of models
  # being less than _MIN_NUM_FILTERED_TRIALS.
  num_models_to_remove = max(
      min(
          num_models_to_remove,
          normalized_trial_scores.shape[0]
          - model_selection_constants.MIN_NUM_FILTERED_TRIALS,
      ),
      0,
  )
  logging.info("About to remove %d models", num_models_to_remove)
  for _ in range(num_models_to_remove):
    remove_trial_idx = find_trial_to_remove(normalized_trial_scores)
    remove_trial_id = normalized_trial_scores[remove_trial_idx][
        model_selection_constants.TRIAL_ID
    ]
    filtered_trials.remove(remove_trial_id)
    logging.info("Removed trial id %s", remove_trial_id)
    # Remove the trial before the next iteration.
    normalized_trial_scores = np.delete(
        normalized_trial_scores, (remove_trial_idx)
    )

  # Save filtered trials plot.
  plot_filename = os.path.join(
      client_utils.get_job_dir_for_nas_job(nas_job),
      model_selection_constants.FILTERED_TRIAL_SCORES_PLOT_FILENAME,
  )
  save_trial_scores_plot(
      plot_filename=plot_filename,
      trial_scores=trial_scores,
      filtered_trials=filtered_trials,
  )
  return filtered_trials


def launch_nas_retrain_job(
    previous_nas_job_name,
    search_job_spec,
    latency_calculator_job_spec,
    service_endpoint,
    region,
    project_id,
    previous_trials_to_retrain,
    iteration,
):
  """Launches a new retrain job with a subset of trials."""
  # Get job directory for previous iteration nas-job.
  previous_nas_job_id = client_utils.nas_job_id_from_nas_job_name(
      previous_nas_job_name
  )
  prev_nas_job = client_utils.get_job(
      vertex_ai_endpoint=service_endpoint,
      project_id=project_id,
      location=region,
      job_id=previous_nas_job_id,
  )
  prev_nas_job_dir = client_utils.get_job_dir_for_nas_job(prev_nas_job)

  # Get new nas job name and directory based on current iteration number.
  base_job_name = client_utils.get_display_name_for_nas_job(search_job_spec)
  job_name = base_job_name + "_iter_{}".format(iteration)
  dir_name = job_name + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
  root_output_dir = client_utils.get_root_output_dir_from_job_dir(
      prev_nas_job_dir
  )
  new_job_dir = os.path.join(root_output_dir, dir_name, "nas", "search")

  # Copy previous trial training metrics to
  # corresponding new trial folders.
  retrain_search_job_trials = ",".join(previous_trials_to_retrain)
  map_finetune_trial_id_to_search_trial_id = (
      cloud_nas_utils.get_map_finetune_trial_id_to_search_trial_id(
          retrain_search_job_trials
      )
  )
  for finetune_trial_idx in range(len(previous_trials_to_retrain)):
    finetune_trial_id = str(finetune_trial_idx + 1)
    search_trial_id = map_finetune_trial_id_to_search_trial_id[
        finetune_trial_id
    ]
    prev_model_dir = os.path.join(prev_nas_job_dir, search_trial_id)
    curr_model_dir = os.path.join(new_job_dir, finetune_trial_id)
    proxy_task_utils.copy_trial_training_metrics(
        prev_model_dir=prev_model_dir, curr_model_dir=curr_model_dir
    )
  proxy_task_utils.copy_trial_training_metrics(
      prev_model_dir=prev_model_dir, curr_model_dir=curr_model_dir
  )

  ############ Update nas job spec for next iteration. ##########
  # Update display name and job directory.
  search_job_spec = client_utils.set_display_name_for_nas_job(
      nas_job=search_job_spec, display_name=job_name
  )
  search_job_spec = client_utils.set_job_dir_for_nas_job(
      nas_job=search_job_spec, job_dir=new_job_dir
  )
  # Update search docker flags with retrain trial information.
  search_job_args_map = client_utils.get_docker_args_map_for_nas_job(
      search_job_spec
  )
  search_job_args_map["retrain_search_job_dir"] = prev_nas_job_dir
  search_job_args_map["retrain_search_job_trials"] = retrain_search_job_trials
  search_job_args_map["retrain_use_search_job_checkpoint"] = "True"
  search_job_spec = client_utils.set_docker_args_map_for_nas_job(
      nas_job=search_job_spec, docker_args_map=search_job_args_map
  )
  # Update number of trials.
  _, prev_max_parallel_trial_count, _ = client_utils.get_num_trials_for_nas_job(
      search_job_spec
  )
  search_job_spec = client_utils.set_num_trials_for_nas_job(
      nas_job=search_job_spec,
      max_trial_count=len(previous_trials_to_retrain),
      max_parallel_trial_count=min(
          len(previous_trials_to_retrain), prev_max_parallel_trial_count
      ),
      max_failed_trial_count=min(
          len(previous_trials_to_retrain),
          model_selection_constants.MAX_ALLOWED_FAILURES,
      ),
  )

  # Launch the new job.
  proxy_task_controller_utils.post_controller_alive_time_stamp(search_job_spec)
  return client_utils.launch_cloud_nas_and_latency_job(
      search_job_spec=search_job_spec,
      latency_calculator_job_spec=latency_calculator_job_spec,
      service_endpoint=service_endpoint,
      region=region,
      project_id=project_id,
  )


def _get_num_total_trials_for_nas_job_name(
    nas_job_name, service_endpoint, region, project_id
):
  """Returns the total num trials given a nas-job name."""
  nas_job_id = client_utils.nas_job_id_from_nas_job_name(nas_job_name)
  nas_job = client_utils.get_job(
      vertex_ai_endpoint=service_endpoint,
      project_id=project_id,
      location=region,
      job_id=nas_job_id,
  )
  total_num_trials, _, _ = client_utils.get_num_trials_for_nas_job(nas_job)
  logging.info(
      "Total num trials for nas job %s is %d", nas_job_name, total_num_trials
  )
  return total_num_trials


def run_model_selection_loop(
    model_selection_dir,
    service_endpoint,
    region,
    project_id,
    accuracy_metric_id,
    latency_metric_id,
):
  """Runs model selection loop to narrow down to the desired number of models."""
  search_job_spec, latency_calculator_job_spec = (
      proxy_task_controller_utils.load_search_and_latency_job_spec(
          model_selection_dir
      )
  )
  requires_latency = latency_calculator_job_spec is not None

  # Check if the job has restarted again.
  model_selection_state, iteration = try_load_model_selection_state(
      model_selection_dir
  )
  model_selection_state.accuracy_metric_id = accuracy_metric_id
  model_selection_state.latency_metric_id = latency_metric_id
  filtered_trials = set()
  search_job_name = ""
  desired_training_step_pct = 0
  skip_nas_job_launch = False
  if model_selection_state.iterations:
    skip_nas_job_launch = True
    logging.warning("Model selection job has restarted.")
    logging.info("Model selection state is %s.", model_selection_state)

  # Start iteration loop:
  # 1. Launch search job.
  # 2. Check if we need to reduce number of models more.
  # 3. Monitor and wait till search job trains for desired steps.
  # 4. Reduce number of models and loop to step 1.
  while True:
    # NOTE: If model-selection has restarted, then we directly go
    # to monitoring a previously launched nas-job instead of launching one
    # right after a restart.
    if not skip_nas_job_launch:
      # Launch search job for this iteration.
      if iteration == 0:
        # Launch first nas job.
        proxy_task_controller_utils.post_controller_alive_time_stamp(
            search_job_spec
        )
        (
            search_job_name,
            search_job_link,
            latency_calculator_job_name,
            latency_calculator_job_link,
        ) = client_utils.launch_cloud_nas_and_latency_job(
            search_job_spec=search_job_spec,
            latency_calculator_job_spec=latency_calculator_job_spec,
            service_endpoint=service_endpoint,
            region=region,
            project_id=project_id,
        )
      else:
        # Launch job for this iteration.
        logging.info("=======================================")
        logging.info("Launching the current iteration of nas job.")
        (
            search_job_name,
            search_job_link,
            latency_calculator_job_name,
            latency_calculator_job_link,
        ) = launch_nas_retrain_job(
            previous_nas_job_name=search_job_name,
            search_job_spec=copy.deepcopy(search_job_spec),
            latency_calculator_job_spec=copy.deepcopy(
                latency_calculator_job_spec
            ),
            service_endpoint=service_endpoint,
            region=region,
            project_id=project_id,
            previous_trials_to_retrain=filtered_trials,
            iteration=iteration,
        )

      # Check if this is last iteration.
      current_num_trials = _get_num_total_trials_for_nas_job_name(
          nas_job_name=search_job_name,
          service_endpoint=service_endpoint,
          region=region,
          project_id=project_id,
      )
      if current_num_trials <= model_selection_constants.FINAL_NUM_MODELS:
        logging.info(
            "Found the desired number of final models: %d.", current_num_trials
        )
        # NOTE: The last training iteration will run till completion:
        # 100% training because the 'desired_training_step_pct' below
        # will not limit it.
        desired_training_step_pct = (
            proxy_task_controller_utils_constants.TRAINING_STEP_PCT_TO_COMPLETE_TRAINING
        )
      else:
        # Train K models for (1/K)th total training time.
        desired_training_step_pct += (
            1.0 / current_num_trials
        ) * proxy_task_controller_utils_constants.MAX_TRAINING_STEP_PCT

      # Save model-selection state so far.
      proxy_task_controller_utils.save_current_nas_job_name(
          nas_job_name=search_job_name, controller_dir=model_selection_dir
      )
      iter_state = ModelSelectionIterationState(
          # NOTE: We do not have trials to retrain yet.
          num_trials=current_num_trials,
          trials_to_retrain=[],
          search_job_name=search_job_name,
          search_job_link=search_job_link,
          latency_calculator_job_name=latency_calculator_job_name,
          latency_calculator_job_link=latency_calculator_job_link,
          desired_training_step_pct=desired_training_step_pct,
      )
      model_selection_state.iterations.append(iter_state)
      save_model_selection_state_file(
          filepath=get_model_selection_state_file(model_selection_dir),
          model_selection_state=model_selection_state,
      )

    # Monitor and wait till search job trains for desired steps.
    skip_nas_job_launch = False
    search_job_name = model_selection_state.iterations[-1].search_job_name
    desired_training_step_pct = model_selection_state.iterations[
        -1
    ].desired_training_step_pct
    current_num_trials = model_selection_state.iterations[-1].num_trials
    monitor_and_stop_trials(
        nas_job_name=search_job_name,
        service_endpoint=service_endpoint,
        project_id=project_id,
        region=region,
        desired_training_step_pct=desired_training_step_pct,
    )

    # Exit if this is the last iteration.
    # Max training step-pct.
    if (
        desired_training_step_pct
        == proxy_task_controller_utils_constants.TRAINING_STEP_PCT_TO_COMPLETE_TRAINING
    ):
      logging.info("Exiting the loop.")
      break

    # Reduce number of models for next iteration.
    # Do not remove excess models:
    # We do not want to go below 'final_num_models'.
    num_models_to_remove = min(
        model_selection_constants.NUM_MODELS_TO_REMOVE_PER_ITER,
        current_num_trials - model_selection_constants.FINAL_NUM_MODELS,
    )
    filtered_trials = filter_trials_based_on_scores(
        nas_job_name=search_job_name,
        service_endpoint=service_endpoint,
        project_id=project_id,
        region=region,
        desired_training_step_pct=desired_training_step_pct,
        num_models_to_remove=num_models_to_remove,
        requires_latency=requires_latency,
    )
    logging.info(
        "Filtered trials to be processed for next round are: %s",
        filtered_trials,
    )

    # Update current model-iteration state to save the
    # filtered trials for next round.
    model_selection_state.iterations[-1].trials_to_retrain = list(
        filtered_trials
    )
    save_model_selection_state_file(
        filepath=get_model_selection_state_file(model_selection_dir),
        model_selection_state=model_selection_state,
    )
    iteration += 1
