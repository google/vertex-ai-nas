# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Base class to compute NAS model-metrics using a saved-model."""

import functools
import json
import logging
import os
import re
import subprocess
import time
from typing import Any, Callable, Dict, Iterable, List, Set, Text, Tuple

import cloud_nas_utils
import vertex_client_utils as client_utils
from gcs_utils import gcs_utils_using_cloud_storage as gcs_utils
from proxy_task import proxy_task_controller_utils_constants as controller_constants

_MODEL_LATENCY_FILENAME = "model_latency.json"
# The name of the file under the corresponding model path that stores the id of
# the latency worker that is responsible for the benchmarking.
_LATENCY_WORKER_ID_FILENAME = "latency_worker_id.json"
# How long the `GetJob` call wait for the next one.
_GET_JOB_SLEEP_SECONDS = 60
# The amount of time for which a worker will sleep when it finds an available
# trial before actually undertaking the latency benchmarking.
_AVOID_RACE_SLEEP_SECONDS = 1.
# Retry interval
_RETRY_INTERVAL_SECONDS = 60
_MAX_RETRY = 20
_VERTEX_AI_ENDPOINT_PATTERN = r".*-aiplatform.*\.googleapis\.com"
_VERTEX_AI_VERSION = "v1beta1"


def get_auth():
  return subprocess.check_output(
      ["gcloud", "auth", "application-default",
       "print-access-token"]).strip().decode("utf-8")


def run_command(args):
  """Returns JSON outputs of process running `args`."""

  logging.info("RunCommand: %s", " ".join(args))
  process = subprocess.Popen(
      args, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
  output, error = process.communicate()
  try:
    return json.loads(output)
  except ValueError:
    print("Command <{}>; Unable to parse json in output <{}> and error <{}>"
          .format(" ".join(args), output, error))
    raise


def retry_func(func, max_retry = _MAX_RETRY):
  """Util to retry func.

  Args:
    func: The function that needs to be retry.
    max_retry: Maximum retry of `func` function, default is `10`.

  Returns:
    output of func.

  Raises:
    Exception: if retries exceeded than max_retry.
  """
  for retry in range(1, max_retry + 1):
    try:
      return func()
    except Exception as e:  # pylint: disable=broad-except
      logging.warning(
          "%s", "Failed to call {}, in retry({}/{}\n{})".format(
              func, retry, max_retry, e))
      time.sleep(_RETRY_INTERVAL_SECONDS)
  raise Exception("Failed to call function multiple times.")


def saved_model_exists(saved_model_path):
  """Checks if saved-model exists."""
  logging.info("Checking if SavedModel exists: %s", saved_model_path)
  return gcs_utils.exists(os.path.join(saved_model_path, "saved_model.pb"))


def get_model_metrics_file(saved_model_path):
  """Returns the model-metrics file path."""
  return os.path.join(saved_model_path, _MODEL_LATENCY_FILENAME)


def write_metrics_to_gcs_location(saved_model_path, model_metrics):
  """Writes the model-metrics to the GCS location."""
  model_metrics_file = get_model_metrics_file(saved_model_path)
  logging.info("Writing metrics to %s", model_metrics_file)
  gcs_utils.save_json(filepath=model_metrics_file, json_data=model_metrics)


def is_vertex_ai_endpoint(service_endpoint):
  """Returns if the container uses Vertex AI service."""
  return bool(re.match(_VERTEX_AI_ENDPOINT_PATTERN, service_endpoint))


class ModelMetricsEvaluator(object):
  """Base class to compute latency and memory utilization for a NAS model."""

  def __init__(self,
               service_endpoint,
               project_id,
               nas_job_id,
               latency_worker_id = 0,
               num_latency_workers = 1):
    self.service_endpoint = service_endpoint
    self.project_id = project_id
    # nas_job_id here is nas-job resource name which includes
    # region etc information too.
    # "projects/{project-id}/locations/{region}/nasJobs/{job-id}" or
    # "projects/{project-id}/locations/{region}/customJobs/{job-id}".
    self.nas_job_id = nas_job_id
    self.job_output_dir = ""
    self.is_vertex_ai = is_vertex_ai_endpoint(service_endpoint)
    self.latency_worker_id = latency_worker_id
    self.num_latency_workers = num_latency_workers
    # Check if "nas_job_id" corresponds to a controller job.
    # This will happen ONLY if an on-prem latency calculator job
    # is spawned for a controller job.
    # In this case, use the proxy_task_controller_dir as the source of truth.
    # This directory will not change even if the customer
    # resumes the proxy-task controller job.
    self.proxy_task_controller_dir = self._get_proxy_task_controller_dir(
        parent_job_name=nas_job_id)

  def _get_proxy_task_controller_dir(self, parent_job_name):
    """Returns the proxy-task controller directory if parent job is a controller."""
    # Check if the parent is a proxy-task controller job.
    if not client_utils.is_job_name_for_custom_job(parent_job_name):
      # Parent job is not a custom job.
      return ""
    logging.info(
        "Parent job is a proxy-task controller job: %s.", parent_job_name
    )
    # Get controller job.
    controller_job_id = client_utils.custom_job_id_from_custom_job_name(
        parent_job_name
    )
    logging.info("controller_job_id is %d", controller_job_id)
    controller_job_region = client_utils.custom_job_region_from_custom_job_name(
        parent_job_name
    )
    logging.info("controller_job_region is %s", controller_job_region)
    controller_job = client_utils.get_job(
        vertex_ai_endpoint=self.service_endpoint,
        project_id=self.project_id,
        location=controller_job_region,
        job_id=controller_job_id, job_type="custom"
    )
    controller_args_map = client_utils.get_docker_args_map_for_custom_job(
        controller_job
    )
    if "search_controller_dir" in controller_args_map:
      controller_dir = controller_args_map["search_controller_dir"]
    elif "model_selection_dir" in controller_args_map:
      controller_dir = controller_args_map["model_selection_dir"]
    else:
      raise ValueError(
          "Could not find controller directory: %s" % controller_args_map
      )
    return controller_dir

  def _get_job(self):
    """Returns a cloud job status."""
    if self.is_vertex_ai:
      resource_uri = "https://{endpoint}/{version}/{resource_name}".format(
          endpoint=self.service_endpoint,
          version=_VERTEX_AI_VERSION,
          resource_name=self.nas_job_id,
      )
    else:
      resource_uri = "{endpoint}/projects/{project}/jobs/{job_id}".format(
          endpoint=self.service_endpoint,
          project=self.project_id,
          job_id=self.nas_job_id,
      )
    job = retry_func(
        functools.partial(
            run_command,
            [
                "curl",
                "-X",
                "GET",
                "-v",
                "-H",
                "Authorization: Bearer {}".format(get_auth()),
                "-H",
                "Content-Type: application/json",
                resource_uri,
            ],
        )
    )
    error = job.get("error", None)
    if error:
      print("Job failed: {}".format(error))
      raise Exception("Job failed: {}".format(error))
    else:
      return job

  def _get_trials_from_vertex_ai_nas_job(self, nas_job):
    """Returns exit-status and running NAS-search trials for Vertex AI nas job.
    """
    running_trials = client_utils.get_running_trials(nas_job)
    logging.info("Found running trials: %s", running_trials)
    self.job_output_dir = client_utils.get_job_dir_for_nas_job(nas_job)

    # Figure out exit-condition.
    exit_latency_job = False
    job_state = nas_job["state"]
    if not client_utils.is_nas_job_active(nas_job):
      logging.info("NAS job [%s] is in final state [%s]. Exit early.",
                   self.nas_job_id, job_state)
      exit_latency_job = True

    return exit_latency_job, running_trials

  def _get_trials_from_vertex_ai_controller_job(
      self, proxy_task_controller_dir):
    """Returns exit-status and running NAS-search trials for proxy-task controller job.
    """
    # This function will be called only when on-prem latency calculator
    # is called with the proxy-task controller job-id.
    # The proxy-task controller job will spawn multiple nas-jobs one by one
    # and store the current controller-job-id and current
    # nas-job-id in a json file on GCS controller-dir.
    # The proxy-task controller job may also die in a rare case. In that case,
    # we still want the current running nas-job and this on-prem
    # latency calculator job to keep running till they finish.
    # The customer may resume the dead proxy-task controller job by
    # reusing the previous controller directory. Therefore, the source of
    # truth for latency calculator job should always be the
    # controller directory.
    # Here is the workflow:
    # 1. Load current nas-job-id from controller-dir.
    # 2. If has current nas-job-id then process it if active EVEN if
    #    the controller job has died. This will prevent premature failure.
    # 3. If current nas-job-id is not active then check if controller job is
    #    active.
    # 4. If the controller job is not active, then exit.

    # Find current child nas-job from the proxy-task controller directory.
    nas_job_name_filepath = os.path.join(
        proxy_task_controller_dir,
        controller_constants.CONTROLLER_CURRENT_NAS_JOB_NAME_FILE,
    )
    if gcs_utils.exists(nas_job_name_filepath):
      nas_job_name = gcs_utils.load_json(nas_job_name_filepath)[
          controller_constants.CONTROLLER_CURRENT_NAS_JOB_NAME_KEY
      ]
    else:
      nas_job_name = ""

    # Process current child nas-job.
    if nas_job_name:
      # Get current nas-job.
      nas_job_id = client_utils.nas_job_id_from_nas_job_name(nas_job_name)
      logging.info("nas_job_id is %d", nas_job_id)
      nas_job_region = client_utils.nas_job_region_from_nas_job_name(
          nas_job_name
      )
      logging.info("nas_job_region is %s", nas_job_region)
      nas_job = client_utils.get_job(
          vertex_ai_endpoint=self.service_endpoint,
          project_id=self.project_id,
          location=nas_job_region,
          job_id=nas_job_id,
      )
      # Process current nas job just like a regular nas-job.
      exit_latency_job, running_trials = (
          self._get_trials_from_vertex_ai_nas_job(nas_job)
      )
      # Do not exit yet if nas-job has finished.
      # We also need to wait for parent controller job.
      if not exit_latency_job:
        return exit_latency_job, running_trials

    # We will reach here if (a) there is no child nas_job_name or (b)
    # child nas-job has exited. So now check if CURRENT controller
    # job is still active. If controller job is not alive then exit.
    # Find current child nas-job from the proxy-task controller directory.
    controller_job_name_filepath = os.path.join(
        proxy_task_controller_dir,
        controller_constants.CONTROLLER_JOB_NAME_FILE,
    )
    if gcs_utils.exists(controller_job_name_filepath):
      controller_job_name = gcs_utils.load_json(controller_job_name_filepath)[
          controller_constants.CONTROLLER_JOB_NAME_KEY
      ]
    else:
      raise Exception("Could not find %s" % controller_job_name_filepath)
    controller_job_id = client_utils.custom_job_id_from_custom_job_name(
        controller_job_name
    )
    logging.info("controller_job_id is %d", controller_job_id)
    controller_job_region = client_utils.custom_job_region_from_custom_job_name(
        controller_job_name
    )
    logging.info("controller_job_region is %s", controller_job_region)
    controller_job = client_utils.get_job(
        vertex_ai_endpoint=self.service_endpoint,
        project_id=self.project_id,
        location=controller_job_region,
        job_id=controller_job_id, job_type="custom"
    )
    # Is controller job still running?
    running_trials = set()
    exit_latency_job = False
    job_state = controller_job["state"]
    logging.info("Controller job state is %s", job_state)
    if not client_utils.is_nas_job_active(controller_job):
      logging.info(
          "Controller job [%s] is in final state [%s]. Exit early.",
          self.nas_job_id,
          job_state,
      )
      exit_latency_job = True
    return exit_latency_job, running_trials

  def _get_trials_from_caip(self, job_output):
    """Returns exit-status and running NAS-search trials for CAIP."""
    running_trials = set()
    multi_trial_job_outputs = (
        job_output.get("trainingOutput", {})
        .get("nasJobOutput", {})
        .get("multiTrialJobOutputs", {})
    )
    if "multiTrialJobOutput" in multi_trial_job_outputs:
      for trial in multi_trial_job_outputs["multiTrialJobOutput"]:
        if trial["state"] == "RUNNING":
          running_trials.add(trial["trialId"])
    job_state = job_output["state"]
    self.job_output_dir = job_output["trainingInput"]["jobDir"]
    exit_latency_job = False
    if job_state in ("FAILED", "CANCELLED", "SUCCEEDED"):
      logging.info("NAS job [%s] is in final state [%s], exit early.",
                   self.nas_job_id, job_state)
      exit_latency_job = True
      return exit_latency_job, running_trials
    elif job_state in ("RUNNING") and running_trials:
      logging.info("Found running trials: %s", running_trials)
      return exit_latency_job, running_trials
    else:
      logging.info("Could not find running trials for job: %s", self.nas_job_id)
      return exit_latency_job, running_trials

  def get_running_trials(self):
    """Returns exit-status and NAS-search trials in RUNNING state."""
    # Is it proxy-task controller use case?
    if self.proxy_task_controller_dir:
      return self._get_trials_from_vertex_ai_controller_job(
          self.proxy_task_controller_dir
      )
    # Normal use case.
    job_output = self._get_job()
    if self.is_vertex_ai:
      return self._get_trials_from_vertex_ai_nas_job(job_output)
    else:
      return self._get_trials_from_caip(job_output)

  def get_saved_model_path(self, trial_id):
    """Returns the saved-model path for the trial."""
    return cloud_nas_utils.get_saved_model_dir(
        trial_job_dir=os.path.join(self.job_output_dir, str(trial_id)))

  def find_saved_model_to_process_for_trial(self, trial_id):
    """Returns a saved-model-path to be processed for a trial.

    Args:
      trial_id: The NAS-search trial-id to process.

    Returns:
      A saved-model-path if it exists and needs to be processed. Otherwise
      an empty string is returned.
    """
    saved_model_path = self.get_saved_model_path(trial_id)
    # Check if the training is done and the saved-model has been
    # generated by the trial already.
    if saved_model_exists(saved_model_path):
      logging.info("Found SavedModel at %s", saved_model_path)
      model_metrics_file = get_model_metrics_file(saved_model_path)
    else:
      logging.info("SavedModel not found yet at %s", saved_model_path)
      return ""
    # Check if this saved-model has been processed already.
    if gcs_utils.exists(model_metrics_file):
      logging.info("Skip since model_metrics_file already exits: %s",
                   model_metrics_file)
      return ""
    else:
      return saved_model_path

  def evaluate_saved_model(self, trial_id, saved_model_path):
    """Derived class should override this method to return computed metrics."""
    # NOTE: This function should be overridden. This is a dummy code.
    del trial_id
    del saved_model_path
    logging.info("Job output directory is %s", self.job_output_dir)
    return {"latency_in_seconds": 0.0, "model_memory": 0}

  def should_process_this_trial(self, trial_id):
    """Check if the latency worker should process this trial-id.

    Args:
      trial_id: The id of the trial that might need processing.

    Returns:
      A saved-model-path if it exists and needs to be processed. Otherwise
      an empty string is returned.
    """

    ########## Worker Pool Scheduling Policy #########
    saved_model_path = self.find_saved_model_to_process_for_trial(trial_id)
    if not saved_model_path:
      # find_saved_model_to_process_for_trial will return an empty string when
      # SavedModel not found or when model_metrics_file already exits.
      return ""

    # If the "latency_worker_id.json" file does not exist, then this worker will
    # create such a file and write its own worker-id inside the file to assign
    # itself as the worker for this trial.
    latency_worker_id_filepath = os.path.join(saved_model_path,
                                              _LATENCY_WORKER_ID_FILENAME)
    if not gcs_utils.exists(latency_worker_id_filepath):
      gcs_utils.save_json(
          filepath=latency_worker_id_filepath,
          json_data={"latency_worker_id": self.latency_worker_id})
      # There is no file lock for the GCS filesystem. In theory, another worker
      # may also create "latency_worker_id.json" at almost the same time
      # overriding the previous worker. The solution for this race condition is
      # to sleep and re-read that file to figure out who is the final assigned
      # latency worker.
      time.sleep(_AVOID_RACE_SLEEP_SECONDS)
    # Even if the file exists, this worker may still be the owner of the trial
    # because of possible restart in the past.
    owner_worker_id = gcs_utils.load_json(
        latency_worker_id_filepath)["latency_worker_id"]
    # After sleeping, if the current worker still remains the assigned worker
    # for this trial, then proceed with latency computation. Else give up this
    # trial and move to the next candidate.
    if owner_worker_id != self.latency_worker_id:
      return ""
    else:
      return saved_model_path

  def run_continuous_evaluation_loop(self):
    """Runs continuous evaluation loop for a NAS search-job."""
    # We are running this job as a monitor with the NAS job. It periodically
    # checks the pending trial, and calculates the metrics (such as latency)
    # for the saved model of each trial. The metrics are then written
    # to a file in the GCS location monitored by the trial job.

    while True:
      # Sleep for `_GET_JOB_SLEEP_SECONDS` seconds every iteration.
      time.sleep(_GET_JOB_SLEEP_SECONDS)
      exit_latency_job, running_trials = self.get_running_trials()
      if exit_latency_job:
        logging.info("Exiting latency job.")
        break
      for trial_id_str in running_trials:
        trial_id = int(trial_id_str)
        saved_model_path = self.should_process_this_trial(trial_id)
        if not saved_model_path:
          continue
        try:
          model_metrics = self.evaluate_saved_model(trial_id, saved_model_path)
        except Exception as e:  # pylint: disable=broad-except
          # Do not fail this job because of one trial failure. If there are
          # many trial failures, the search job will be failed after
          # `maxFailedNasTrials`.
          model_metrics = None
          logging.error("Got failure for trial %d: %s", trial_id, str(e))
        if model_metrics:
          write_metrics_to_gcs_location(saved_model_path, model_metrics)
