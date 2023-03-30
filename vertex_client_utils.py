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
"""Cloud NAS client utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import importlib
import io
import json
import logging
import os
import subprocess
import sys
import time

from typing import Any, Callable, Dict, Mapping, Optional, Set, Tuple, Union

from gcs_utils import gcs_utils_using_cloud_storage as gcs_utils
import six

# Latency calculator job suffix.
_LATENCY_CALCULATOR_JOB_SUFFIX = "_latency_calculation"

# Vertex AI service name for corelated jobs.
_JOB_SERVICE_MAP = {"nas": "nasJobs", "custom": "customJobs"}

# API version for corelated jobs.
# GA features are on v1 endpoint; Preview features are on v1beta1.
_API_VERSION_MAP = {"nas": "v1", "custom": "v1"}

# If enabled, NAS CLI will log the API request sent to Vertex AI server.
_ENABLE_REQUEST_LOGS = False

# Page size for ListNasTrialDetails.
_PAGE_SIZE = 20

# Job console template.
_JOB_CONSOLE_TEMPLATE = "https://console.cloud.google.com/vertex-ai/locations/{location}/training/{job_id}/cpu?project={project_id}"

# Service endpoint where cloud job is launched.
_SERVICE_ENDPOINT = "{region}-aiplatform.googleapis.com"

_SERVICE_ENDPOINT_MAP = {
    "PROD": _SERVICE_ENDPOINT,
}

# Retry settings for get-job function.
_RETRY_INTERVAL_SECONDS = 60
_MAX_RETRY = 20


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
          "%s",
          "Failed to call {}, in retry({}/{}\n{})".format(
              func, retry, max_retry, e
          ),
      )
      time.sleep(_RETRY_INTERVAL_SECONDS)
  raise Exception("Failed to call function multiple times.")


def get_service_endpoint(service_endpoint, region):
  return _SERVICE_ENDPOINT_MAP[service_endpoint].format(region=region)


def get_auth():
  """Get access token from the argument or using gcloud auth."""
  return (
      subprocess.check_output(
          ["gcloud", "auth", "application-default", "print-access-token"]
      )
      .strip()
      .decode("utf-8")
  )


def run_command(
    args, expect_json_output=True, enable_logs=_ENABLE_REQUEST_LOGS
):
  """Creates a sub-process running 'args' and returns its JSON outputs."""

  if enable_logs:
    logging.info("Running Command: \n\n%s\n", "\n".join(args))

  process = subprocess.Popen(
      args, stderr=subprocess.PIPE, stdout=subprocess.PIPE
  )
  output, error = process.communicate()

  if process.returncode:
    logging.error(
        "Request failed.\n\n%s\n\nstdout: %s\nstderr: %s\nret: %s\n",
        "\n".join(args),
        output,
        error,
        process.returncode,
    )
    raise RuntimeError(str(error))
  if expect_json_output:
    try:
      if isinstance(output, six.binary_type):
        output = output.decode("utf-8")
      return json.loads(output)
    except ValueError:
      logging.error(
          "Command <%s>; Unable to parse json in output <%s> and error <%s>"
          "\n".join(args),
          output,
          error,
      )
      raise


def run_command_with_stdout(cmd, job_log_file=None, error_message=""):
  """Runs the command and stream the command outputs."""
  if job_log_file is None:
    job_log_file = sys.stdout
  buf = io.StringIO()
  ret_code = None

  with subprocess.Popen(
      cmd,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      universal_newlines=False,
  ) as p:
    out = io.TextIOWrapper(p.stdout, newline="")

    for line in out:
      buf.write(line)
      job_log_file.write(line)
      job_log_file.flush()

    # flush to force the contents to display.
    job_log_file.flush()

    while p.poll() is None:
      # Process hasn't exited yet, let's wait some
      time.sleep(0.5)

    ret_code = p.returncode
    p.stdout.close()

  if ret_code:
    raise RuntimeError(
        "Error: {} with return code {}".format(error_message, ret_code)
    )
  return buf.getvalue(), ret_code


def has_job_failed(job):
  """Checks if the job has failed status."""
  error = job.get("error", None)
  # If the job was cancelled, then do not mark it as failed.
  if error and error["message"] != "CANCELED":
    name = job.get("name", None)
    logging.warning("Could not get info for failed job: %s", name)
    logging.warning("The above job has error status: %s", error)
    logging.warning("The job is: %s", job)
    return True
  else:
    return False


def get_job_by_resource_name(
    vertex_ai_endpoint, resource_name, job_type="nas", check_job_failed=True
):
  """Returns job status by resource name."""
  job = run_command([
      "curl",
      "-X",
      "GET",
      "-v",
      "-H",
      "Authorization: Bearer {}".format(get_auth()),
      "https://{endpoint}/{version}/{resource_name}".format(
          endpoint=vertex_ai_endpoint,
          version=_API_VERSION_MAP[job_type],
          resource_name=resource_name,
      ),
  ])
  if check_job_failed and has_job_failed(job):
    raise Exception("Could not get job status.")
  else:
    return job


def get_job(
    vertex_ai_endpoint,
    project_id,
    location,
    job_id,
    job_type = "nas",
    check_job_failed = True,
):
  """Returns nas job status."""
  resource_uri = "https://{endpoint}/{version}/projects/{project}/locations/{location}/{service}/{job_id}".format(
      endpoint=vertex_ai_endpoint,
      version=_API_VERSION_MAP[job_type],
      project=project_id,
      location=location,
      service=_JOB_SERVICE_MAP[job_type],
      job_id=job_id,
  )
  # Using a retry when making API call in case a failure happens.
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
  if check_job_failed and has_job_failed(job):
    raise Exception("Could not get job status.")
  else:
    return job


def get_job_dir_for_nas_job(nas_job):
  """Returns job-dir for nas-job."""
  return nas_job["nasJobSpec"]["multiTrialAlgorithmSpec"]["searchTrialSpec"][
      "searchTrialJobSpec"
  ]["baseOutputDirectory"]["outputUriPrefix"]


def set_job_dir_for_nas_job(
    nas_job, job_dir
):
  """Updates job-dir for the nas job and returns this updated nas-job."""
  nas_job["nasJobSpec"]["multiTrialAlgorithmSpec"]["searchTrialSpec"][
      "searchTrialJobSpec"
  ]["baseOutputDirectory"]["outputUriPrefix"] = job_dir
  return nas_job


def get_search_trial_dir(job, trial_id):
  """Returns the search trial directory given a job."""
  return os.path.join(get_job_dir_for_nas_job(job), trial_id)


def get_target_metric_for_job(job):
  """Returns the target metric for a job."""
  metric_id = job["nasJobSpec"]["multiTrialAlgorithmSpec"]["metric"]["metricId"]
  return metric_id


def get_goal_for_job(job):
  """Returns the goal for a job."""
  goal = job["nasJobSpec"]["multiTrialAlgorithmSpec"]["metric"]["goal"]
  return goal


def load_valid_trial_metrics_for_job(
    job, metric_id
):
  """Extracts valid metric values and returns a map of trial-id to metric value."""
  search_trials_list = job["nasJobOutput"]["multiTrialJobOutput"][
      "searchTrials"
  ]
  map_trial_id_to_metric_value = {}
  for search_trial in search_trials_list:
    if search_trial["state"] != "SUCCEEDED":
      continue

    # Search for metric_id among multiple metrics.
    metric_val = [
        metric["value"]
        for metric in search_trial["finalMeasurement"]["metrics"]
        if metric["metricId"] == metric_id
    ]

    # There should only be one value corresponding to the metric_id.
    if len(metric_val) != 1:
      continue
    # Extract value from single length list.
    metric_val = metric_val[0]

    trial_id = search_trial["id"]
    map_trial_id_to_metric_value[trial_id] = metric_val

  return map_trial_id_to_metric_value


def get_module(module_path):
  """Returns a module given its full path and returns the result."""
  module_file, module_method_name = module_path.rsplit(".", 1)
  module = importlib.import_module(module_file)
  method = getattr(module, module_method_name)
  return method


def get_trials_page(
    vertex_ai_endpoint, project_id, location, job_id, page_size, page_token
):
  """Get trials page."""
  if page_token:
    page_param = "pageSize={}&pageToken={}".format(page_size, page_token)
  else:
    page_param = "pageSize={}".format(page_size)
  result = run_command([
      "curl",
      "-X",
      "GET",
      "-v",
      "-H",
      "Authorization: Bearer {}".format(get_auth()),
      (
          "https://{endpoint}/{version}/projects/{project}/locations/{location}"
          "/nasJobs/{job_id}/nasTrialDetails?{page}"
      ).format(
          endpoint=vertex_ai_endpoint,
          version=_API_VERSION_MAP["nas"],
          project=project_id,
          location=location,
          job_id=job_id,
          page=page_param,
      ),
  ])
  return result


def list_trials(vertex_ai_endpoint, project_id, location, job_id, max_trials):
  """Returns trials ordered by final measurement."""
  trials = []
  next_page_token = ""

  while True:
    result = get_trials_page(
        vertex_ai_endpoint,
        project_id,
        location,
        job_id,
        page_size=_PAGE_SIZE
        if max_trials <= 0
        else min(_PAGE_SIZE, max_trials - len(trials)),
        page_token=next_page_token,
    )
    if "nasTrialDetails" in result:
      trials.extend(result["nasTrialDetails"])
    if "nextPageToken" in result:
      next_page_token = result["nextPageToken"]
    if "nasTrialDetails" not in result or "nextPageToken" not in result:
      break

  return trials


def start_job(
    vertex_ai_endpoint, project_id, location, job_spec, job_type="nas"
):
  """Starts a nas job."""
  job = run_command([
      "curl",
      "-X",
      "POST",
      "-v",
      "-H",
      "Content-Type: application/json",
      "-d",
      "%s" % json.dumps(job_spec, indent=2),
      "-H",
      "Authorization: Bearer {}".format(get_auth()),
      "https://{endpoint}/{version}/projects/{project}/locations/{location}/{service}"
      .format(
          endpoint=vertex_ai_endpoint,
          version=_API_VERSION_MAP[job_type],
          project=project_id,
          location=location,
          service=_JOB_SERVICE_MAP[job_type],
      ),
  ])
  if job.get("error", None):
    raise Exception("Starting job failed: {}".format(job["error"]))
  return job


def log_job_details(service_endpoint, project_id, location, job, job_type):
  """Logs job details."""
  job_id = job["name"].split("/")[-1]
  job_state = get_job_by_resource_name(service_endpoint, job["name"])["state"]
  job_console_link = _JOB_CONSOLE_TEMPLATE.format(
      location=location, job_id=job_id, project_id=project_id
  )

  logging.info(
      "%s %s started with state %s.", job_type, job["displayName"], job_state
  )
  logging.info("%s job ID: %s\n", job_type, job_id)
  logging.info(
      "View %s job in the Cloud Console at: \n%s\n", job_type, job_console_link
  )
  return job_console_link


def launch_cloud_nas_job(
    service_endpoint, project_id, location, job_spec, job_type
):
  """Launches NasJob through Vertex AI NAS Service."""
  job = start_job(
      service_endpoint, project_id, location, job_spec, job_type="nas"
  )
  job_console_link = log_job_details(
      service_endpoint, project_id, location, job, job_type
  )
  return job["name"], job_console_link


def launch_cloud_custom_job(
    service_endpoint, project_id, location, job_spec, job_type
):
  """Launches CustomJob through Vertex AI Training Service."""
  job = start_job(
      service_endpoint, project_id, location, job_spec, job_type="custom"
  )
  job_console_link = log_job_details(
      service_endpoint, project_id, location, job, job_type
  )
  return job["name"], job_console_link


def launch_cloud_nas_and_latency_job(
    search_job_spec,
    latency_calculator_job_spec,
    service_endpoint,
    region,
    project_id,
):
  """Launches nas and latency jobs on cloud and returns their name and links."""
  search_job_name, search_job_link = launch_cloud_nas_job(
      service_endpoint=service_endpoint,
      project_id=project_id,
      location=region,
      job_spec=search_job_spec,
      job_type="NAS Search",
  )
  if latency_calculator_job_spec:
    # Update the nas-job name launched above in the latency job spec.
    latency_calculator_job_spec = copy.deepcopy(latency_calculator_job_spec)
    latency_calculator_job_spec = reset_nas_job_name_in_latency_job_spec(
        latency_job_spec=latency_calculator_job_spec,
        nas_job_name=search_job_name,
    )
    latency_calculator_job_name, latency_calculator_job_link = (
        launch_cloud_custom_job(
            service_endpoint=service_endpoint,
            project_id=project_id,
            location=region,
            job_spec=latency_calculator_job_spec,
            job_type="Latency Calculator",
        )
    )
  else:
    latency_calculator_job_name = ""
    latency_calculator_job_link = ""
  return (
      search_job_name,
      search_job_link,
      latency_calculator_job_name,
      latency_calculator_job_link,
  )


def extract_container_flags(container_flags):
  """Returns a dict for the `container_flags`.

  Args:
    container_flags: A list of flags in the form ['flag1=val1', 'flag2=val2',
      ...].

  Returns:
    A dictionary mapping flag to its value - {flag1:val1, flag2:val2, ...}.
  """
  container_flags_kv = {}
  for key_val_str in container_flags:
    key_val = key_val_str.split("=", 1)
    if len(key_val) < 2:
      raise ValueError(
          "Container-flag must be of the form 'flag=val': {}".format(
              key_val_str
          )
      )
    key = key_val[0]
    val = key_val[1]
    container_flags_kv[key] = val
  return container_flags_kv


def convert_flag_map_to_list(flag_map):
  """Converts a key-value pair of flag to a list to pass to NAS containers."""
  flag_list = []
  for k, v in flag_map.items():
    flag_list.extend(["--%s=%s" % (k, v)])
  return flag_list


def convert_list_to_flag_map(flag_list):
  """Converts a args list to key-value pair to pass to NAS containers."""
  flag_map = {}
  for flag in flag_list:
    # flag is a string of form "--%s=%s" where first item is a key.
    key = flag.split("=")[0].split("--")[1]
    val = flag.split("=")[1]
    flag_map[key] = val
  return flag_map


def get_latency_calculator_display_name(nas_job_name):
  return nas_job_name + _LATENCY_CALCULATOR_JOB_SUFFIX


def reset_nas_job_name_in_latency_job_spec(
    latency_job_spec, nas_job_name
):
  """Resets nas-job name in the latency job."""
  latency_job_spec["displayName"] = get_latency_calculator_display_name(
      nas_job_name
  )
  # Add nas-job id to the worker-pool specs.
  for worker_pool_spec in latency_job_spec["jobSpec"]["workerPoolSpecs"]:
    flag_map = convert_list_to_flag_map(
        worker_pool_spec["containerSpec"]["args"]
    )
    flag_map["nas_job_id"] = nas_job_name
    worker_pool_spec["containerSpec"]["args"] = convert_flag_map_to_list(
        flag_map
    )
  return latency_job_spec


def nas_job_id_from_nas_job_name(nas_job_name):
  """Returns nas job id from nas job name."""
  # Job name is of the form
  # "projects/<project-id>/locations/<region>/nasJobs/<nas-job-id>"
  return int(nas_job_name.split("/nasJobs/")[1])


def nas_job_region_from_nas_job_name(nas_job_name):
  """Returns nas job region from nas job name."""
  # Job name is of the form
  # "projects/<project-id>/locations/<region>/nasJobs/<nas-job-id>"
  return nas_job_name.split("/locations/")[1].split("/nasJobs/")[0]


def custom_job_id_from_custom_job_name(custom_job_name):
  """Returns custom job id from nas job name."""
  # Job name is of the form
  # "projects/<project-id>/locations/<region>/cusomJobs/<custom-job-id>"
  return int(custom_job_name.split("/customJobs/")[1])


def custom_job_region_from_custom_job_name(custom_job_name):
  """Returns custom job region from nas job name."""
  # Job name is of the form
  # "projects/<project-id>/locations/<region>/customJobs/<custom-job-id>"
  return custom_job_name.split("/locations/")[1].split("/customJobs/")[0]


def is_nas_job_active(nas_job):
  """Return True if NAS job is active."""
  job_state = nas_job["state"]
  return job_state not in (
      "JOB_STATE_FAILED",
      "JOB_STATE_CANCELLING",
      "JOB_STATE_CANCELLED",
      "JOB_STATE_EXPIRED",
      "JOB_STATE_SUCCEEDED",
  )


def is_nas_job_running(nas_job):
  """Return True if NAS job is running."""
  job_state = nas_job["state"]
  return job_state in ("JOB_STATE_RUNNING")


def is_nas_job_complete(nas_job):
  job_state = nas_job["state"]
  return job_state in (
      "JOB_STATE_CANCELLED",
      "JOB_STATE_EXPIRED",
      "JOB_STATE_SUCCEEDED",
  )


def get_running_trials(nas_job):
  """Returns a Set of running trials."""
  running_trials = set()
  multi_trial_job_output = nas_job.get("nasJobOutput", {}).get(
      "multiTrialJobOutput", {}
  )
  if "searchTrials" in multi_trial_job_output:
    for trial in multi_trial_job_output["searchTrials"]:
      if trial["state"] in ("REQUESTED", "ACTIVE", "STOPPING"):
        running_trials.add(trial["id"])
  return running_trials


def get_root_output_dir_from_job_dir(nas_job_dir):
  """Returns GCS root directory for the nas job directory."""
  # GCS job directory is of the form "gs://<GCS root directory>"/..."
  return "gs://" + nas_job_dir.split("/")[2]


def is_job_name_for_custom_job(custom_job_name):
  """Returns True if job-name corresponds to custom job."""
  return "/customJobs/" in custom_job_name


def get_docker_args_map_for_custom_job(
    custom_job
):
  """Returns docker args map for the nas job."""
  job_args_list = custom_job["jobSpec"]["workerPoolSpecs"][0]["containerSpec"][
      "args"
  ]
  return convert_list_to_flag_map(job_args_list)


def get_docker_args_map_for_nas_job(
    nas_job
):
  """Returns docker args map for the nas job."""
  job_args_list = nas_job["nasJobSpec"]["multiTrialAlgorithmSpec"][
      "searchTrialSpec"
  ]["searchTrialJobSpec"]["workerPoolSpecs"][0]["containerSpec"]["args"]
  return convert_list_to_flag_map(job_args_list)


def set_docker_args_map_for_nas_job(
    nas_job, docker_args_map
):
  """Updates docker args map for the nas job and returns this updated nas job."""
  job_args_list = convert_flag_map_to_list(docker_args_map)
  nas_job["nasJobSpec"]["multiTrialAlgorithmSpec"]["searchTrialSpec"][
      "searchTrialJobSpec"
  ]["workerPoolSpecs"][0]["containerSpec"]["args"] = job_args_list
  return nas_job


def get_num_trials_for_nas_job(
    nas_job
):
  """Returns (max_trial_count, max_parallel_trial_count, max_failed_trial_count) for the nas job."""
  search_trial_spec = nas_job["nasJobSpec"]["multiTrialAlgorithmSpec"][
      "searchTrialSpec"
  ]
  max_trial_count = search_trial_spec["maxTrialCount"]
  max_parallel_trial_count = search_trial_spec["maxParallelTrialCount"]
  max_failed_trial_count = search_trial_spec["maxFailedTrialCount"]
  return max_trial_count, max_parallel_trial_count, max_failed_trial_count


def set_num_trials_for_nas_job(
    nas_job,
    max_trial_count,
    max_parallel_trial_count,
    max_failed_trial_count,
):
  """Updates number of trials for the nas job and returns this updated nas job."""
  search_trial_spec = nas_job["nasJobSpec"]["multiTrialAlgorithmSpec"][
      "searchTrialSpec"
  ]
  search_trial_spec["maxTrialCount"] = max_trial_count
  search_trial_spec["maxParallelTrialCount"] = max_parallel_trial_count
  search_trial_spec["maxFailedTrialCount"] = max_failed_trial_count
  return nas_job


def get_display_name_for_nas_job(nas_job):
  """Returns display name for nas-job."""
  return nas_job["displayName"]


def set_display_name_for_nas_job(
    nas_job, display_name
):
  """Updates display-name for the nas job and returns this updated nas-job."""
  nas_job["displayName"] = display_name
  return nas_job


def extract_bucket_name_from_gcs_path(gcs_path):
  gcs_path = gcs_path.strip()
  if not gcs_path.startswith("gs://"):
    raise ValueError("Not a valid gcs-path %s." % gcs_path)
  return (gcs_path.split("gs://", 1))[1].split("/", 1)[0]


def extract_blob_name_from_gcs_path(gcs_path):
  gcs_path = gcs_path.strip()
  if not gcs_path.startswith("gs://"):
    raise ValueError("Not a valid gcs-path %s." % gcs_path)
  return (gcs_path.split("gs://", 1))[1].split("/", 1)[1]


def write_json_object_to_gcs(
    project_id, json_object, gcs_filename
):
  """Writes a json object to a gcs file location."""
  # The storage.Client() complains about a missing project if
  # it was not already set for the user environment.
  os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
  gcs_utils.save_json(filepath=gcs_filename, json_data=json_object)


def get_search_job_spec_file(controller_dir):
  return os.path.join(controller_dir, "search_job_spec.json")


def save_search_job_spec_file(
    project_id, search_job_spec, controller_dir
):
  search_job_spec_json_file = get_search_job_spec_file(controller_dir)
  write_json_object_to_gcs(
      project_id=project_id,
      json_object=search_job_spec,
      gcs_filename=search_job_spec_json_file,
  )


def get_search_space_str(job):
  """Returns the search space json string for a job."""
  return job["nasJobSpec"]["searchSpaceSpec"]


def get_latency_calculator_job_spec_file(controller_dir):
  return os.path.join(controller_dir, "latency_calculator_job_spec.json")


def save_latency_calculator_job_spec_file(
    project_id,
    latency_calculator_job_spec,
    controller_dir,
):
  latency_calculator_job_spec_file = get_latency_calculator_job_spec_file(
      controller_dir
  )
  write_json_object_to_gcs(
      project_id=project_id,
      json_object=latency_calculator_job_spec,
      gcs_filename=latency_calculator_job_spec_file,
  )


def get_trial_ids_by_start_time(job):
  """Returns a list of trial_ids ordered by start time."""
  search_trials_list = job["nasJobOutput"]["multiTrialJobOutput"][
      "searchTrials"
  ]
  trial_start_map = {}
  for search_trial in search_trials_list:
    if search_trial["state"] != "SUCCEEDED":
      continue
    start_time = time.strptime(
        search_trial["startTime"].split(".")[0], "%Y-%m-%dT%H:%M:%S"
    )
    trial_start_map[search_trial["id"]] = start_time
  sorted_trial_ids = sorted(trial_start_map.keys(), key=trial_start_map.get)

  return sorted_trial_ids
