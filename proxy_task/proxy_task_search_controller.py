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
"""Proxy-task model selection main file."""

from collections.abc import Sequence

from absl import app
from absl import flags
import cloud_nas_utils
from proxy_task import proxy_task_search_controller_lib as search_controller

FLAGS = flags.FLAGS

_PROXY_TASK_MODEL_SELECTION_JOB_ID = flags.DEFINE_integer(
    'proxy_task_model_selection_job_id',
    default=-1,
    help='The numeric job-id for a previous '
    'proxy-task model selection job. This job provides the reference '
    'models for proxy-task evaluation. These models will be reused to '
    'launch multiple proxy-task-jobs, each corresponding to a new '
    'proxy-task-configuration that your generator provides.')
_PROXY_TASK_MODEL_SELECTION_JOB_REGION = flags.DEFINE_string(
    'proxy_task_model_selection_job_region',
    default='',
    help='The region for the proxy-task model selection job.')
_PROXY_TASK_MODEL_SELECTION_SERVICE_ENDPOINT = flags.DEFINE_string(
    'proxy_task_model_selection_service_endpoint',
    default='',
    help='Service endpoint for proxy-task model selection job.')
_PROXY_TASK_CONFIG_GENERATOR_MODULE = flags.DEFINE_string(
    'proxy_task_config_generator_module',
    default='',
    help='The proxy-task configuration generator module. When called, the '
    'module should return a list of proxy-task configurations to try.')
_SEARCH_CONTROLLER_DIR = flags.DEFINE_string(
    'search_controller_dir',
    default='',
    help='Input and output directory for the search controller job.')
_SERVICE_ENDPOINT = flags.DEFINE_string(
    'service_endpoint', default='', help='Service endpoint for cloud job.')
_REGION = flags.DEFINE_string(
    'region', default='', help='Region for cloud job.')
_PROJECT_ID = flags.DEFINE_string('project_id', default='', help='Project id.')
_DESIRED_ACCURACY_CORRELATION = flags.DEFINE_float(
    'desired_accuracy_correlation',
    default=None,
    help="If not 'None', the proxy-task will be stopped if its correlation "
    "crosses this limit. If 'None', then this check is not done. "
    "NOTE: Any of the checks for 'desired_accuracy_correlation', "
    "'desired_accuracy', and 'training_time_hrs_limit' "
    'can independently stop the proxy-task '
    'when triggered. For a more customized behavior, please modify '
    'proxy_task_search_controller_lib.has_met_stopping_condition function.')
_DESIRED_ACCURACY = flags.DEFINE_float(
    'desired_accuracy',
    default=None,
    help="If not 'None', the proxy-task will be stopped if its accuracy "
    "crosses this limit. If 'None', then this check is not done. "
    "NOTE: Any of the checks for 'desired_accuracy_correlation', "
    "'desired_accuracy', and 'training_time_hrs_limit' "
    'can independently stop the proxy-task '
    'when triggered. For a more customized behavior, please modify '
    'proxy_task_search_controller_lib.has_met_stopping_condition function.')
_TRAINING_TIME_HRS_LIMIT = flags.DEFINE_float(
    'training_time_hrs_limit',
    default=None,
    help="If not 'None', the proxy-task will be stopped if its training-time "
    "crosses this limit. If 'None', then this check is not done. "
    "NOTE: Any of the checks for 'desired_accuracy_correlation', "
    "'desired_accuracy', and 'training_time_hrs_limit' "
    'can independently stop the proxy-task '
    'when triggered. For a more customized behavior, please modify '
    'proxy_task_search_controller_lib.has_met_stopping_condition function.')
_DESIRED_LATENCY_CORRELATION = flags.DEFINE_float(
    'desired_latency_correlation',
    default=None,
    help="If not 'None', a proxy-task job is considered a failure if its "
    'latency correlation is less than this threshold.')
_EARLY_STOP_PROXY_TASK_IF_NOT_BEST = flags.DEFINE_bool(
    'early_stop_proxy_task_if_not_best',
    default=True,
    help="If 'True', a proxy-task will be stopped early if its current-cost "
    'is found larger than the previous best proxy-task.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  cloud_nas_utils.setup_logging()
  search_controller.run_search_controller_loop(
      proxy_task_model_selection_job_id=_PROXY_TASK_MODEL_SELECTION_JOB_ID
      .value,
      proxy_task_model_selection_job_region=_PROXY_TASK_MODEL_SELECTION_JOB_REGION
      .value,
      proxy_task_model_selection_service_endpoint=_PROXY_TASK_MODEL_SELECTION_SERVICE_ENDPOINT
      .value,
      proxy_task_config_generator_module=_PROXY_TASK_CONFIG_GENERATOR_MODULE
      .value,
      search_controller_dir=_SEARCH_CONTROLLER_DIR.value,
      service_endpoint=_SERVICE_ENDPOINT.value,
      region=_REGION.value,
      project_id=_PROJECT_ID.value,
      desired_accuracy_correlation=_DESIRED_ACCURACY_CORRELATION.value,
      desired_accuracy=_DESIRED_ACCURACY.value,
      training_time_hrs_limit=_TRAINING_TIME_HRS_LIMIT.value,
      desired_latency_correlation=_DESIRED_LATENCY_CORRELATION.value,
      early_stop_proxy_task_if_not_best=_EARLY_STOP_PROXY_TASK_IF_NOT_BEST
      .value,
  )


if __name__ == '__main__':
  app.run(main)
