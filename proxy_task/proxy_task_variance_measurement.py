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
"""Proxy-task variance and smoothness measurement main file."""

from collections.abc import Sequence

from absl import app
from absl import flags
import cloud_nas_utils
from proxy_task import proxy_task_variance_measurement_lib as variance_measurement

_VARIANCE_MEASUREMENT_DIR = flags.DEFINE_string(
    'variance_measurement_dir',
    default='',
    help='Input and output directory for the variance measurement job.')
_SERVICE_ENDPOINT = flags.DEFINE_string(
    'service_endpoint', default='', help='Service endpoint for cloud job.')
_REGION = flags.DEFINE_string(
    'region', default='', help='Region for cloud job.')
_PROJECT_ID = flags.DEFINE_string('project_id', default='', help='Project id.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  cloud_nas_utils.setup_logging()
  variance_measurement.run_measurement(
      measurement_dir=_VARIANCE_MEASUREMENT_DIR.value,
      service_endpoint=_SERVICE_ENDPOINT.value,
      region=_REGION.value,
      project_id=_PROJECT_ID.value)


if __name__ == '__main__':
  app.run(main)
