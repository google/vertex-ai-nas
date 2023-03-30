# Copyright 2018 Google Inc.
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
"""MetricsReporter class to work with Google CloudML NAS."""

import collections
import json
import logging
import os
import time

_DEFAULT_HYPERPARAMETER_METRIC_TAG = 'training/hptuning/metric'
_DEFAULT_METRIC_PATH = '/tmp/hypertune/output.metrics'

_MAX_NUM_METRIC_ENTRIES_TO_PRESERVE = 100
_MAX_CUSTOM_METRICS = 5
_METRIC_KEY_PREFIX = 'key_{}'
_METRIC_VALUE_PREFIX = 'value_{}'


class NasMetricsReporter(object):
  """Main class for HyperTune."""

  def __init__(self):
    """Constructor."""
    self.metric_path = os.environ.get('CLOUD_ML_HP_METRIC_FILE',
                                      _DEFAULT_METRIC_PATH)
    if not os.path.exists(os.path.dirname(self.metric_path)):
      os.makedirs(os.path.dirname(self.metric_path))

    self.trial_id = os.environ.get('CLOUD_ML_TRIAL_ID', 0)
    self.metrics_queue = collections.deque(
        maxlen=_MAX_NUM_METRIC_ENTRIES_TO_PRESERVE)

  def _dump_metrics_to_file(self):
    with open(self.metric_path, 'w') as metric_file:
      for metric in self.metrics_queue:
        metric_file.write(json.dumps(metric, sort_keys=True) + '\n')

  def report_metrics(  # pylint: disable=dangerous-default-value
      self,
      hyperparameter_metric_tag,
      metric_value,
      global_step=None,
      checkpoint_path='',
      model_flops=0.0,
      other_metrics={}):
    """Method to report hyperparameter tuning metric.

    Args:
      hyperparameter_metric_tag: The hyperparameter metric name this metric
        value is associated with. Should keep consistent with the tag specified
        in HyperparameterSpec.
      metric_value: float, the values for the hyperparameter metric to report.
      global_step: int, the global step this metric value is associated with.
      checkpoint_path: The checkpoint path which can be used to warmstart from.
      model_flops: Number of FLOPS for the model.
      other_metrics: Dictionary of key value pair of metrics. Only a maximum of
        5 custom metrics are allowed to report. key should be a string with less
        than 65 characters and value should be a number.
    """
    metric_value = float(metric_value)
    if metric_value == 0.0:
      metric_value += 0.00000001
    metric_tag = _DEFAULT_HYPERPARAMETER_METRIC_TAG
    if hyperparameter_metric_tag:
      metric_tag = hyperparameter_metric_tag
    metric_body = {
        'timestamp': time.time(),
        'trial': str(self.trial_id),
        metric_tag: str(metric_value),
        'global_step': str(int(global_step) if global_step else 0),
        'checkpoint_path': checkpoint_path,
        'model_flops': str(float(model_flops))
    }
    count = 0
    if len(other_metrics) > _MAX_CUSTOM_METRICS:
      logging.warning('More than %s metrics found in other_metrics',
                      _MAX_CUSTOM_METRICS)

    for k, v in other_metrics.items():
      if count == _MAX_CUSTOM_METRICS:
        break
      if len(k) > 64:
        logging.warning('Skipping metric %s that exceeds 64 character limit.',
                        k)
        continue
      # Check if other_metrics has any protected keyword.
      if k in (hyperparameter_metric_tag, 'model_flops', 'global_step'):
        logging.warning('Protected keyword %s used in other_metrics. Skipping.',
                        k)
        continue
      metric_body[_METRIC_KEY_PREFIX.format(count)] = k
      metric_body[_METRIC_VALUE_PREFIX.format(count)] = str(float(v))
      count += 1
    self.metrics_queue.append(metric_body)
    self._dump_metrics_to_file()
