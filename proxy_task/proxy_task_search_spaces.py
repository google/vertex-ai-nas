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
"""Proxy-task search space definitions.
"""

import copy
import logging
import os
import re
from typing import Any, Dict, List, Optional
from proxy_task import proxy_task_utils

# MNasnet training-data size choices.
MNASNET_TRAINING_DATA_PCT_LIST = [25, 50, 75, 95]

# Training data path regex pattern.
_TRAINING_DATA_PATH_REGEX = r"gs://.*/.*"


def update_mnasnet_proxy_training_data(
    baseline_docker_args_map,
    training_data_pct):
  """Updates MNasnet baseline docker to use a certain training_data_pct."""
  proxy_task_docker_args_map = copy.deepcopy(baseline_docker_args_map)
  # Imagenet training data path looks like:
  # gs://<path to imagenet data>/train-00[0-7]??-of-01024.
  if not re.match(_TRAINING_DATA_PATH_REGEX,
                  baseline_docker_args_map["training_data_path"]):
    raise ValueError(
        "Training data path %s does not match the desired pattern." %
        baseline_docker_args_map["training_data_path"])

  root_path, _ = baseline_docker_args_map["training_data_path"].rsplit("/", 1)
  if training_data_pct == 25:
    proxy_task_docker_args_map["training_data_path"] = os.path.join(
        root_path, "train-00[0-1][0-4]?-of-01024*")
  elif training_data_pct == 50:
    proxy_task_docker_args_map["training_data_path"] = os.path.join(
        root_path, "train-00[0-4]??-of-01024*")
  elif training_data_pct == 75:
    proxy_task_docker_args_map["training_data_path"] = os.path.join(
        root_path, "train-00[0-6][0-4]?-of-01024*")
  elif training_data_pct == 95:
    proxy_task_docker_args_map["training_data_path"] = os.path.join(
        root_path, "train-00[0-8][0-4]?-of-01024*")
  else:
    logging.warning("Mnasnet training_data_pct %d is not supported.",
                    training_data_pct)
    return None
  proxy_task_docker_args_map["validation_data_path"] = os.path.join(
      root_path, "train-009[0-4]?-of-01024")
  return proxy_task_docker_args_map


def mnasnet_proxy_task_config_generator(
    baseline_docker_args_map
):
  """Returns a list of proxy-task configs to be evaluated for MNasnet.

  Args:
    baseline_docker_args_map: A set of baseline training-docker arguments in
      the form of a dictionary of {'key', val}. The different proxy-task
      configs to try can be built by modifying this baseline.

  Returns:
    A list of proxy-task configs to be evaluated for this
    proxy-task search space.
  """
  proxy_task_config_list = []
  # NOTE: Will not search over model-scale for MNasnet.
  for training_data_pct in MNASNET_TRAINING_DATA_PCT_LIST:
    proxy_task_docker_args_map = update_mnasnet_proxy_training_data(
        baseline_docker_args_map=baseline_docker_args_map,
        training_data_pct=training_data_pct)
    if not proxy_task_docker_args_map:
      continue
    proxy_task_name = "mnasnet_proxy_training_data_pct_{}".format(
        training_data_pct)
    proxy_task_config_list.append(
        proxy_task_utils.ProxyTaskConfig(
            name=proxy_task_name, docker_args_map=proxy_task_docker_args_map))
  return proxy_task_config_list
