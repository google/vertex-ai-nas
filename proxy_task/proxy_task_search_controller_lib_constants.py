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
"""Proxy-task search controller constants."""

INVALID_ACCURACY = -1.0
INVALID_TRAINING_TIME = -1.0
INVALID_TRAINING_STEP = -1
INVALID_CORRELATION = -5.0
INVALID_P_VALUE = -5.0
NUM_SECS_IN_ONE_HOUR = 3600

# Minimum number of values needed to compute a correlation score.
MIN_REQUIRED_NUM_CORRELATION_METRICS = 5
# Minimum ratio of valid trials to total trials to compute
# a correlation score. For example, need 7 out of 10 models with valid scores.
MIN_RATIO_OF_VALID_TRIALS_TO_TOTAL_TRIALS = 0.7

# Search-controller-state filename.
SEARCH_CONTROLLER_STATE_FILENAME = "SEARCH_CONTROLLER_STATE.json"
