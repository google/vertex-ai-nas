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
"""Proxy-task variance measurement constants."""

# File name for storing MeasurementState json data.
MEASUREMENT_STATE_FILE_NAME = 'measurement_state.json'

# File name for storing measurement results.
MEASUREMENT_RESULT_FILE_NAME = 'variance_and_smoothness.json'

# Number of trials to run for measuring variance and smoothness.
NUM_TRIALS_FOR_MEASUREMENT = 5

# The measurement job will abort if failed models are more than this number.
MAX_FAILED_MODELS_TO_ABORT = 10

# Model variance is calculated using Coefficient of variation:
# https://en.wikipedia.org/wiki/Coefficient_of_variation
# If model variance is no more than this value, the variance is good.
MAX_VARIANCE_TO_BE_GOOD = 0.4

# Model smoothness is calculated by summing up derivatives of points sampled
# from interpolated spline curve:
# https://en.wikipedia.org/wiki/Spline_(mathematics)
# If model smoothness is no more than this value, the smoothness is good.
MAX_SMOOTHNESS_TO_BE_GOOD = 10
# Percentage steps for sampling points, the numbers are empirically picked from
# step-wise learning rate decay strategy.
SMOOTHNESS_SAMPLE_SECTIONS = [0.0, 0.7, 0.9, 1.0]
# Number of points to be sampled from each section.
SMOOTHNESS_POINTS_PER_SECTION = 10
