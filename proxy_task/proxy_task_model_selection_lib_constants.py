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
"""Proxy-task model selection constants."""

# Initial number of models to begin with.
START_NUM_MODELS = 30
# Final number of models to end with.
FINAL_NUM_MODELS = 10
# Max allowed failed models: 20% of start num models.
MAX_ALLOWED_FAILURES = int(START_NUM_MODELS / 5.0)
# Number of models to remove per iteration.
NUM_MODELS_TO_REMOVE_PER_ITER = 5

# Desired ratio of valid trial scores to total trials.
VALID_TRIALS_RATIO_THRESHOLD = 0.7

# A numpy array stores the trial-id, accuracy, and latency keys.
TRIAL_ID = "trial_id"
ACCURACY = "accuracy"
LATENCY = "latency"

FILTERED_TRIAL_SCORES_PLOT_FILENAME = "filtered_trial_scores.png"

# Maximum distance between trial scores.
MAX_TRIAL_SCORE_DISTANCE = 100.0

# Invalid trial id.
INVALID_TRIAL_ID = "Invalid"

# Minimum number of required models.
MIN_NUM_FILTERED_TRIALS = 2

# Model-selection-state filename.
MODEL_SELECTION_STATE_FILENAME = "MODEL_SELECTION_STATE.json"
