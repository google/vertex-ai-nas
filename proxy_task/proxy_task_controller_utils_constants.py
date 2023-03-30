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
"""Proxy-task controller utils constants."""

# Duration in seconds for which trial monitoring function sleeps.
TRIAL_MONITOR_SLEEP_SECS = 60

# Max training step-pct.
MAX_TRAINING_STEP_PCT = 100.0
# Min training step-pct.
MIN_TRAINING_STEP_PCT = 0.0
# Training-step pct to complete training.
TRAINING_STEP_PCT_TO_COMPLETE_TRAINING = 101.0
# Controller alive timestamp filename.
CONTROLLER_ALIVE_TIMESTAMP_FILE = "controller_alive_timestamp.json"
# Controller alive timestamp key.
CONTROLLER_ALIVE_TIMESTAMP_KEY = "timestamp"
# Controller alive threshold in seconds.
# In case the controller restarts, it needs around 10 mins to come back up.
# Therefore set this check to 15 mins to give it time.
CONTROLLER_ALIVE_TIMESTAMP_THRESHOLD = 15 * 60
# Controller current child nas-job name file.
CONTROLLER_CURRENT_NAS_JOB_NAME_FILE = "current_nas_job_name.json"
CONTROLLER_CURRENT_NAS_JOB_NAME_KEY = "current_nas_job_name"
# Controller job name.
CONTROLLER_JOB_NAME_FILE = "controller_job_name.json"
CONTROLLER_JOB_NAME_KEY = "controller_job_name"
