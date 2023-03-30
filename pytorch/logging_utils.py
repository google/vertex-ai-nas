# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Util functions for logging set up."""
import logging
import sys


def setup_logging():
  """Sets up logging."""
  root_logger = logging.getLogger()
  root_logger_previous_handlers = list(root_logger.handlers)
  for h in root_logger_previous_handlers:
    root_logger.removeHandler(h)
  root_logger.setLevel(logging.INFO)
  root_logger.propagate = False

  # Set tf logging to avoid duplicate logging. If the handlers are not removed,
  # then we will have duplicate logging :
  # From tf loggging written to stderr stream, and
  # From python logger written to stdout stream.
  tf_logger = logging.getLogger("tensorflow")
  while tf_logger.handlers:
    tf_logger.removeHandler(tf_logger.handlers[0])

  # Redirect INFO logs to stdout
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging.INFO)
  root_logger.addHandler(stdout_handler)
