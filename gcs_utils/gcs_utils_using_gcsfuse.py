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
"""Gcsfuse based I/O functions which work with both GCS and local filesystem.

Gcsfuse gets mounted automatically for Vertex AI training. However, for
GCS access via on-prem docker run, it needs to be mounted inside the docker.
Gcsfuse usage is not recommended for fast I/O between multiple
workers working on the same file. It may give errors such
as the read failing even when the file exists and was written amoment ago
by another process. But if only one process is doing read or write
or there are only multiple readers and no writers then it is safe to use it.

NOTE: The functions here share the same interface as all other gcs_utils
files. Using the same interface makes it easy to switch between
different underlying libraries.
"""

import json
import logging
import os
import shutil
from typing import Any, Optional

from gcs_utils import gcs_path_utils


def exists(filepath):
  """Returns true if filepath exists."""
  filepath = gcs_path_utils.gcs_fuse_path(filepath)
  return os.path.exists(filepath)


def makedirs(dir_path):
  """Creates directory path if does not exists."""
  dir_path = gcs_path_utils.gcs_fuse_path(dir_path)
  os.makedirs(dir_path, exist_ok=True)


def file_open(filepath, mode):
  """Returns a file open context manager."""
  filepath = gcs_path_utils.gcs_fuse_path(filepath)
  return open(filepath, mode)


def load_json(filepath):
  """Loads json file from GCS and returns json-data if successful else returns None."""
  if exists(filepath):
    try:
      with file_open(filepath, "r") as f:
        json_data = json.load(f)
    except Exception:  # pylint: disable=broad-except
      logging.warning("Loading %s failed.", filepath)
      return None
    return json_data
  else:
    logging.info("File %s does not exist.", filepath)
    return None


def save_json(filepath, json_data):
  """Saves json file and returns True if successful."""
  # Create the path to the file if it does not exist already.
  makedirs(os.path.dirname(filepath))
  with file_open(filepath, "w") as f:
    json.dump(json_data, f, indent=2)
  logging.info("Saved %s.", filepath)


def copy(src_filepath, dst_filepath):
  """Copies src_filepath to dst_filepath."""
  src_filepath = gcs_path_utils.gcs_fuse_path(src_filepath)
  dst_filepath = gcs_path_utils.gcs_fuse_path(dst_filepath)
  makedirs(os.path.dirname(dst_filepath))
  shutil.copy(src_filepath, dst_filepath)
  logging.info("Copied file %s to %s", src_filepath, dst_filepath)
