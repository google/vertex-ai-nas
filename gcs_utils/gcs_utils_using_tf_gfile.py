# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tensorflow 'gfile' based I/O functions which work with both GCS and local filesystem.

NOTE: The functions here share the same interface as all other gcs_utils
files. Using the same interface makes it easy to switch between
different underlying libraries.
"""

import json
import logging
import os
from typing import Any, Optional

from gcs_utils import gcs_path_utils
import tensorflow as tf


def file_open(filepath, mode):
  """Returns a file open context manager."""
  filepath = gcs_path_utils.gs_path(filepath)
  return tf.io.gfile.GFile(filepath, mode)


def exists(filepath):
  """Returns true if filepath exists."""
  filepath = gcs_path_utils.gs_path(filepath)
  return tf.io.gfile.exists(filepath)


def makedirs(dir_path):
  """Creates directory path if does not exists."""
  dir_path = gcs_path_utils.gs_path(dir_path)
  tf.io.gfile.makedirs(dir_path)


def load_json(filepath):
  """Loads json file from GCS and returns json-data if successful else returns None."""
  filepath = gcs_path_utils.gs_path(filepath)
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
  filepath = gcs_path_utils.gs_path(filepath)
  makedirs(os.path.dirname(filepath))
  with file_open(filepath, "w") as f:
    json.dump(json_data, f, indent=2)
  logging.info("Saved %s.", filepath)


def copy(src_filepath, dst_filepath):
  """Copies src_filepath to dst_filepath and overwrites it if exists."""
  src_filepath = gcs_path_utils.gs_path(src_filepath)
  dst_filepath = gcs_path_utils.gs_path(dst_filepath)
  makedirs(os.path.dirname(dst_filepath))
  tf.io.gfile.copy(src_filepath, dst_filepath, overwrite=True)
  logging.info("Copied file %s to %s", src_filepath, dst_filepath)
