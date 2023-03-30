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
"""Google-cloud-storage lib based I/O functions which work with both GCS and local filesystem.

NOTE: The functions here share the same interface as all other gcs_utils
files. Using the same interface makes it easy to switch between
different underlying libraries.
"""

import json
import logging
import os
import shutil
from typing import Any, Optional, Tuple

from google.cloud import storage
from gcs_utils import gcs_path_utils


def is_cloud_path(path):
  """Return True if path is for cloud GCS location."""
  path = gcs_path_utils.gs_path(path)
  if path.startswith("gs://"):
    return True
  else:
    return False


def get_bucket_and_blob_name(filepath):
  # The gcs path is of the form gs://<bucket-name>/<blob-name>
  filepath = gcs_path_utils.gs_path(filepath)
  gs_suffix = filepath.split("gs://", 1)[1]
  return tuple(gs_suffix.split("/", 1))


def exists(filepath):
  """Returns true if filepath exists."""
  filepath = gcs_path_utils.gs_path(filepath)
  if is_cloud_path(filepath):
    client = storage.Client()
    bucket_name, blob_name = get_bucket_and_blob_name(filepath)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.exists()
  else:
    return os.path.exists(filepath)


def makedirs(dir_path):
  """Creates directory path if does not exists."""
  dir_path = gcs_path_utils.gs_path(dir_path)
  if is_cloud_path(dir_path):
    bucket_name, blob_name = get_bucket_and_blob_name(dir_path)
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    # Need to add an extra '/' at the end.
    blob = bucket.blob(blob_name + "/")
    blob.upload_from_string("")
  else:
    os.makedirs(dir_path, exist_ok=True)


def file_open(filepath, mode):
  """Returns a file open context manager."""
  filepath = gcs_path_utils.gs_path(filepath)
  if is_cloud_path(filepath):
    client = storage.Client()
    bucket_name, blob_name = get_bucket_and_blob_name(filepath)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.open(mode)
  else:
    return open(filepath, mode)


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
  """Saves json file."""
  # Create the path to the file if it does not exist already.
  filepath = gcs_path_utils.gs_path(filepath)
  makedirs(os.path.dirname(filepath))
  with file_open(filepath, "w") as f:
    json.dump(json_data, f, indent=2)
  logging.info("Saved %s.", filepath)


def copy(src_filepath, dst_filepath):
  """Copies src_filepath to dst_filepath."""
  src_filepath = gcs_path_utils.gs_path(src_filepath)
  dst_filepath = gcs_path_utils.gs_path(dst_filepath)
  makedirs(os.path.dirname(dst_filepath))
  if is_cloud_path(src_filepath):
    src_bucket_name, src_blob_name = get_bucket_and_blob_name(src_filepath)
    dst_bucket_name, dst_blob_name = get_bucket_and_blob_name(dst_filepath)
    client = storage.Client()
    src_bucket = client.get_bucket(src_bucket_name)
    src_blob = src_bucket.blob(src_blob_name)
    dst_bucket = client.get_bucket(dst_bucket_name)
    src_bucket.copy_blob(src_blob, dst_bucket, dst_blob_name)
  else:
    shutil.copy(src_filepath, dst_filepath)
  logging.info("Copied file %s to %s", src_filepath, dst_filepath)
