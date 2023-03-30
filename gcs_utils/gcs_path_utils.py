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
"""Functions to convert GCS paths."""


def gcs_fuse_path(path):
  """Try to convert path to gcsfuse path if it starts with gs:// else do not modify it."""
  path = path.strip()
  if path.startswith("gs://"):
    return "/gcs/" + path[5:]
  return path


def gs_path(path):
  """Try to convert path to gs:// path if it starts with /gcs/ else do not modify it."""
  path = path.strip()
  if path.startswith("/gcs/"):
    return "gs://" + path[5:]
  return path


def is_gs_path(path):
  """Return True if path is for cloud GCS location."""
  path = gs_path(path)
  if path.startswith("gs://"):
    return True
  else:
    return False
