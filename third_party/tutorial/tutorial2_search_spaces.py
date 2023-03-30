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

"""Search space examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cloud_nas_utils
import tf_utils
from gcs_utils import gcs_path_utils
from third_party.tutorial import search_spaces


def create_arg_parser():
  """Creates arg parser."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # Nas-service related flags.
  parser.add_argument(
      '--nas_params_str', type=str, help='Nas args serialized in JSON string.')
  parser.add_argument(
      '--job-dir', type=str, default='tmp', help='Job output directory.')
  parser.add_argument(
      '--retrain_search_job_dir',
      type=str,
      help='The job dir of the NAS search job to retrain.')
  parser.add_argument(
      '--retrain_search_job_trials',
      type=str,
      help='A list of trial IDs of the NAS search job to retrain, '
      'separated by comma.')
  # Flags specific to this trainer.
  parser.add_argument(
      '--search_space', type=str, help='Search space choice.')

  return parser


def train_and_eval(argv):
  """Trains and eval the NAS model."""

  argv.job_dir = cloud_nas_utils.get_job_dir_from_environment_if_exist(
      current_job_dir=argv.job_dir)
  argv.job_dir = gcs_path_utils.gcs_fuse_path(argv.job_dir)
  # Create job-dir if it does not exist.
  if not os.path.exists(argv.job_dir):
    os.makedirs(argv.job_dir)

  # Get search-space from search-space-name.
  search_space = search_spaces.get_search_space(argv.search_space)
  # Process nas_params_str passed by the NAS-service.
  # This gives one instance of the search-space to be used for this trial.
  tunable_object = cloud_nas_utils.parse_and_save_nas_params_str(
      search_space=search_space,
      nas_params_str=argv.nas_params_str,
      model_dir=argv.job_dir,
  )
  unused_serialized_tunable_object = (
      cloud_nas_utils.serialize_and_save_tunable_object(
          tunable_object=tunable_object, model_dir=argv.job_dir
      )
  )
  # NOTE: tunable_object is a `model-spec` which can be used to build
  # a model for this trial.

  # Since this trainer does not write output files, we will save out
  # dummy files here to illustrate GCS file I/O.
  with open(os.path.join(argv.job_dir, 'dummy_output.txt'), 'w') as fp:
    fp.write('Finished training.')


if __name__ == '__main__':
  tf_utils.suppress_tf_logging()
  cloud_nas_utils.setup_logging()
  flags = create_arg_parser().parse_args()
  train_and_eval(flags)
