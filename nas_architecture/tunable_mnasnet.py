# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation of Mnas tunable model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import pyglove as pg

import tensorflow.compat.v1 as tf

from tf1.detection.modeling.architecture import mnasnet
from tf1.detection.modeling.architecture import nn_ops
from nas_architecture import tunable_mnasnet_search_space  # pylint: disable=unused-import


def build_static_block_specs_json(tunable_block_specs_json):
  """Builds the static MnasNet block specs JSON string.

  Args:
    tunable_block_specs_json: a JSON string that specifies the MnasNet
      block configuration and can be tuned for architecture search.

  Returns:
    a JSON string that can be loaded by the MnasNet static model.
  """
  # pylint: disable=g-complex-comprehension
  static_block_specs_list = [
      (b.num_repeats,
       b.block_fn,
       b.expand_ratio,
       b.kernel_size,
       b.se_ratio,
       b.output_filters)
      for b in pg.from_json_str(tunable_block_specs_json)]
  # pylint: enable=g-complex-comprehension
  return json.dumps(static_block_specs_list)


def build_static_block_specs(tunable_block_specs):
  """Builds the static MnasNet BlockSpec.

  Args:
    tunable_block_specs: a list of TunableBlockSpec that specifies the MnasNet
      block configuration and can be tuned for architecture search.

  Returns:
    a list of static BlockSpec that can be loaded by the MnasNet static model.
  """
  # pylint: disable=g-complex-comprehension
  return [
      mnasnet.BlockSpec(
          b.num_repeats,
          b.block_fn,
          b.expand_ratio,
          b.kernel_size,
          b.se_ratio,
          b.output_filters)
      for b in tunable_block_specs]
  # pylint: enable=g-complex-comprehension


def build_mnasnet(tunable_block_specs,
                  batch_norm_activation=nn_ops.BatchNormActivation()):
  """MnasNet network architecture."""

  return mnasnet.MnasNet(
      block_specs=build_static_block_specs(tunable_block_specs),
      batch_norm_activation=batch_norm_activation,
      data_format='channels_last')


def tunable_mnasnet_builder(block_specs_json,
                            batch_norm_activation=nn_ops.BatchNormActivation()):
  """Builds the MnasNet network."""
  if block_specs_json.endswith('.json'):
    with tf.gfile.GFile(block_specs_json) as f:
      block_specs_json = f.read()
  tunable_block_specs = pg.from_json_str(block_specs_json)
  return build_mnasnet(
      tunable_block_specs=tunable_block_specs,
      batch_norm_activation=batch_norm_activation)
