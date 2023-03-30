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
"""Implementation of Mnas search space and tunable model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

from pytorch.classification import mnasnet
from pytorch.classification import search_space
import pyglove as pg


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


@pg.members([
    ('tunable_block_specs',
     pg.typing.List(pg.typing.Object(search_space.TunableBlockSpec)),
     'a list of tunable block spec to define the MnasNet network.'),
    ('batch_norm_activation',
     pg.typing.Any(default=mnasnet.BatchNormActivation),
     'Batchnorm followed by an optional activation layer'),
])
class TunableMnasNet(pg.Object):
  """MnasNet network architecture."""

  def __call__(self, inputs, is_training=False):
    mnasnet_fn = mnasnet.MnasNet(
        block_specs=build_static_block_specs(self.tunable_block_specs),
        batch_norm_activation=self.batch_norm_activation,
        data_format='channels_last')
    return mnasnet_fn(inputs, is_training=is_training)


def tunable_mnasnet_builder(block_specs_json,
                            batch_norm_activation=mnasnet.BatchNormActivation):
  """Builds the MnasNet network."""
  if block_specs_json.endswith('.json'):
    with open(block_specs_json) as f:
      block_specs_json = f.read()
  tunable_block_specs = pg.from_json_str(block_specs_json)
  return TunableMnasNet(
      tunable_block_specs=tunable_block_specs,
      batch_norm_activation=batch_norm_activation)
