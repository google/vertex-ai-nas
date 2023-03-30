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
"""Implementation of Mnas search space."""

import pyglove as pg
from six.moves import zip

from tf1.detection.modeling.architecture import mnasnet_constants

MOBILENET_V2_NUM_REPEATS = [1, 2, 3, 4, 3, 3, 1]
MOBILENET_V2_OUTPUT_FILTERS = [16, 24, 32, 64, 96, 160, 320]
MNASNET_A1_NUM_REPEATS = [1, 2, 3, 4, 2, 3, 1]
MNASNET_A1_OUTPUT_FILTERS = [16, 24, 40, 80, 112, 160, 320]


@pg.members([
    ('num_repeats', pg.typing.Int(default=1),
     'The number of repeats for each block.'),
    ('block_fn', pg.typing.Enum(default='mbconv', values=['conv', 'mbconv']),
     'The type of block functions.'),
    ('expand_ratio', pg.typing.Int(default=1),
     'The expansion ratio of the MBConv block.'),
    ('kernel_size', pg.typing.Int(default=3), 'The kernel size.'),
    ('se_ratio', pg.typing.Float(default=0.0), 'The squeeze-excitation ratio.'),
    ('output_filters', pg.typing.Int(), 'The number of output filters'),
])
class TunableBlockSpec(pg.Object):
  """The tunable specifications of a MnasNet block."""


@pg.functor([
    ('blocks',
     pg.typing.List(
         pg.typing.Dict([
             ('num_repeats', pg.typing.Int(default=1),
              'The number of repeats for each block.'),
             ('block_fn',
              pg.typing.Enum(default='mbconv', values=['conv', 'mbconv']),
              'The type of block functions.'),
             ('expand_ratio', pg.typing.Int(default=1),
              'The expansion ratio of the MBConv block.'),
             ('kernel_size', pg.typing.Int(default=3), 'The kernel size.'),
             ('se_ratio', pg.typing.Float(default=0.0),
              'The squeeze-excitatio ratio.'),
             ('output_filters', pg.typing.Int(),
              'The number of output filters'),
         ]), size=mnasnet_constants.MNASNET_NUM_BLOCKS)),
])
def build_tunable_block_specs(blocks):
  """Builds the MnasNet block specification."""

  # pylint: disable=g-complex-comprehension
  return [
      TunableBlockSpec(
          num_repeats=b.num_repeats,
          block_fn=b.block_fn,
          expand_ratio=b.expand_ratio,
          kernel_size=b.kernel_size,
          se_ratio=b.se_ratio,
          output_filters=b.output_filters)
      for b in blocks]
  # pylint: enable=g-complex-comprehension


def mnasnet_search_space(reference='mobilenet_v2'):
  """Builds the MnasNet search space.

  Args:
    reference: the reference model the search is based on. The search space of
      num_repeats and num_output_filters are based on the setup in the reference
      model. Current support `mobilenet_v2` and `mnasnet_a1`.

  Returns:
    a TunableBlockSpecsBuilder object that can be called to build the list of
    BlockSpec for MnasNet architecture.
  """
  if reference == 'mobilenet_v2':
    base_num_repeats = MOBILENET_V2_NUM_REPEATS
    base_num_output_filters = MOBILENET_V2_OUTPUT_FILTERS
  elif reference == 'mnasnet_a1':
    base_num_repeats = MNASNET_A1_NUM_REPEATS
    base_num_output_filters = MNASNET_A1_OUTPUT_FILTERS

  # pylint: disable=g-long-ternary
  # pylint: disable=g-complex-comprehension
  blocks = [
      pg.Dict(
          num_repeats=pg.one_of([r, r - 1, r + 1] if r > 1 else [1, 2]),
          block_fn=pg.one_of(['conv', 'mbconv']),
          expand_ratio=pg.one_of([1, 3, 6]),
          kernel_size=pg.one_of([3, 5]),
          se_ratio=pg.one_of([0.0, 0.25]),
          output_filters=pg.one_of([int(o * 0.75), o, int(o * 1.25)]))
      for r, o in zip(base_num_repeats, base_num_output_filters)]
  # pylint: enable=g-long-ternary
  # pylint: enable=g-complex-comprehension
  return build_tunable_block_specs(blocks=blocks)
