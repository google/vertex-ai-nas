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
"""Implementation of EfficientNetV2 search space."""

import pyglove as pg
from six.moves import zip

NUM_REPEATS = [1, 2, 2, 3, 5, 8]
NUM_OUTPUT_FILTERS = [16, 32, 48, 96, 112, 192]


@pg.members([
    ('num_repeats', pg.typing.Int(default=1),
     'The number of repeats for each block.'),
    ('block_type', pg.typing.Enum(default='mbconv',
                                  values=['conv', 'mbconv', 'fused_mbconv']),
     'The type of block functions.'),
    ('expand_ratio', pg.typing.Int(default=1),
     'The expansion ratio of the MBConv block.'),
    ('kernel_size', pg.typing.Int(default=3), 'The kernel size.'),
    ('se_ratio', pg.typing.Float(default=0.0), 'The squeeze-excitation ratio.'),
    ('output_filters', pg.typing.Int(), 'The number of output filters'),
    ('act_fn', pg.typing.Enum(default='swish', values=['swish', 'relu']),
     'Activation functions'),
])
class TunableBlockSpec(pg.Object):
  """The tunable specifications of a EfficientNetV2 block."""


@pg.functor([
    ('blocks',
     pg.typing.List(
         pg.typing.Dict([
             ('num_repeats', pg.typing.Int(default=1),
              'The number of repeats for each block.'),
             ('block_type',
              pg.typing.Enum(default='mbconv',
                             values=['conv', 'mbconv', 'fused_mbconv']),
              'The type of block functions.'),
             ('expand_ratio', pg.typing.Int(default=1),
              'The expansion ratio of the MBConv block.'),
             ('kernel_size', pg.typing.Int(default=3), 'The kernel size.'),
             ('se_ratio', pg.typing.Float(default=0.0),
              'The squeeze-excitatio ratio.'),
             ('output_filters', pg.typing.Int(),
              'The number of output filters'),
             ('act_fn',
              pg.typing.Enum(default='swish', values=['swish', 'relu'])),
         ]),
         size=6)),
])
def build_tunable_block_specs(blocks):  # pytype: disable=not-instantiable  # always-use-return-annotations
  """Builds the EfficientNetV2 block specification."""

  # pylint: disable=g-complex-comprehension
  return [
      TunableBlockSpec(
          num_repeats=b.num_repeats,
          block_type=b.block_type,
          expand_ratio=b.expand_ratio,
          kernel_size=b.kernel_size,
          se_ratio=b.se_ratio,
          output_filters=b.output_filters,
          act_fn=b.act_fn)
      for b in blocks]
  # pylint: enable=g-complex-comprehension


def efficientnetv2_search_space(base_num_repeats=None,
                                base_num_output_filters=None):
  """Builds the EfficientNetV2 search space.

  Args:
    base_num_repeats: a list of integers that specifies the number of layers to
      repeat in all blocks.
    base_num_output_filters: a list of integers that specifies the number of
      output filters per block.

  Returns:
    a TunableBlockSpecsBuilder object that can be called to build the list of
    BlockSpec for EfficientNetV2 architecture.
  """
  # pylint: disable=g-long-ternary
  # pylint: disable=g-complex-comprehension

  if not base_num_repeats:
    base_num_repeats = NUM_REPEATS

  if not base_num_output_filters:
    base_num_output_filters = NUM_OUTPUT_FILTERS

  if len(base_num_repeats) != len(base_num_output_filters):
    raise ValueError(
        ('The length of base_num_output_filters {} is different from the length'
         'of base_num_output_filters {}').format(base_num_repeats,
                                                 base_num_output_filters))

  blocks = [
      pg.Dict(
          num_repeats=pg.one_of([r, r - 1, r + 1] if r > 1 else [1, 2]),
          block_type=pg.one_of(['mbconv', 'fused_mbconv']),
          expand_ratio=pg.one_of([1, 2, 4, 6]),
          kernel_size=pg.one_of([3, 5]),
          se_ratio=pg.one_of([0.0, 0.25]),
          output_filters=pg.one_of([int(o * 0.75), o,
                                    int(o * 1.25)]),
          act_fn=pg.one_of(['swish', 'relu']))
      for r, o in zip(base_num_repeats, base_num_output_filters)
  ]
  # pylint: enable=g-long-ternary
  # pylint: enable=g-complex-comprehension
  return build_tunable_block_specs(blocks=blocks)
