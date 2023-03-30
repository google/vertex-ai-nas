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
"""Implementation of SpineNet search space.

X. Du, T-Y. Lin, P. Jin, G. Ghiasi, M. Tan, Y. Cui, Q. V. Le, X. Song
SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization
https://arxiv.org/abs/1912.05027
"""

import pyglove as pg
import six
from six.moves import range
from six.moves import zip


@pg.members([
    ('level_base', pg.typing.Int(min_value=0),
     'The base feature level of the current block.'),
    ('level_offset', pg.typing.Int(default=0),
     'The feature level offset. The block level is level_base + level_offset.'),
    ('block_fn',
     pg.typing.Enum(default='bottleneck',
                    values=['bottleneck', 'residual', 'mbconv']),
     'The type of block functions.'),
    ('input_offsets', pg.typing.List(pg.typing.Int(min_value=0), size=2),
     'The offsets of two input features from the previously accumlated '
     'features.'),
    ('is_output', pg.typing.Bool(), 'Whether the current block is an output.'),
])
class TunableBlockSpec(pg.Object):
  """The tunable specifications of a SpineNet block."""


@pg.functor([
    ('intermediate_blocks',
     pg.typing.List(
         pg.typing.Dict([
             ('level_base', pg.typing.Int(min_value=1).noneable(),
              'The base feature level of the current block.'),
             ('level_offset', pg.typing.Int(),
              'The feature level offset. The block level is '
              'level_base + level_offset.'),
             ('block_fn',
              pg.typing.Enum('bottleneck',
                             ['residual', 'bottleneck', 'mbconv']),
              'The type of block functions.'),
             ('input_offsets',
              pg.typing.List(pg.typing.Int(min_value=0), size=2),
              'The offsets of two input features from the previous accumlated '
              'features.'),
         ])), 'A list of specifications that define the intermediate blocks.'),
    ('intermediate_block_levels', pg.typing.List(pg.typing.Int(min_value=1)),
     'A list of intgers that specify the feature level of intermediate blocks'),
    ('output_blocks',
     pg.typing.List(
         pg.typing.Dict([
             ('level_base', pg.typing.Int(min_value=1).noneable(),
              'The base feature level of the current block.'),
             ('level_offset', pg.typing.Int(),
              'The feature level offset. The block level is '
              'level_base + level_offset.'),
             ('block_fn',
              pg.typing.Enum('bottleneck',
                             ['residual', 'bottleneck', 'mbconv']),
              'The type of block functions.'),
             ('input_offsets',
              pg.typing.List(pg.typing.Int(min_value=0), size=2),
              'The offsets of two input features from the previous accumlated '
              'features.'),
         ])), 'A list of specifications that define the output blocks.'),
    ('output_block_levels', pg.typing.List(pg.typing.Int(min_value=1)),
     'A list of intgers that specify the feature level of output blocks'),
])
def build_tunable_block_specs(intermediate_blocks, intermediate_block_levels,
                              output_blocks, output_block_levels):
  """Builds the SpineNet block specification."""

  def _build_block_spec(block, is_output):
    return TunableBlockSpec(
        level_base=block.level_base,
        level_offset=block.level_offset,
        block_fn=block.block_fn,
        input_offsets=block.input_offsets,
        is_output=is_output)

  # Rebinds level_base as they are default to None initially.
  for block, level in zip(intermediate_blocks, intermediate_block_levels):
    block.rebind(level_base=level)
  for block, level in zip(output_blocks, output_block_levels):
    block.rebind(level_base=level)
  return ([
      _build_block_spec(block, is_output=False) for block in intermediate_blocks
  ] + [_build_block_spec(block, is_output=True) for block in output_blocks])


def spinenet_search_space(min_level,
                          max_level,
                          intermediate_blocks_alloc,
                          intermediate_level_offsets,
                          block_fn_candidates,
                          num_blocks_search_window=5):
  """Builds the SpineNet search space.

  Args:
    min_level: an integer that represents the min feature level of the output
      features.
    max_level: an integer that represents the max feature level of the output
      features.
    intermediate_blocks_alloc: a dict that maps intermediate levels to the
      number of intermediate blocks of those levels that can be selected to
      build the SpineNet network. For example: {2:3, 3:3, 4:2} means the network
      will build 3 level-2 blocks, 3 level-3 blocks and 2 level-4 blocks.
    intermediate_level_offsets: a list of integers that specifies for each block
      the possible level adjustments. For example, [0, -1, 1] means for a given
      block of the base level i, its final level can be chosen from
      {i, i-1, i+1}.
    block_fn_candidates: a list of candidates for block function. During search,
      one candidate will be selected as the block function for one block.
    num_blocks_search_window: an integer that specifies how many previous blocks
      before the current block can be considered as candidates for block
      combintation. If -1, all the previous blocks will be considered. Note that
      this is only used for building intermediate blocks.

  Returns:
    a TunableBlockSpecsBuilder object that can be called to build the list of
    BlockSpec for SpineNet architecture.
  """
  num_init_blocks = 2
  num_outputs = max_level - min_level + 1
  num_intermediate_blocks = sum(intermediate_blocks_alloc.values())
  intermediate_block_levels = []
  for k, v in six.iteritems(intermediate_blocks_alloc):
    intermediate_block_levels += v * [k]

  # pylint: disable=g-complex-comprehension
  # pylint: disable=g-long-ternary
  intermediate_blocks = [
      pg.Dict(
          level_base=None,
          level_offset=pg.one_of(intermediate_level_offsets),
          block_fn=pg.one_of(block_fn_candidates),
          input_offsets=pg.sublist_of(
              k=2,
              candidates=list(
                  range(num_init_blocks + i) if num_blocks_search_window == -1
                  else
                  range(max(num_init_blocks + i - num_blocks_search_window, 0),
                        num_init_blocks + i)),
              choices_distinct=True,
              choices_sorted=True))
      for i in range(num_intermediate_blocks)]
  intermediate_levels = pg.sublist_of(
      k=num_intermediate_blocks,
      candidates=intermediate_block_levels,
      choices_distinct=True,
      choices_sorted=False)

  output_blocks = [
      pg.Dict(
          level_base=None,
          level_offset=0,
          block_fn=pg.one_of(block_fn_candidates),
          input_offsets=pg.sublist_of(
              k=2,
              candidates=list(range(
                  num_init_blocks + num_intermediate_blocks + i)),
              choices_distinct=True,
              choices_sorted=True))
      for i in range(num_outputs)]
  output_levels = pg.sublist_of(
      k=num_outputs,
      candidates=list(range(min_level, max_level + 1)),
      choices_distinct=True,
      choices_sorted=False)
  # pylint: enable=g-long-ternary
  # pylint: enable=g-complex-comprehension

  return build_tunable_block_specs(
      intermediate_blocks=intermediate_blocks,
      intermediate_block_levels=intermediate_levels,
      output_blocks=output_blocks,
      output_block_levels=output_levels)

