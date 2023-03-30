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
"""Tunable NAS-FPN search-space.

Golnaz Ghiasi, Tsung-Yi Lin, Ruoming Pang, Quoc V. Le.
NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection.
https://arxiv.org/abs/1904.07392. CVPR 2019.
"""

import pyglove as pg


@pg.members([
    ('level', pg.typing.Int(min_value=0),
     'The feature level of the current block.'),
    ('combine_fn',
     pg.typing.Enum(default='sum', values=['sum', 'attention']),
     'The type of block functions.'),
    ('input_offsets', pg.typing.List(pg.typing.Int(min_value=0), size=2),
     'The offsets of two input features from the previously accumlated '
     'features.'),
    ('is_output', pg.typing.Bool(), 'Whether the current block is an output.'),
])
class TunableBlockSpec(pg.Object):
  """The tunable specifications of a NAS-FPN block."""


@pg.functor([
    ('intermediate_blocks',
     pg.typing.List(
         pg.typing.Dict([
             ('level', pg.typing.Int(min_value=1).noneable(),
              'The base feature level of the current block.'),
             ('combine_fn', pg.typing.Enum('sum', ['sum', 'attention']),
              'The type of combine functions.'),
             ('input_offsets',
              pg.typing.List(pg.typing.Int(min_value=0), size=2),
              'The offsets of two input features from the previous accumlated '
              'features.'),
         ])), 'A list of specifications that define the intermediate blocks.'),
    ('output_blocks',
     pg.typing.List(
         pg.typing.Dict([
             ('level', pg.typing.Int(min_value=1).noneable(),
              'The base feature level of the current block.'),
             ('combine_fn', pg.typing.Enum('sum', ['sum', 'attention']),
              'The type of combine functions.'),
             ('input_offsets',
              pg.typing.List(pg.typing.Int(min_value=0), size=2),
              'The offsets of two input features from the previous accumlated '
              'features.'),
         ])), 'A list of specifications that define the output blocks.'),
    ('output_block_levels', pg.typing.List(pg.typing.Int(min_value=1)),
     'A list of intgers that specify the feature level of output blocks'),
])
def build_tunable_block_specs(intermediate_blocks, output_blocks,
                              output_block_levels):
  """Builds the NAS-FPN block specification."""

  def _build_block_spec(block, is_output):
    return TunableBlockSpec(
        level=block.level,
        combine_fn=block.combine_fn,
        input_offsets=block.input_offsets,
        is_output=is_output)

  # Rebinds level_base as they are default to None initially.
  for block, level in zip(output_blocks, output_block_levels):
    block.rebind(level=level)
  return ([
      _build_block_spec(block, is_output=False) for block in intermediate_blocks
  ] + [_build_block_spec(block, is_output=True) for block in output_blocks])


def nasfpn_search_space(min_level,
                        max_level,
                        level_candidates,
                        num_intermediate_blocks):
  """Builds the NAS-FPN search space.

  Args:
    min_level: an integer that represents the min feature level of the output
      features.
    max_level: an integer that represents the max feature level of the output
      features.
    level_candidates: the set of candidate levels to be searched.
    num_intermediate_blocks: the number of blocks to be searched.

  Returns:
    a TunableBlockSpecsBuilder object that can be called to build the list of
    BlockSpec for NAS-FPN architecture.
  """
  num_inputs = max_level - min_level + 1
  num_outputs = num_inputs
  # pylint: disable=g-complex-comprehension
  # pylint: disable=g-long-ternary
  intermediate_blocks = [
      pg.Dict(
          level=pg.one_of(level_candidates),
          combine_fn=pg.one_of(['sum', 'attention']),
          input_offsets=pg.sublist_of(
              k=2,
              candidates=list(range(num_inputs + i)),
              choices_distinct=True,
              choices_sorted=True))
      for i in range(num_intermediate_blocks)]

  output_blocks = [
      pg.Dict(
          level=None,
          combine_fn=pg.one_of(['sum', 'attention']),
          input_offsets=pg.sublist_of(
              k=2,
              candidates=list(range(num_inputs + num_intermediate_blocks + i)),
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
      output_blocks=output_blocks,
      output_block_levels=output_levels)
