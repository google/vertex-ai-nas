# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Implementation of tunable PointPillars search space."""

import pyglove as pg

# The minimum level for model output.
_MODEL_MIN_LEVEL = 1


@pg.members([
    ('block_fn',
     pg.typing.Enum(default='conv',
                    values=['conv', 'residual', 'bottleneck', 'mbconv']),
     'Type of block function.'),
    ('block_repeats', pg.typing.Int(default=1, min_value=1),
     'Number of repeats of the block.'),
    ('kernel_size',
     pg.typing.Int(default=1, min_value=1),
     'Kernel size of convolution layers.'),
    ('expand_ratio', pg.typing.Int(default=1, min_value=1),
     'Expand ratio for a mbconv block.'),
    ('se_ratio', pg.typing.Float(default=0.0, min_value=0.0),
     'Ratio of the Squeeze-and-Excitation layer for a mbconv block.'),
    ('activation_fn',
     pg.typing.Enum(default='swish', values=['relu', 'swish']),
     'Type of activation function.'),
    ('output_filters', pg.typing.Int(default=64, min_value=1),
     'Number of output filters.'),
    ('level', pg.typing.Int(default=0, min_value=0),
     'The level the output would be at after applying this block')
])
class TunableBlockSpec(pg.Object):
  """The tunable specifications of a PointPillars block."""


@pg.functor([
    ('featurizer_blocks',
     pg.typing.List(
         pg.typing.Dict([
             ('block_repeats', pg.typing.Int()),
             ('activation_fn', pg.typing.Str()),
             ('output_filters', pg.typing.Int()),
         ]))),
    ('backbone_blocks',
     pg.typing.List(
         pg.typing.Dict([
             ('block_fn', pg.typing.Str()),
             ('block_repeats', pg.typing.Int()),
             ('kernel_size', pg.typing.Int()),
             ('expand_ratio', pg.typing.Int()),
             ('se_ratio', pg.typing.Float()),
             ('activation_fn', pg.typing.Str()),
             ('output_filters', pg.typing.Int()),
             ('level', pg.typing.Int()),
         ]))),
    ('decoder_blocks',
     pg.typing.List(
         pg.typing.Dict([
             ('block_fn', pg.typing.Str()),
             ('block_repeats', pg.typing.Int()),
             ('kernel_size', pg.typing.Int()),
             ('expand_ratio', pg.typing.Int()),
             ('se_ratio', pg.typing.Float()),
             ('activation_fn', pg.typing.Str()),
             ('output_filters', pg.typing.Int()),
             ('level', pg.typing.Int()),
         ]))),
])
def build_tunable_block_specs(featurizer_blocks,  # pytype: disable=not-instantiable  # always-use-return-annotations
                              backbone_blocks,
                              decoder_blocks):
  """Build tunable blocks specs."""
  # pylint: disable=g-complex-comprehension
  return {
      'featurizer': [
          TunableBlockSpec(
              block_repeats=b.block_repeats,
              activation_fn=b.activation_fn,
              output_filters=b.output_filters)
          for b in featurizer_blocks],
      'backbone': [
          TunableBlockSpec(
              block_fn=b.block_fn,
              block_repeats=b.block_repeats,
              kernel_size=b.kernel_size,
              expand_ratio=b.expand_ratio,
              se_ratio=b.se_ratio,
              activation_fn=b.activation_fn,
              output_filters=b.output_filters,
              level=b.level)
          for b in backbone_blocks],
      'decoder': [
          TunableBlockSpec(
              block_fn=b.block_fn,
              block_repeats=b.block_repeats,
              kernel_size=b.kernel_size,
              expand_ratio=b.expand_ratio,
              se_ratio=b.se_ratio,
              activation_fn=b.activation_fn,
              output_filters=b.output_filters,
              level=b.level)
          for b in decoder_blocks]
  }
  # pylint: enable=g-complex-comprehension


_NUM_BLOCKS_AT_LEVEL = {
    1: 1,
    2: 2,
    3: 2,
}


def pointpillars_search_space(max_level = 3,
                              base_output_filters = 64):
  """The search space of PointPillars."""
  # Only support fixed min_level now.
  min_level = _MODEL_MIN_LEVEL

  o = base_output_filters
  featurizer_blocks = [
      pg.Dict(
          block_repeats=pg.one_of([1, 2, 3]),
          activation_fn=pg.one_of(['swish', 'relu']),
          output_filters=pg.one_of([int(o * 0.75), o, int(o * 1.25)])
      )
  ]

  backbone_blocks = []
  decoder_blocks = []
  for level in range(min_level, max_level + 1):
    o = base_output_filters * (2 ** (level - 1))
    for _ in range(_NUM_BLOCKS_AT_LEVEL[level]):
      backbone_blocks.append(
          pg.Dict(
              block_fn=pg.one_of(['residual', 'bottleneck', 'mbconv']),
              block_repeats=pg.one_of([1, 2, 3]),
              kernel_size=pg.one_of([3, 5]),
              expand_ratio=pg.one_of([1, 3, 6]),
              se_ratio=pg.one_of([0.0, 0.25]),
              activation_fn=pg.one_of(['swish', 'relu']),
              output_filters=pg.one_of([int(o * 0.75), o, int(o * 1.25)]),
              level=level
          )
      )

    decoder_blocks.append(
        pg.Dict(
            block_fn=pg.one_of(['conv', 'residual', 'bottleneck', 'mbconv']),
            block_repeats=pg.one_of([1, 2, 3]),
            kernel_size=pg.one_of([3, 5]),
            expand_ratio=pg.one_of([1, 3, 6]),
            se_ratio=pg.one_of([0.0, 0.25]),
            activation_fn=pg.one_of(['swish', 'relu']),
            output_filters=base_output_filters * 2,
            level=level
        )
    )

  return build_tunable_block_specs(featurizer_blocks,
                                   backbone_blocks,
                                   decoder_blocks)
