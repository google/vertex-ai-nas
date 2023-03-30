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
"""Medical 3D search spaces."""

import pyglove as pg

from nas_architecture import tunable_nasfpn_search_space


def get_search_space(search_space):
  if search_space == 'static_unet':
    return static_unet_search_space()
  elif search_space == 'tunable_unet_encoder':
    return unet_encoder_search_space()
  elif search_space == 'tunable_unet_nasfpn':
    return unet_nasfpn_search_space()
  else:
    raise ValueError('Unknown search-space: %s' % search_space)


@pg.members([
    ('dummy_param', pg.typing.Int(default=1), 'Dummy parameter.'),
])
class DummySpecBuilder(pg.Object):
  """Define a dummy spec for UNet."""
  pass


@pg.members([
    ('blocks', pg.typing.List(
        pg.typing.Dict([
            ('block_type', pg.typing.Str(default='basic')),
            ('num_filters', pg.typing.Int()),
            ('num_repeats', pg.typing.Int()),
            ('kernel_size', pg.typing.Int(default=3)),
            ('id_skip', pg.typing.Bool(default=False)),
            ('se_ratio', pg.typing.Float(default=0.0)),
        ])), 'List of backbone block specs.'),
    ('activation', pg.typing.Str(default='relu')),
])
class EncoderSpecBuilder(pg.Object):
  """Define an encoder spec for UNet."""
  pass


def static_unet_search_space():
  """Returns a dummy search-space for Static-UNet."""
  return DummySpecBuilder(dummy_param=pg.one_of([1]))


def unet_encoder_search_space():
  """Returns an encoder search space for UNet."""
  def block(num_filters):
    return pg.Dict(
        block_type=pg.one_of(['basic_block', 'bottleneck_block']),
        num_filters=pg.one_of(
            [int(num_filters * 0.75), num_filters,
             int(num_filters * 1.25)]),
        num_repeats=pg.one_of([1, 2, 3]),
        kernel_size=pg.one_of([3, 5]),
        id_skip=pg.one_of([False, True]),
        se_ratio=pg.one_of([0.0, 0.25, 0.5]))

  return EncoderSpecBuilder(
      blocks=[
          block(32),
          block(64),
          block(128),
          block(256),
      ],
      activation=pg.one_of(['relu', 'swish']))


def unet_nasfpn_search_space():
  """Returns NAS-FPN search space."""
  min_level = 1
  max_level = 4
  num_intermediate_blocks = 2
  return tunable_nasfpn_search_space.nasfpn_search_space(
      min_level=min_level,
      max_level=max_level,
      level_candidates=list(range(min_level, max_level + 1)),
      num_intermediate_blocks=num_intermediate_blocks)
