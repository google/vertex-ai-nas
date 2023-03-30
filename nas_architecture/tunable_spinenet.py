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
"""Implementation of SpineNet tunable model.

X. Du, T-Y. Lin, P. Jin, G. Ghiasi, M. Tan, Y. Cui, Q. V. Le, X. Song
SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization
https://arxiv.org/abs/1912.05027
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import pyglove as pg

import tensorflow.compat.v1 as tf

from tf1.detection.modeling.architecture import nn_ops
from tf1.detection.modeling.architecture import spinenet
from tf1.detection.modeling.architecture import spinenet_mbconv
from nas_architecture import tunable_spinenet_search_space


def build_static_block_specs_json(tunable_block_specs_json):
  """Builds the static SpineNet block specs JSON string.

  Args:
    tunable_block_specs_json: a JSON string that specifies the SpineNet
      block configuration and can be tuned for architecture search.

  Returns:
    a JSON string that can be loaded by the SpineNet static model.
  """
  # pylint: disable=g-complex-comprehension
  static_block_specs_list = [
      (b.level_base + b.level_offset,
       b.block_fn,
       tuple(b.input_offsets),
       b.is_output)
      for b in pg.from_json_str(tunable_block_specs_json)]
  return json.dumps(static_block_specs_list)
  # pylint: enable=g-complex-comprehension


def build_static_block_specs(tunable_block_specs):
  """Builds the static SpineNet BlockSpec.

  Args:
    tunable_block_specs: a list of TunableBlockSpec that specifies the SpineNet
      block configuration and can be tuned for architecture search.

  Returns:
    a list of static BlockSpec that can be loaded by the SpineNet static model.
  """
  # pylint: disable=g-complex-comprehension
  return [
      spinenet.BlockSpec(
          b.level_base + b.level_offset,
          b.block_fn,
          tuple(b.input_offsets),
          b.is_output)
      for b in tunable_block_specs]
  # pylint: enable=g-complex-comprehension


@pg.members([
    ('min_level', pg.typing.Int(min_value=1),
     'The min feature level of the output features.'),
    ('max_level', pg.typing.Int(min_value=1),
     'The max feature level of the output features.'),
    ('tunable_block_specs',
     pg.typing.List(
         pg.typing.Object(
             tunable_spinenet_search_space.TunableBlockSpec)).noneable(),
     'a list of tunable block spec to define the SpineNet network.'),
    ('endpoints_num_filters', pg.typing.Int(default=256),
     'The number of filters of the output endpoints.'),
    ('resample_alpha', pg.typing.Float(default=0.5), 'Resample alpha.'),
    ('use_native_resize_op', pg.typing.Bool(default=False),
     'Whether to use the native tf.image.resize op.'),
    ('block_repeats', pg.typing.Int(default=1, min_value=1),
     'The number of blocks to repeat.'),
    ('filter_size_scale', pg.typing.Float(default=1.0, min_value=0.0),
     'The multiplier of the number of filters for each block.'),
    ('activation', pg.typing.Enum(default='swish', values=['relu', 'swish']),
     'The type of the activation function.'),
    ('batch_norm_activation',
     pg.typing.Any(default=nn_ops.BatchNormActivation()),
     'Batchnorm followed by an optional activation layer'),
    ('init_drop_connect_rate',
     pg.typing.Float().noneable(), 'Drop connect rate.'),
])
class TunableSpineNet(pg.Object):
  """SpineNet network architecture."""

  def __call__(self, inputs, is_training=False):
    if self.tunable_block_specs is None:
      block_specs = spinenet.build_block_specs()
    else:
      block_specs = build_static_block_specs(self.tunable_block_specs)
    spinenet_fn = spinenet.SpineNet(
        min_level=self.min_level,
        max_level=self.max_level,
        block_specs=block_specs,
        endpoints_num_filters=self.endpoints_num_filters,
        resample_alpha=self.resample_alpha,
        use_native_resize_op=self.use_native_resize_op,
        block_repeats=self.block_repeats,
        filter_size_scale=self.filter_size_scale,
        activation=self.activation,
        batch_norm_activation=self.batch_norm_activation,
        init_drop_connect_rate=self.init_drop_connect_rate,
        data_format='channels_last')
    return spinenet_fn(inputs, is_training=is_training)


@pg.members([
    ('tunable_block_specs',
     pg.typing.List(
         pg.typing.Object(tunable_spinenet_search_space.TunableBlockSpec)),
     'a list of tunable block spec to define the SpineNetv2 network.'),
    ('min_level', pg.typing.Int(min_value=1),
     'The min feature level of the output features.'),
    ('max_level', pg.typing.Int(min_value=1),
     'The max feature level of the output features.'),
    ('endpoints_num_filters', pg.typing.Int(default=48),
     'The number of filters of the output endpoints.'),
    ('block_repeats', pg.typing.Int(default=1, min_value=1),
     'The number of blocks to repeat.'),
    ('filter_size_scale', pg.typing.Float(default=1.0, min_value=0.0),
     'The multiplier of the number of filters for each block.'),
    ('use_native_resize_op', pg.typing.Bool(default=False),
     'Whether to use the native tf.image.resize op.'),
    ('activation', pg.typing.Enum(default='swish', values=['relu', 'swish']),
     'The type of the activation function.'),
    ('se_ratio', pg.typing.Float(default=0.2),
     'Squeeze and excitation ratio.'),
    ('batch_norm_activation',
     pg.typing.Any(default=nn_ops.BatchNormActivation()),
     'Batchnorm followed by an optional acitvation layer'),
    ('init_drop_connect_rate',
     pg.typing.Float().noneable(), 'Drop connect rate.'),
])
class TunableSpineNetMBConv(pg.Object):
  """SpineNetMBConv network architecture."""

  def __call__(self, inputs, is_training=False):
    if self.tunable_block_specs is None:
      block_specs = spinenet.build_block_specs()
    else:
      block_specs = build_static_block_specs(self.tunable_block_specs)
    spinenet_mbconv_fn = spinenet_mbconv.SpineNetMBConv(
        min_level=self.min_level,
        max_level=self.max_level,
        endpoints_num_filters=self.endpoints_num_filters,
        block_repeats=self.block_repeats,
        filter_size_scale=self.filter_size_scale,
        use_native_resize_op=self.use_native_resize_op,
        block_specs=block_specs,
        activation=self.activation,
        se_ratio=self.se_ratio,
        batch_norm_activation=self.batch_norm_activation,
        init_drop_connect_rate=self.init_drop_connect_rate,
        data_format='channels_last')
    return spinenet_mbconv_fn(inputs, is_training=is_training)


def tunable_spinenet_builder(min_level,
                             max_level,
                             block_specs_json=None,
                             endpoints_num_filters=256,
                             resample_alpha=0.5,
                             use_native_resize_op=False,
                             block_repeats=1,
                             filter_size_scale=1.0,
                             activation='swish',
                             batch_norm_activation=nn_ops.BatchNormActivation(
                                 activation='swish'),
                             init_drop_connect_rate=None):
  """Builds the tunable SpineNet network."""
  if block_specs_json is None:
    if min_level != 3 or max_level != 7:
      raise ValueError(
          'The default SpineNet architecture expects `min_level` = 3 and '
          '`min_level` = 7, got `min_level` = {}, `max_leve` = {}'
          .format(min_level, max_level))
    tunable_block_specs = None
  else:
    if block_specs_json.endswith('.json'):
      with tf.gfile.GFile(block_specs_json) as f:
        block_specs_json = f.read()
    tunable_block_specs = pg.from_json_str(block_specs_json)
  return TunableSpineNet(
      min_level=min_level,
      max_level=max_level,
      tunable_block_specs=tunable_block_specs,
      endpoints_num_filters=endpoints_num_filters,
      resample_alpha=resample_alpha,
      use_native_resize_op=use_native_resize_op,
      block_repeats=block_repeats,
      filter_size_scale=filter_size_scale,
      activation=activation,
      batch_norm_activation=batch_norm_activation,
      init_drop_connect_rate=init_drop_connect_rate)


def tunable_spinenet_mbconv_builder(min_level,
                                    max_level,
                                    block_specs_json=None,
                                    endpoints_num_filters=256,
                                    block_repeats=1,
                                    filter_size_scale=1.0,
                                    use_native_resize_op=False,
                                    activation='swish',
                                    se_ratio=0.2,
                                    batch_norm_activation=
                                    nn_ops.BatchNormActivation(
                                        activation='swish'),
                                    init_drop_connect_rate=None):
  """Builds the SpineNet-MBConv network."""
  if block_specs_json is None:
    if min_level != 3 or max_level != 7:
      raise ValueError(
          'The default SpineNet architecture expects `min_level` = 3 and '
          '`min_level` = 7, got `min_level` = {}, `max_leve` = {}'
          .format(min_level, max_level))
    tunable_block_specs = None
  else:
    if block_specs_json.endswith('.json'):
      with tf.gfile.GFile(block_specs_json) as f:
        block_specs_json = f.read()
    tunable_block_specs = pg.from_json_str(block_specs_json)
  return TunableSpineNetMBConv(
      min_level=min_level,
      max_level=max_level,
      tunable_block_specs=tunable_block_specs,
      endpoints_num_filters=endpoints_num_filters,
      use_native_resize_op=use_native_resize_op,
      block_repeats=block_repeats,
      filter_size_scale=filter_size_scale,
      activation=activation,
      se_ratio=se_ratio,
      batch_norm_activation=batch_norm_activation,
      init_drop_connect_rate=init_drop_connect_rate)
