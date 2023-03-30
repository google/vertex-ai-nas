# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tunable NAS-FPN for architecture search.

Golnaz Ghiasi, Tsung-Yi Lin, Ruoming Pang, Quoc V. Le.
NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection.
https://arxiv.org/abs/1904.07392. CVPR 2019.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
from absl import logging
import pyglove as pg
import six
import tensorflow.compat.v1 as tf

from nas_lib.lib.layers import nn_ops as nn_ops_v1
from nas_lib.nasfpn import nasfpn as nasfpn_v1
from tf1.detection.modeling.architecture import nasfpn
from tf1.detection.modeling.architecture import nn_ops
from nas_architecture import tunable_nasfpn_search_space


_STATIC_MODEL_NODES_STR = '''[
    {"_type": "nas_lib.nasfpn.nasfpn.GlobalAttentionNode",
     "node_type": "intermediate", "level": 4, "input_offsets": [1, 3]},
    {"_type": "nas_lib.nasfpn.nasfpn.SumNode",
     "node_type": "intermediate", "level": 4, "input_offsets": [1, 5]},
    {"_type": "nas_lib.nasfpn.nasfpn.SumNode",
     "node_type": "output", "level": 3, "input_offsets": [0, 6]},
    {"_type": "nas_lib.nasfpn.nasfpn.SumNode",
     "node_type": "output", "level": 4, "input_offsets": [6, 7]},
    {"_type": "nas_lib.nasfpn.nasfpn.GlobalAttentionNode",
     "node_type": "output", "level": 5, "input_offsets": [7, 8]},
    {"_type": "nas_lib.nasfpn.nasfpn.GlobalAttentionNode",
     "node_type": "output", "level": 7, "input_offsets": [6, 9]},
    {"_type": "nas_lib.nasfpn.nasfpn.GlobalAttentionNode",
     "node_type": "output", "level": 6, "input_offsets": [9, 10]}
]'''


def model_v1(min_level=3,
             max_level=7,
             feature_dims=256,
             num_repeats=7,
             use_separable_conv=False,
             dropblock=nn_ops_v1.DropBlock(),
             batch_norm_relu=nn_ops_v1.BatchNormRelu(),
             nodes=None):
  """Create Feature Pyramid Network using FPN nodes."""
  if nodes is None:
    nodes = _STATIC_MODEL_NODES_STR
  if isinstance(nodes, six.string_types):
    if nodes.endswith('.json'):
      with tf.gfile.GFile(nodes) as f:
        nodes = f.read()
    nodes = pg.from_json_str(nodes)
  logging.info('Creating tunable NAS-FPN with nodes: %r', nodes)
  return nasfpn_v1.Fpn(
      min_level=min_level,
      max_level=max_level,
      feature_dims=feature_dims,
      num_repeats=num_repeats,
      feature_resampler=nasfpn_v1.FpnFeatureResampler(
          batch_norm_relu=batch_norm_relu),
      cell=nasfpn_v1.FpnCellV1(
          use_separable_conv=use_separable_conv,
          dropblock=dropblock,
          batch_norm_relu=batch_norm_relu,
          nodes=nodes))


def build_static_block_specs_json(tunable_block_specs_json):
  """Builds the static NAS-FPN block specs JSON string.

  Args:
    tunable_block_specs_json: a JSON string that specifies the NAS-FPN
      block configuration and can be tuned for architecture search.

  Returns:
    a JSON string that can be loaded by the NAS-FPN static model.
  """
  # pylint: disable=g-complex-comprehension
  static_block_specs_list = [
      (b.level, b.combine_fn, tuple(b.input_offsets), b.is_output)
      for b in pg.from_json_str(tunable_block_specs_json)]
  return json.dumps(static_block_specs_list)
  # pylint: enable=g-complex-comprehension


def build_static_block_specs(tunable_block_specs):
  """Builds the static NAS-FPN BlockSpec.

  Args:
    tunable_block_specs: a list of TunableBlockSpec that specifies the NAS-FPN
      block configuration and can be tuned for architecture search.

  Returns:
    a list of static BlockSpec that can be loaded by the NAS-FPN static model.
  """
  # pylint: disable=g-complex-comprehension
  return [
      nasfpn.BlockSpec(
          b.level, b.combine_fn, tuple(b.input_offsets), b.is_output)
      for b in tunable_block_specs]
  # pylint: enable=g-complex-comprehension


@pg.members([
    ('min_level', pg.typing.Int(min_value=1),
     'The min feature level of the output features.'),
    ('max_level', pg.typing.Int(min_value=1),
     'The max feature level of the output features.'),
    ('tunable_block_specs',
     pg.typing.List(
         pg.typing.Object(tunable_nasfpn_search_space.TunableBlockSpec)),
     'a list of tunable block spec to define the NAS-FPN network.'),
    ('fpn_feat_dims', pg.typing.Int(default=256),
     'The number of filters of the pyramid features.'),
    ('num_repeats', pg.typing.Int(default=7, min_value=1),
     'The number of times to repeat the NAS-FPN structure.'),
    ('use_separable_conv', pg.typing.Bool(default=False),
     'Whether to use the separable convolutions.'),
    ('activation', pg.typing.Enum(default='swish', values=['relu', 'swish']),
     'The type of the activation function.'),
    ('batch_norm_activation',
     pg.typing.Any(default=nn_ops.BatchNormActivation()),
     'Batchnorm followed by an optional activation layer'),
])
class TunableNasfpn(pg.Object):
  """NAS-FPN network architecture."""

  def __call__(self, multilevel_features, is_training=False):
    nasfpn_fn = nasfpn.Nasfpn(
        min_level=self.min_level,
        max_level=self.max_level,
        block_specs=build_static_block_specs(self.tunable_block_specs),
        fpn_feat_dims=self.fpn_feat_dims,
        num_repeats=self.num_repeats,
        use_separable_conv=self.use_separable_conv,
        dropblock=nn_ops.Dropblock(),
        block_fn='conv',
        block_repeats=1,
        activation=self.activation,
        batch_norm_activation=self.batch_norm_activation,
        init_drop_connect_rate=None,
        data_format='channels_last')
    return nasfpn_fn(multilevel_features, is_training=is_training)


def tunable_nasfpn_builder(min_level,
                           max_level,
                           block_specs_json,
                           fpn_feat_dims=256,
                           num_repeats=7,
                           use_separable_conv=False,
                           activation='relu',
                           batch_norm_activation=nn_ops.BatchNormActivation(
                               activation='relu')):
  """Builds the tunable NAS-FPN network."""
  if block_specs_json.endswith('.json'):
    with tf.gfile.GFile(block_specs_json) as f:
      block_specs_json = f.read()
  tunable_block_specs = pg.from_json_str(block_specs_json)
  return TunableNasfpn(
      min_level=min_level,
      max_level=max_level,
      tunable_block_specs=tunable_block_specs,
      fpn_feat_dims=fpn_feat_dims,
      num_repeats=num_repeats,
      use_separable_conv=use_separable_conv,
      activation=activation,
      batch_norm_activation=batch_norm_activation)
