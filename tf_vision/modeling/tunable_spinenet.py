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
"""Tunable SpineNet model implementation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Optional

from tf_vision import utils
import tensorflow as tf

from official.modeling import hyperparams
from official.vision.beta.modeling.backbones import factory
from official.vision.beta.modeling.backbones import spinenet
from official.vision.beta.modeling.backbones import spinenet_mobile


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
          level=b.level_base + b.level_offset,
          block_fn=b.block_fn,
          input_offsets=tuple(b.input_offsets),
          is_output=b.is_output) for b in tunable_block_specs
  ]
  # pylint: enable=g-complex-comprehension


class TunableSpineNet(spinenet.SpineNet):
  """Tunable SpineNet network architecture."""

  def __init__(
      self,
      input_specs,
      tunable_block_specs = None,
      min_level = 3,
      max_level = 7,
      endpoints_num_filters = 256,
      resample_alpha = 0.5,
      block_repeats = 1,
      filter_size_scale = 1.0,
      init_stochastic_depth_rate = 0.0,
      activation = 'relu',
      use_sync_bn = False,
      norm_momentum = 0.99,
      norm_epsilon = 0.001,
      l2_regularizer = None):

    if tunable_block_specs is None:
      block_specs = spinenet.build_block_specs()
    else:
      block_specs = build_static_block_specs(tunable_block_specs)
    super().__init__(
        input_specs=input_specs,
        min_level=min_level,
        max_level=max_level,
        block_specs=block_specs,
        endpoints_num_filters=endpoints_num_filters,
        resample_alpha=resample_alpha,
        block_repeats=block_repeats,
        filter_size_scale=filter_size_scale,
        init_stochastic_depth_rate=init_stochastic_depth_rate,
        kernel_regularizer=l2_regularizer,
        activation=activation,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon)


@factory.register_backbone_builder('tunable_spinenet')
def build_tunable_spinenet(
    input_specs,
    backbone_config,
    norm_activation_config,
    l2_regularizer = None):  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds the tunable SpineNet network."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'tunable_spinenet', (f'Inconsistent backbone type '
                                               f'{backbone_type}')

  model_id = backbone_cfg.model_id
  if model_id not in spinenet.SCALING_MAP:
    raise ValueError(
        'SpineNet-{} is not a valid architecture.'.format(model_id))

  if backbone_cfg.block_specs_json is None:
    if backbone_cfg.min_level != 3 or backbone_cfg.max_level != 7:
      raise ValueError(
          'The default SpineNet architecture expects `min_level` = 3 and '
          '`max_level` = 7, got `min_level` = {}, `max_level` = {}'.format(
              backbone_cfg.min_level, backbone_cfg.max_level))
    tunable_block_specs = None
  else:
    tunable_block_specs = utils.read_tunable_block_specs(
        backbone_cfg.block_specs_json)

  return TunableSpineNet(
      input_specs=input_specs,
      tunable_block_specs=tunable_block_specs,
      min_level=backbone_cfg.min_level,
      max_level=backbone_cfg.max_level,
      endpoints_num_filters=backbone_cfg.endpoints_num_filters,
      resample_alpha=backbone_cfg.resample_alpha,
      block_repeats=backbone_cfg.block_repeats,
      filter_size_scale=backbone_cfg.filter_size_scale,
      init_stochastic_depth_rate=backbone_cfg.init_stochastic_depth_rate,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      l2_regularizer=l2_regularizer)


class TunableSpineNetMBConv(spinenet_mobile.SpineNetMobile):
  """Tunable SpineNetMBConv network architecture."""

  def __init__(
      self,
      input_specs,
      tunable_block_specs = None,
      min_level = 3,
      max_level = 7,
      endpoints_num_filters = 256,
      se_ratio = 0.2,
      block_repeats = 1,
      filter_size_scale = 1.0,
      init_stochastic_depth_rate = 0.0,
      activation = 'relu',
      use_sync_bn = False,
      norm_momentum = 0.99,
      norm_epsilon = 0.001,
      l2_regularizer = None):

    if tunable_block_specs is None:
      block_specs = spinenet_mobile.build_block_specs()
    else:
      block_specs = build_static_block_specs(tunable_block_specs)
    super().__init__(
        input_specs=input_specs,
        min_level=min_level,
        max_level=max_level,
        block_specs=block_specs,
        endpoints_num_filters=endpoints_num_filters,
        se_ratio=se_ratio,
        block_repeats=block_repeats,
        filter_size_scale=filter_size_scale,
        init_stochastic_depth_rate=init_stochastic_depth_rate,
        kernel_regularizer=l2_regularizer,
        activation=activation,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon)


@factory.register_backbone_builder('tunable_spinenet_mbconv')
def build_tunable_spinenet_mbconv(
    input_specs,
    backbone_config,
    norm_activation_config,
    l2_regularizer = None):  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds the tunable SpineNetMBConv network."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'tunable_spinenet_mbconv', (
      f'Inconsistent backbone type '
      f'{backbone_type}')

  model_id = backbone_cfg.model_id
  if model_id not in spinenet_mobile.SCALING_MAP:
    raise ValueError(
        'SpineNet-{} is not a valid architecture.'.format(model_id))

  if backbone_cfg.block_specs_json is None:
    if backbone_cfg.min_level != 3 or backbone_cfg.max_level != 7:
      raise ValueError(
          'The default SpineNet architecture expects `min_level` = 3 and '
          '`max_level` = 7, got `min_level` = {}, `max_level` = {}'.format(
              backbone_cfg.min_level, backbone_cfg.max_level))
    tunable_block_specs = None
  else:
    tunable_block_specs = utils.read_tunable_block_specs(
        backbone_cfg.block_specs_json)

  return TunableSpineNetMBConv(
      input_specs=input_specs,
      tunable_block_specs=tunable_block_specs,
      min_level=backbone_cfg.min_level,
      max_level=backbone_cfg.max_level,
      endpoints_num_filters=backbone_cfg.endpoints_num_filters,
      se_ratio=backbone_cfg.se_ratio,
      block_repeats=backbone_cfg.block_repeats,
      filter_size_scale=backbone_cfg.filter_size_scale,
      init_stochastic_depth_rate=backbone_cfg.init_stochastic_depth_rate,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      l2_regularizer=l2_regularizer)
