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
"""Implementation of Mnas tunable model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, List, Optional

from tf_vision import utils
from tf_vision.modeling import mnasnet
import tensorflow as tf

from nas_architecture.tunable_mnasnet_search_space import TunableBlockSpec
from official.modeling import hyperparams
from official.vision.beta.modeling.backbones import factory


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
          block_repeats=b.num_repeats,
          block_fn=b.block_fn,
          expand_ratio=b.expand_ratio,
          kernel_size=b.kernel_size,
          se_ratio=b.se_ratio,
          out_filters=b.output_filters)
      for i, b in enumerate(tunable_block_specs)
  ]
  # pylint: enable=g-complex-comprehension


class TunableMnasNet(mnasnet.MnasNet):
  """Tunable MnasNet network architecture."""

  def __init__(
      self,
      input_specs,
      tunable_block_specs = None,
      endpoints_num_filters = 1280,
      features_only = False,
      activation = 'relu',
      use_sync_bn = False,
      norm_momentum = 0.99,
      norm_epsilon = 0.001,
      l2_regularizer = None):

    if tunable_block_specs is None:
      block_specs = mnasnet.build_block_specs()
    else:
      block_specs = build_static_block_specs(tunable_block_specs)
    super().__init__(
        input_specs=input_specs,
        decoded_specs=block_specs,
        endpoints_num_filters=endpoints_num_filters,
        features_only=features_only,
        kernel_regularizer=l2_regularizer,
        activation=activation,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon)


@factory.register_backbone_builder('tunable_mnasnet')
def build_tunable_mnasnet(
    input_specs,
    backbone_config,
    norm_activation_config,
    l2_regularizer = None):  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds the tunable MnasNet network."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'tunable_mnasnet', (f'Inconsistent backbone type '
                                              f'{backbone_type}')

  tunable_block_specs = utils.read_tunable_block_specs(
      backbone_cfg.block_specs_json) if backbone_cfg.block_specs_json else None

  return TunableMnasNet(
      input_specs=input_specs,
      tunable_block_specs=tunable_block_specs,
      endpoints_num_filters=backbone_cfg.endpoints_num_filters,
      features_only=backbone_cfg.features_only,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      l2_regularizer=l2_regularizer)
