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
"""Tunable NAS-FPN for architecture search.

Golnaz Ghiasi, Tsung-Yi Lin, Ruoming Pang, Quoc V. Le.
NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection.
https://arxiv.org/abs/1904.07392. CVPR 2019.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Optional, Mapping

from tf_vision import utils
import tensorflow as tf

from official.modeling import hyperparams
from official.vision.beta.modeling.decoders import factory
from official.vision.beta.modeling.decoders import nasfpn


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


class TunableNasfpn(nasfpn.NASFPN):
  """Tunable NAS-FPN network architecture."""

  def __init__(
      self,
      input_specs,
      tunable_block_specs = None,
      min_level = 3,
      max_level = 7,
      num_filters = 256,
      num_repeats = 7,
      use_separable_conv = False,
      activation = 'relu',
      use_sync_bn = False,
      norm_momentum = 0.99,
      norm_epsilon = 0.001,
      l2_regularizer = None):

    if tunable_block_specs is None:
      block_specs = nasfpn.build_block_specs()
    else:
      block_specs = build_static_block_specs(tunable_block_specs)
    super().__init__(
        input_specs=input_specs,
        min_level=min_level,
        max_level=max_level,
        block_specs=block_specs,
        num_filters=num_filters,
        num_repeats=num_repeats,
        use_separable_conv=use_separable_conv,
        activation=activation,
        use_sync_bn=use_sync_bn,
        norm_momentum=norm_momentum,
        norm_epsilon=norm_epsilon,
        kernel_regularizer=l2_regularizer,
    )


@factory.register_decoder_builder('tunable_nasfpn')
def build_tunable_nasfpn(
    input_specs,
    model_config,
    l2_regularizer = None):  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds the tunable NAS-FPN network."""
  decoder_type = model_config.decoder.type
  decoder_cfg = model_config.decoder.get()
  norm_activation_config = model_config.norm_activation
  assert decoder_type == 'tunable_nasfpn', (f'Inconsistent decoder type '
                                            f'{decoder_type}')

  tunable_block_specs = utils.read_tunable_block_specs(
      decoder_cfg.block_specs_json) if decoder_cfg.block_specs_json else None

  return TunableNasfpn(
      input_specs=input_specs,
      tunable_block_specs=tunable_block_specs,
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      num_filters=decoder_cfg.num_filters,
      num_repeats=decoder_cfg.num_repeats,
      use_separable_conv=decoder_cfg.use_separable_conv,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      l2_regularizer=l2_regularizer)
