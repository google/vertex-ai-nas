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
"""Implementation of MnasNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow.compat.v1 as tf

from tf1.detection.modeling.architecture import mnasnet_constants
from tf1.detection.modeling.architecture import nn_blocks
from tf1.detection.modeling.architecture import nn_ops


class BlockSpec(object):
  """A container class that specifies the block configuration for MnasNet."""

  def __init__(self,
               num_repeats,
               block_fn,
               expand_ratio,
               kernel_size,
               se_ratio,
               output_filters):
    self.num_repeats = num_repeats
    self.block_fn = block_fn
    self.expand_ratio = expand_ratio
    self.kernel_size = kernel_size
    self.se_ratio = se_ratio
    self.output_filters = output_filters


def build_block_specs(block_specs=None):
  """Builds the list of BlockSpec objects for MnasNet."""
  if not block_specs:
    block_specs = mnasnet_constants.MNASNET_A1_BLOCK_SPECS
  if len(block_specs) != mnasnet_constants.MNASNET_NUM_BLOCKS:
    raise ValueError('The block_specs of MnasNet must be a length {} list.'
                     .format(mnasnet_constants.MNASNET_NUM_BLOCKS))
  logging.info('Building MnasNet block specs: %s', block_specs)
  return [BlockSpec(*b) for b in block_specs]


class MnasNet(object):
  """Class to build MnasNet family models."""

  def __init__(self,
               block_specs=build_block_specs(),
               batch_norm_activation=nn_ops.BatchNormActivation(),
               data_format='channels_last'):
    """MnasNet initialization function.

    Args:
      block_specs: a list of BlockSpec objects that specifies the MnasNet
        network. By default, the previously discovered MnasNet-A1 is used.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
      data_format: An optional string from: "channels_last", "channels_first".
        Defaults to "channels_last".
    """
    self._block_specs = block_specs
    self._batch_norm_activation = batch_norm_activation
    self._data_format = data_format

  def __call__(self, images, is_training=False):
    """Generate a multiscale feature pyramid.

    Args:
      images: The input image tensor.
      is_training: `bool` if True, the model is in training mode.

    Returns:
      a `dict` containing `int` keys for continuous feature levels
      [min_level, min_level + 1, ..., max_level]. The values are corresponding
      features with shape [batch_size, height_l, width_l,
      endpoints_num_filters].
    """
    x = images
    with tf.variable_scope('mnasnet'):
      x = nn_ops.conv2d_fixed_padding(
          inputs=x,
          filters=32,
          kernel_size=3,
          strides=2,
          data_format=self._data_format)
      x = tf.identity(x, 'initial_conv')
      x = self._batch_norm_activation(x, is_training=is_training)

      endpoints = []
      for i, block_spec in enumerate(self._block_specs):
        with tf.variable_scope('block_{}'.format(i)):
          for j in range(block_spec.num_repeats):
            strides = 1 if j > 0 else mnasnet_constants.MNASNET_STRIDES[i]

            if block_spec.block_fn == 'conv':
              x = nn_ops.conv2d_fixed_padding(
                  inputs=x,
                  filters=block_spec.output_filters,
                  kernel_size=block_spec.kernel_size,
                  strides=strides,
                  data_format=self._data_format)
              x = self._batch_norm_activation(x, is_training=is_training)
            elif block_spec.block_fn == 'mbconv':
              x_shape = x.get_shape().as_list()
              in_filters = (x_shape[1] if self._data_format == 'channel_first'
                            else x_shape[-1])
              x = nn_blocks.mbconv_block(
                  inputs=x,
                  in_filters=in_filters,
                  out_filters=block_spec.output_filters,
                  expand_ratio=block_spec.expand_ratio,
                  strides=strides,
                  kernel_size=block_spec.kernel_size,
                  se_ratio=block_spec.se_ratio,
                  batch_norm_activation=self._batch_norm_activation,
                  data_format=self._data_format,
                  is_training=is_training)
            else:
              raise ValueError(
                  'Un-supported block_fn `{}`!'.format(block_spec.block_fn))
          x = tf.identity(x, 'endpoints')
          endpoints.append(x)

    return {2: endpoints[1], 3: endpoints[2], 4: endpoints[4], 5: endpoints[6]}
