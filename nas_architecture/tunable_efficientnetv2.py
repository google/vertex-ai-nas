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
"""Implementation of tunable efficientnet v2 models."""
import pyglove as pg
import tensorflow.compat.v1 as tf

from tf1.detection.modeling.architecture import nn_blocks
from tf1.detection.modeling.architecture import nn_ops
from nas_architecture import tunable_efficientnetv2_search_space  # pylint: disable=unused-import

_EFFICIENTNETV2_BLOCK_SPECS = [
    (1, 'fused_mbconv', 1, 3, 0.0, 16, 'swish'),
    (2, 'fused_mbconv', 4, 3, 0.0, 32, 'swish'),
    (2, 'fused_mbconv', 4, 3, 0.0, 48, 'swish'),
    (3, 'mbconv', 4, 3, 0.25, 96, 'swish'),
    (5, 'mbconv', 6, 3, 0.25, 112, 'swish'),
    (8, 'mbconv', 6, 3, 0.25, 192, 'swish'),
]


class BlockSpec(object):
  """A container class that specifies the block configuration for EfficientNet."""

  def __init__(self, num_repeats, block_type, expand_ratio, kernel_size,
               se_ratio, output_filters, act_fn):
    self.num_repeats = num_repeats
    self.block_type = block_type
    self.expand_ratio = expand_ratio
    self.kernel_size = kernel_size
    self.se_ratio = se_ratio
    self.output_filters = output_filters
    self.act_fn = act_fn


def build_static_block_specs(tunable_block_specs):
  """Builds the static EfficientNetV2 BlockSpec.

  Args:
    tunable_block_specs: a list of TunableBlockSpec that specifies the
      EfficientNetV2 block configs and can be tuned for architecture search.

  Returns:
    a list of static BlockSpec that can be loaded by EfficientNetV2 model.
  """
  # pylint: disable=g-complex-comprehension
  if not tunable_block_specs:
    return None

  return [
      BlockSpec(b.num_repeats, b.block_type, b.expand_ratio, b.kernel_size,
                b.se_ratio, b.output_filters, b.act_fn)
      for b in tunable_block_specs
  ]
  # pylint: enable=g-complex-comprehension


class EfficientNetV2(object):
  """Class to build EfficientNetV2 family models."""

  def __init__(self,
               block_specs=None,
               batch_norm_activation=nn_ops.BatchNormActivation(),
               data_format='channels_last'):
    """EfficientNetV2 initialization function.

    Args:
      block_specs: a list of BlockSpec objects that specifies the EfficientNetV2
        network. By default, the previously discovered EfficientNetV2 is used.
      batch_norm_activation: an operation that includes a batch normalization
        layer followed by an optional activation layer.
      data_format: An optional string from: "channels_last", "channels_first".
        Defaults to "channels_last".
    """
    if not block_specs:
      tf.logging.info('Building static efficientnet_v2.')
      self._block_specs = [BlockSpec(*b) for b in _EFFICIENTNETV2_BLOCK_SPECS]
    else:
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
    with tf.variable_scope('efficientnetv2'):
      x = nn_ops.conv2d_fixed_padding(
          inputs=x,
          filters=32,
          kernel_size=3,
          strides=2,
          data_format=self._data_format)
      x = tf.identity(x, 'initial_conv')
      x = self._batch_norm_activation(x, is_training=is_training)

      all_strides = [1, 2, 2, 2, 1, 2]
      endpoints = []
      for i, block_spec in enumerate(self._block_specs):
        bn_act = nn_ops.BatchNormActivation(activation=block_spec.act_fn)
        with tf.variable_scope('block_{}'.format(i)):
          for j in range(block_spec.num_repeats):
            strides = 1 if j > 0 else all_strides[i]

            if block_spec.block_type == 'conv':
              x = nn_ops.conv2d_fixed_padding(
                  inputs=x,
                  filters=block_spec.output_filters,
                  kernel_size=block_spec.kernel_size,
                  strides=strides,
                  data_format=self._data_format)
              x = bn_act(x, is_training=is_training)
            elif block_spec.block_type == 'mbconv':
              x_shape = x.get_shape().as_list()
              in_filters = (
                  x_shape[1]
                  if self._data_format == 'channel_first' else x_shape[-1])
              x = nn_blocks.mbconv_block(
                  inputs=x,
                  in_filters=in_filters,
                  out_filters=block_spec.output_filters,
                  expand_ratio=block_spec.expand_ratio,
                  strides=strides,
                  kernel_size=block_spec.kernel_size,
                  se_ratio=block_spec.se_ratio,
                  batch_norm_activation=bn_act,
                  data_format=self._data_format,
                  is_training=is_training)
            elif block_spec.block_type == 'fused_mbconv':
              x_shape = x.get_shape().as_list()
              in_filters = (
                  x_shape[1]
                  if self._data_format == 'channel_first' else x_shape[-1])
              x = nn_blocks.fused_mbconv_block(
                  inputs=x,
                  in_filters=in_filters,
                  out_filters=block_spec.output_filters,
                  expand_ratio=block_spec.expand_ratio,
                  strides=strides,
                  kernel_size=block_spec.kernel_size,
                  se_ratio=block_spec.se_ratio,
                  batch_norm_activation=bn_act,
                  data_format=self._data_format,
                  is_training=is_training)
            else:
              raise ValueError('Un-supported block_type `{}`!'.format(
                  block_spec.block_type))
          x = tf.identity(x, 'endpoints')
          endpoints.append(x)

    return {2: endpoints[1], 3: endpoints[2], 4: endpoints[4], 5: endpoints[5]}


def build_efficientnetv2(tunable_block_specs,
                         batch_norm_activation=nn_ops.BatchNormActivation()):
  """EfficientNetV2 network architecture."""
  return EfficientNetV2(
      block_specs=build_static_block_specs(tunable_block_specs),
      batch_norm_activation=batch_norm_activation,
      data_format='channels_last')


def tunable_efficientnetv2_builder(
    block_specs_json, batch_norm_activation=nn_ops.BatchNormActivation()):
  """Builds the EfficientNetV2 network."""
  if block_specs_json.endswith('.json'):
    with tf.gfile.GFile(block_specs_json) as f:
      block_specs_json = f.read()
  tunable_block_specs = pg.from_json_str(block_specs_json)
  return build_efficientnetv2(
      tunable_block_specs=tunable_block_specs,
      batch_norm_activation=batch_norm_activation)
