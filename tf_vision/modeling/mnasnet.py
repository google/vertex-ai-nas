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

"""Contains definitions of MnasNet Networks."""
import dataclasses
from typing import Any, List, Mapping, Optional, Sequence, Tuple

# Standard Imports

import tensorflow as tf

from official.modeling import hyperparams
from official.modeling import tf_utils
from official.vision.beta.modeling.backbones import factory
from official.vision.beta.modeling.backbones import mobilenet
from official.vision.beta.modeling.layers import nn_blocks

layers = tf.keras.layers

# The static MnasNet-A1 architecture discovered by NAS.
# Each element represents a specification of a building block:
# (block_fn, num_repeats, kernel_size, strides, expand_ratio, se_ratio,
# output_filters, is_output).
_MNASNET_A1_BLOCK_SPECS = (
    ('mbconv', 1, 3, 1, 1, 0.0, 16, False),
    ('mbconv', 2, 3, 2, 6, 0.0, 24, True),
    ('mbconv', 3, 5, 2, 3, 0.25, 40, True),
    ('mbconv', 4, 3, 2, 6, 0.0, 80, False),
    ('mbconv', 2, 3, 1, 6, 0.25, 112, True),
    ('mbconv', 3, 5, 2, 6, 0.25, 160, False),
    ('mbconv', 1, 3, 1, 6, 0.0, 320, True),
)

_MNASNET_STRIDES = [1, 2, 2, 2, 1, 2, 1]  # Same as MobileNet-V2.
_MNASNET_IS_OUTPUT = [False, True, True, False, True, False, True]
_STEM_NUM_FILTERS = 32


@dataclasses.dataclass
class BlockSpec:
  """A container class that specifies the block configuration for MnasNet."""
  block_fn: str = ''
  block_repeats: int = 7
  kernel_size: int = 3
  strides: Optional[int] = None
  expand_ratio: float = 1
  se_ratio: float = 0.0
  out_filters: int = 32
  is_output: Optional[bool] = None

  def get_config(self):
    return dataclasses.asdict(self)


def build_block_specs(
    block_specs = None):
  """Decodes and returns specs for a block."""
  # use the static block specs if not given.
  if block_specs is None:
    block_specs = _MNASNET_A1_BLOCK_SPECS

  decoded_specs = []
  for spec in block_specs:
    decoded_specs.append(BlockSpec(*spec))
  return tuple(decoded_specs)


class MnasNet(tf.keras.Model):
  """Creates a MnasNet family model."""

  def __init__(self,
               input_specs = layers.InputSpec(
                   shape=[None, None, None, 3]),
               decoded_specs = build_block_specs(),
               endpoints_num_filters = 1280,
               features_only = False,
               kernel_initializer = 'VarianceScaling',
               kernel_regularizer = None,
               bias_regularizer = None,
               activation = 'relu',
               use_sync_bn = False,
               norm_momentum = 0.99,
               norm_epsilon = 0.001,  # pytype: disable=annotation-type-mismatch  # typed-keras
               **kwargs):
    """Initializes a MnasNet model.

    Args:
      input_specs: A `tf.keras.layers.InputSpec` of the input tensor.
      decoded_specs: A list of block specifications for the MnasNet model
        discovered by NAS.
      endpoints_num_filters: The number of filters for the last head.
      features_only: If true, no final conv layer is applied and only
        intermediate features are in the output.
      kernel_initializer: A `str` for kernel initializer of convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
        Default to None.
      activation: A `str` of name of the activation function.
      use_sync_bn: If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      **kwargs: Additional keyword arguments to be passed.
    """
    self._input_specs = input_specs
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    if use_sync_bn:
      self._norm = layers.experimental.SyncBatchNormalization
    else:
      self._norm = layers.BatchNormalization
    self._features_only = features_only
    self._decoded_specs = decoded_specs
    self._endpoints_num_filters = endpoints_num_filters

    if tf.keras.backend.image_data_format() == 'channels_last':
      bn_axis = -1
    else:
      bn_axis = 1

    # Build MnasNet.
    inputs = tf.keras.Input(shape=input_specs.shape[1:])

    # Build stem.
    x = layers.Conv2D(
        filters=_STEM_NUM_FILTERS,
        kernel_size=3,
        strides=2,
        use_bias=False,
        padding='same',
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)(
            inputs)
    x = self._norm(
        axis=bn_axis, momentum=norm_momentum, epsilon=norm_epsilon)(
            x)
    x = tf_utils.get_activation(activation)(x)

    # Build intermediate blocks.
    endpoints = {}
    endpoint_level = 2

    for i, specs in enumerate(decoded_specs):
      if specs.strides is None:
        specs.strides = _MNASNET_STRIDES[i]
      if specs.is_output is None:
        specs.is_output = _MNASNET_IS_OUTPUT[i]
      x = self._block_group(
          inputs=x, specs=specs, name='block_group_{}'.format(i))
      if specs.is_output:
        endpoints[str(endpoint_level)] = x
        endpoint_level += 1

    # Build output specs for downstream tasks.
    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}

    # Build the final conv for classification if needed.
    if not features_only:
      x = layers.Conv2D(
          filters=endpoints_num_filters,
          kernel_size=1,
          strides=1,
          use_bias=False,
          padding='same',
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)(
              x)
      x = self._norm(
          axis=bn_axis, momentum=norm_momentum, epsilon=norm_epsilon)(
              x)
      endpoints[str(endpoint_level)] = tf_utils.get_activation(activation)(x)

    super(MnasNet, self).__init__(inputs=inputs, outputs=endpoints, **kwargs)

  def _block_group(self,
                   inputs,
                   specs,
                   name = 'block_group'):
    """Creates one group of blocks for the MnasNet model.

    Args:
      inputs: A `tf.Tensor` of size `[batch, channels, height, width]`.
      specs: The specifications for one inverted bottleneck block group.
      name: A `str` name for the block.

    Returns:
      The output `tf.Tensor` of the block layer.
    """
    if specs.block_fn == 'mbconv':
      block_fn = nn_blocks.InvertedBottleneckBlock
    elif specs.block_fn == 'conv':
      block_fn = mobilenet.Conv2DBNBlock
    else:
      raise ValueError('Block func {} not supported.'.format(specs.block_fn))

    in_filters = inputs.shape.as_list(
    )[-1] if tf.keras.backend.image_data_format(
    ) == 'channels_last' else inputs.shape.as_list()[1]

    if specs.block_fn == 'conv':
      x = block_fn(
          filters=specs.out_filters,
          kernel_size=specs.kernel_size,
          strides=specs.strides,
          activation=self._activation,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon)(
              inputs)
    elif specs.block_fn == 'mbconv':
      x = block_fn(
          in_filters=in_filters,
          out_filters=specs.out_filters,
          expand_ratio=specs.expand_ratio,
          strides=specs.strides,
          kernel_size=specs.kernel_size,
          se_ratio=specs.se_ratio,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=self._activation,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon)(
              inputs)

    for _ in range(1, specs.block_repeats):
      if specs.block_fn == 'conv':
        x = block_fn(
            filters=specs.out_filters,
            kernel_size=specs.kernel_size,
            strides=1,  # Fix strides to 1.
            activation=self._activation,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            use_sync_bn=self._use_sync_bn,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon)(
                x)
      elif specs.block_fn == 'mbconv':
        x = block_fn(
            in_filters=specs.out_filters,  # Set 'in_filters' to 'out_filters'.
            out_filters=specs.out_filters,
            expand_ratio=specs.expand_ratio,
            strides=1,  # Fix strides to 1.
            kernel_size=specs.kernel_size,
            se_ratio=specs.se_ratio,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activation=self._activation,
            use_sync_bn=self._use_sync_bn,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon)(
                x)

    return tf.identity(x, name=name)

  def get_config(self):
    return {
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'decoded_specs': self._decoded_specs,
        'endpoints_num_filters': self._endpoints_num_filters,
        'features_only': self._features_only,
    }

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('mnasnet')
def build_mnasnet(
    input_specs,
    backbone_config,
    norm_activation_config,
    l2_regularizer = None):
  """Builds the MnasNet network."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'mnasnet', (f'Inconsistent backbone type '
                                      f'{backbone_type}')

  return MnasNet(
      input_specs=input_specs,
      decoded_specs=build_block_specs(),
      endpoints_num_filters=backbone_cfg.endpoints_num_filters,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)
