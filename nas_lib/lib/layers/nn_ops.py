# Copyright 2019 Google Research. All Rights Reserved.
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
"""Neural network operations commonly shared by the architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyglove as pg
import tensorflow.compat.v1 as tf


@pg.members([
    ('momentum', pg.typing.Float(min_value=0.0, max_value=1.0, default=0.997),
     'Momentum used in batch normalization layer.'),
    ('epsilon', pg.typing.Float(min_value=0.0, default=1e-4),
     'Epsilon used in batch normalizaton layer.'),
    ('trainable', pg.typing.Bool(default=True),
     'If True, add batch normalization variables to TRAINABLE_VARIABLES '
     'collection, otherwise freeze batch normalization layer.')
])
class BatchNormRelu(pg.Object):
  """Batch normaization plus an optional relu layer."""

  def __call__(
      self, inputs, relu=True, init_zero=False, is_training=False, name=None):
    if init_zero:
      gamma_initializer = tf.zeros_initializer()
    else:
      gamma_initializer = tf.ones_initializer()

    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        momentum=self.momentum,
        epsilon=self.epsilon,
        center=True,
        scale=True,
        training=(is_training and self.trainable),
        trainable=self.trainable,
        fused=True,
        gamma_initializer=gamma_initializer,
        name=name)

    if relu:
      inputs = tf.nn.relu(inputs)
    return inputs


@pg.members([
    ('keep_prob', pg.typing.Float(min_value=0.0, max_value=1.0).noneable(),
     'Keep probability of this drop block.'),
    ('size', pg.typing.Int().noneable(),
     'Size of this drop block.'),
    ('data_format', pg.typing.Enum(
        'channels_last', ['channels_first', 'channels_last', 'NCHW', 'NHWC']),
     'Format of inputs.')
])
class DropBlock(pg.Object):
  """DropBlock: a regularization method for convolutional neural networks.

    DropBlock is a form of structured dropout, where units in a contiguous
    region of a feature map are dropped together. DropBlock works better than
    dropout on convolutional layers due to the fact that activation units in
    convolutional layers are spatially correlated.
    See https://arxiv.org/pdf/1810.12890.pdf for details.
  """

  def __call__(self, inputs, is_training=False):
    """Builds Dropblock layer.

    Args:
      inputs: `Tensor` input tensor.
      is_training: `bool` if True, the model is in training mode.

    Returns:
      A version of input tensor with DropBlock applied.
    """
    if not is_training or self.keep_prob is None:
      return inputs

    tf.logging.info('Applying DropBlock: size {},'
                    'inputs.shape {}'.format(self.size, inputs.shape))

    data_format = self.data_format
    if data_format == 'channels_first':
      data_format = 'NCHW'
    elif data_format == 'channels_last':
      data_format = 'NHWC'

    if data_format == 'NHWC':
      _, height, width, _ = inputs.get_shape().as_list()
    else:
      _, _, height, width = inputs.get_shape().as_list()

    assert height is not None
    assert width is not None

    total_size = width * height
    dropblock_size = min(self.size, min(width, height))
    # Seed_drop_rate is the gamma parameter of DropBlcok.
    seed_drop_rate = (
        1.0 - self.keep_prob) * total_size / dropblock_size**2 / (
            (width - self.size + 1) *
            (height - self.size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = tf.meshgrid(tf.range(width), tf.range(height))
    valid_block = tf.logical_and(
        tf.logical_and(w_i >= int(dropblock_size // 2),
                       w_i < width - (dropblock_size - 1) // 2),
        tf.logical_and(h_i >= int(dropblock_size // 2),
                       h_i < width - (dropblock_size - 1) // 2))

    if data_format == 'NHWC':
      valid_block = tf.reshape(valid_block, [1, height, width, 1])
    else:
      valid_block = tf.reshape(valid_block, [1, 1, height, width])

    randnoise = tf.random_uniform(inputs.shape, dtype=tf.float32)
    valid_block = tf.cast(valid_block, dtype=tf.float32)
    seed_keep_rate = tf.cast(1 - seed_drop_rate, dtype=tf.float32)
    block_pattern = (1 - valid_block + seed_keep_rate + randnoise) >= 1
    block_pattern = tf.cast(block_pattern, dtype=tf.float32)

    if data_format == 'NHWC':
      ksize = [1, self.size, self.size, 1]
    else:
      ksize = [1, 1, self.size, self.size]
    block_pattern = -tf.nn.max_pool(
        -block_pattern,
        ksize=ksize,
        strides=[1, 1, 1, 1],
        padding='SAME',
        data_format=data_format)

    percent_ones = tf.cast(tf.reduce_sum(block_pattern), tf.float32) / tf.cast(
        tf.size(block_pattern), tf.float32)

    inputs = inputs / tf.cast(percent_ones, inputs.dtype) * tf.cast(
        block_pattern, inputs.dtype)
    return inputs
