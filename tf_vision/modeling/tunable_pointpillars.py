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

"""Implementation of tunable PointPillars model."""

import dataclasses
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from absl import logging
from tf_vision import utils
from tf_vision.configs import pointpillars as cfg
import numpy as np
import tensorflow as tf

from official.core import config_definitions as base_cfg
from tf_vision.pointpillars.modeling import layers
from tf_vision.pointpillars.modeling import models
from tf_vision.pointpillars.utils import model_exporter
from official.vision.beta.modeling.layers import detection_generator as detection_generator_lib
from official.vision.beta.modeling.layers import nn_blocks
from official.vision.beta.ops import spatial_transform_ops

# The fixed architecture discovered by NAS.
# Each element represents a BlockSpec:
# (block_fn, block_repeats, kernel_size, expand_ratio, se_ratio, activation_fn,
#  output_filters, level)
_POINTPILLARS_BLOCK_SPECS = {
    'featurizer': (
        ('conv', 1, 1, 1, 0.0, 'relu', 64, 0),
    ),
    'backbone': (
        ('mbconv', 2, 5, 1, 0.0, 'relu', 80, 1),
        ('bottleneck', 3, 5, 3, 0.0, 'relu', 160, 2),
        ('mbconv', 1, 3, 1, 0.0, 'relu', 128, 2),
        ('mbconv', 2, 3, 1, 0.0, 'swish', 192, 3),
        ('bottleneck', 2, 3, 1, 0.0, 'relu', 256, 3),
    ),
    'decoder': (
        ('conv', 3, 5, 1, 0.0, 'relu', 128, 1),
        ('residual', 3, 5, 1, 0.0, 'swish', 128, 2),
        ('conv', 2, 3, 3, 0.0, 'swish', 128, 3),
    ),
}


@dataclasses.dataclass
class BlockSpec():
  """The data class for block spec.

  Attributes:
    block_fn: The str type of the block function, could be one of
      [conv, residual, bottleneck, mbconv].
    block_repeats: An int number of repeat times of the block.
    kernel_size: An int of kernel size of convolution layer.
    expand_ratio: An int of expand_ratio for a mbconv block.
    se_ratio: A float of ratio of the Squeeze-and-Excitation layer for a
      mbconv block.
    activation_fn: The str type of the activation function.
    output_filters: An int number of filters of the block output.
    level: An int number of level where the output would be after applying
      this block.
  """
  block_fn: str = 'conv'
  block_repeats: int = 1
  kernel_size: int = 3
  expand_ratio: int = 1
  se_ratio: float = 0.0
  activation_fn: str = 'relu'
  output_filters: int = 1
  level: int = 0


def build_block_specs(
    block_specs = None
):
  """Build block specs.

  Args:
    block_specs: A dict of {k: v}, k is choices of
      [featurizer, backbone, decoder], v is a sequence of either tuples or
      TunableBlockSpec defined by search space.

  Returns:
    decoded_specs: A dict of {k: v}, v is a sequence of BlockSpec.
  """
  if block_specs is None:
    block_specs = _POINTPILLARS_BLOCK_SPECS
  decoded_specs = {}
  for name, specs in block_specs.items():
    decoded_specs[name] = []
    for s in specs:
      if isinstance(s, Tuple):
        decoded_specs[name].append(BlockSpec(*s))
      else:
        decoded_specs[name].append(
            BlockSpec(block_fn=s.block_fn,
                      block_repeats=s.block_repeats,
                      kernel_size=s.kernel_size,
                      expand_ratio=s.expand_ratio,
                      se_ratio=s.se_ratio,
                      activation_fn=s.activation_fn,
                      output_filters=s.output_filters,
                      level=s.level))
  return decoded_specs


@tf.keras.utils.register_keras_serializable(package='Vision')
class TunablePointPillars(models.PointPillarsModel):
  """The tunable PointPillars class."""

  def __init__(
      self,
      min_level,
      max_level,
      image_size,
      pillars_size,
      anchor_sizes,
      train_batch_size,
      eval_batch_size,
      num_classes,
      num_anchors_per_location,
      num_params_per_anchor,
      attribute_heads,
      detection_generator,
      block_specs,
      use_sync_bn = False,
      **kwargs):
    super().__init__(
        featurizer=None,
        backbone=None,
        decoder=None,
        head=None,
        detection_generator=detection_generator,
        min_level=min_level,
        max_level=max_level,
        image_size=image_size,
        anchor_sizes=anchor_sizes,
    )

    self._min_level = min_level
    self._max_level = max_level
    self._image_size = image_size
    self._pillars_size = pillars_size
    self._anchor_sizes = anchor_sizes
    self._train_batch_size = train_batch_size
    self._eval_batch_size = eval_batch_size
    self._num_classes = num_classes
    self._num_anchors_per_location = num_anchors_per_location
    self._num_params_per_anchor = num_params_per_anchor
    self._attribute_heads = attribute_heads
    self._detection_generator = detection_generator
    self._block_specs = block_specs
    self._use_sync_bn = use_sync_bn

    self._train_batch_dims = self._get_batch_dims(self._train_batch_size)
    self._eval_batch_dims = self._get_batch_dims(self._eval_batch_size)
    self._test_batch_dims = self._get_batch_dims(1)

    input_filters = self._pillars_size[-1]
    ########## Featurizer ##########
    self._featurizer = []
    for block_spec in self._block_specs['featurizer']:
      input_filters, blocks = self._create_blocks(input_filters, 1, block_spec)
      self._featurizer.extend(blocks)

    ########## Backbone ##########
    self._backbone = {
        str(level): [] for level in range(min_level, max_level + 1)
    }
    backbone_output_filters = {}
    for block_spec in self._block_specs['backbone']:
      level = block_spec.level
      strides = 1 if self._backbone[str(level)] else 2
      input_filters, blocks = self._create_blocks(
          input_filters, strides, block_spec)
      self._backbone[str(level)].extend(blocks)
      backbone_output_filters[str(level)] = input_filters

    ########## Decoder ##########
    self._decoder_prehoc = {
        str(level): [] for level in range(min_level, max_level + 1)
    }
    self._decoder_posthoc = {
        str(level): [] for level in range(min_level, max_level + 1)
    }
    for block_spec in self._block_specs['decoder']:
      level = block_spec.level
      input_filters, blocks = self._create_blocks(
          backbone_output_filters[str(level)], 1, block_spec)
      self._decoder_prehoc[str(level)].extend(blocks)
      input_filters, blocks = self._create_blocks(input_filters, 1, block_spec)
      self._decoder_posthoc[str(level)].extend(blocks)

    ########## Head ##########
    self._classifier = tf.keras.layers.Conv2D(
        filters=self._num_classes * self._num_anchors_per_location,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
        bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
    )
    self._box_regressor = tf.keras.layers.Conv2D(
        filters=self._num_params_per_anchor * self._num_anchors_per_location,
        kernel_size=3,
        strides=1,
        padding='same',
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
        bias_initializer=tf.zeros_initializer()
    )
    self._attr_regressor = {}
    for attr in self._attribute_heads:
      self._attr_regressor[attr['name']] = tf.keras.layers.Conv2D(
          filters=attr['size'] * self._num_anchors_per_location,
          kernel_size=3,
          strides=1,
          padding='same',
          kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
          bias_initializer=tf.zeros_initializer()
      )

  def _get_batch_dims(self, batch_size):
    p = self._pillars_size[0]
    batch_dims = np.indices([batch_size, p])[0]
    batch_dims = tf.convert_to_tensor(batch_dims, dtype=tf.int32)
    batch_dims = tf.expand_dims(batch_dims, axis=-1)
    return batch_dims

  def _get_batch_size_and_dims(self, training):
    if training is None:
      batch_size = 1
      batch_dims = self._test_batch_dims
    else:
      if training:
        batch_size = self._train_batch_size
        batch_dims = self._train_batch_dims
      else:
        batch_size = self._eval_batch_size
        batch_dims = self._eval_batch_dims
    return batch_size, batch_dims

  def call(self,
           pillars,
           indices,
           image_shape = None,
           anchor_boxes = None,
           training = None):
    h, w = self._image_size
    p, n, d = self._pillars_size
    assert pillars.get_shape().as_list()[1:] == [p, n, d]

    ########## Featurizer ##########
    # Prepare batch indices.
    batch_size, batch_dims = self._get_batch_size_and_dims(training)
    batch_indices = tf.concat([batch_dims, indices], axis=-1)
    # Generate BEV pseudo image.
    x = pillars
    for block in self._featurizer:
      x = block(x)
    x = tf.reduce_max(x, axis=2, keepdims=False)
    c = x.get_shape().as_list()[-1]
    x = tf.scatter_nd(batch_indices, x, shape=[batch_size, h, w, c])

    ########## Backbone ##########
    # Build bottom-down path.
    feats = {}
    for level in range(self._min_level, self._max_level + 1):
      for block in self._backbone[str(level)]:
        x = block(x)
      feats[str(level)] = x

    ########## Decoder ##########
    # Build pre-hoc features.
    for level in range(self._min_level, self._max_level + 1):
      for block in self._decoder_prehoc[str(level)]:
        feats[str(level)] = block(feats[str(level)])
    # Build top-down path and lateral connections.
    for level in range(self._max_level - 1, self._min_level - 1, -1):
      feat_a = spatial_transform_ops.nearest_upsampling(
          feats[str(level + 1)], 2)
      feat_b = feats[str(level)]
      feats[str(level)] = tf.add(feat_a, feat_b)
    # Build post-hoc features.
    for level in range(self._min_level, self._max_level + 1):
      for block in self._decoder_posthoc[str(level)]:
        feats[str(level)] = block(feats[str(level)])

    ########## Head ##########
    scores = {}
    boxes = {}
    attributes = {attr['name']: {} for attr in self._attribute_heads}
    for level in range(self._min_level, self._max_level + 1):
      scores[str(level)] = self._classifier(feats[str(level)])
      boxes[str(level)] = self._box_regressor(feats[str(level)])
      for attr in self._attribute_heads:
        name = attr['name']
        attributes[name][str(level)] = self._attr_regressor[name](
            feats[str(level)])

    ########## Detection Generator ##########
    return super().generate_outputs(
        raw_scores=scores,
        raw_boxes=boxes,
        raw_attributes=attributes,
        image_shape=image_shape,
        anchor_boxes=anchor_boxes,
        generate_detections=not training)

  def _create_blocks(
      self,
      input_filters,
      strides,
      block_spec):
    block_fn_table = {
        'conv': layers.ConvBlock,
        'residual': nn_blocks.ResidualBlock,
        'bottleneck': nn_blocks.BottleneckBlock,
        'mbconv': nn_blocks.InvertedBottleneckBlock,
    }
    block_fn = block_fn_table[block_spec.block_fn]

    output_filters = block_spec.output_filters
    if block_spec.block_fn == 'residual':
      use_projection = not (input_filters == output_filters and
                            strides == 1)
    elif block_spec.block_fn == 'bottleneck':
      # Since bottleneck uses 4 times as many filters as the argument, we
      # firstly scale it down to make its output filters be output_filters.
      if output_filters % 4 != 0:
        raise ValueError('Output filters of bottleneck block should be '
                         'multiple of 4, but {}'.format(output_filters))
      output_filters = int(output_filters / 4)
      use_projection = not (input_filters == 4 * output_filters and
                            strides == 1)

    blocks = []
    if (block_spec.block_fn == 'residual' or
        block_spec.block_fn == 'bottleneck'):
      blocks.append(block_fn(
          filters=output_filters,
          strides=strides,
          use_projection=use_projection,
          activation=block_spec.activation_fn,
          use_sync_bn=self._use_sync_bn))
    elif block_spec.block_fn == 'conv':
      blocks.append(block_fn(
          filters=output_filters,
          kernel_size=block_spec.kernel_size,
          strides=strides,
          use_sync_bn=self._use_sync_bn,
          activation=block_spec.activation_fn))
    elif block_spec.block_fn == 'mbconv':
      blocks.append(block_fn(
          in_filters=input_filters,
          out_filters=output_filters,
          expand_ratio=block_spec.expand_ratio,
          strides=strides,
          kernel_size=block_spec.kernel_size,
          se_ratio=block_spec.se_ratio,
          activation=block_spec.activation_fn,
          use_sync_bn=self._use_sync_bn))

    for _ in range(1, block_spec.block_repeats):
      if (block_spec.block_fn == 'residual' or
          block_spec.block_fn == 'bottleneck'):
        blocks.append(block_fn(
            filters=output_filters,
            strides=1,  # Fixed to 1
            use_projection=False,
            activation=block_spec.activation_fn,
            use_sync_bn=self._use_sync_bn))
      elif block_spec.block_fn == 'conv':
        blocks.append(block_fn(
            filters=output_filters,
            kernel_size=block_spec.kernel_size,
            strides=1,  # Fixed to 1
            use_sync_bn=self._use_sync_bn,
            activation=block_spec.activation_fn))
      elif block_spec.block_fn == 'mbconv':
        blocks.append(block_fn(
            in_filters=output_filters,
            out_filters=output_filters,
            expand_ratio=block_spec.expand_ratio,
            strides=1,  # Fixed to 1
            kernel_size=block_spec.kernel_size,
            se_ratio=block_spec.se_ratio,
            activation=block_spec.activation_fn,
            use_sync_bn=self._use_sync_bn))
    return block_spec.output_filters, blocks

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    # Since tunable model doesn't have additional components, return an empty
    # dict to override the function of parent class.
    items = dict()
    return items

  def get_config(self):
    config_dict = {
        'min_level': self._min_level,
        'max_level': self._max_level,
        'image_size': self._image_size,
        'pillars_size': self._pillars_size,
        'anchor_sizes': self._anchor_sizes,
        'train_batch_size': self._train_batch_size,
        'eval_batch_size': self._eval_batch_size,
        'num_classes': self._num_classes,
        'num_anchors_per_location': self._num_anchors_per_location,
        'num_params_per_anchor': self._num_params_per_anchor,
        'attribute_heads': self._attribute_heads,
        'detection_generator': self._detection_generator,
        'block_specs': self._block_specs,
        'use_sync_bn': self._use_sync_bn,
    }
    return config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)


def build_tunable_pointpillars(
    model_config,
    train_batch_size,
    eval_batch_size,
    block_specs = None
):
  """Build tunable pointpillars."""

  image_size = (model_config.image.height, model_config.image.width)
  pillars_size = (model_config.pillars.num_pillars,
                  model_config.pillars.num_points_per_pillar,
                  model_config.pillars.num_features_per_point)
  anchor_sizes = [(a.length, a.width) for a in model_config.anchors]
  num_anchors_per_location = (len(anchor_sizes))
  attribute_heads = [
      attr.as_dict() for attr in (model_config.head.attribute_heads or [])
  ]
  generator_config = model_config.detection_generator
  detection_generator = detection_generator_lib.MultilevelDetectionGenerator(
      apply_nms=generator_config.apply_nms,
      pre_nms_top_k=generator_config.pre_nms_top_k,
      pre_nms_score_threshold=generator_config.pre_nms_score_threshold,
      nms_iou_threshold=generator_config.nms_iou_threshold,
      max_num_detections=generator_config.max_num_detections,
      nms_version=generator_config.nms_version,
      use_cpu_nms=generator_config.use_cpu_nms)

  if block_specs is None:
    if model_config.block_specs_json is None:
      logging.warning('No block_specs_json found in model config, '
                      'will use the default block_specs to build model.')
      block_specs = build_block_specs()
    else:
      logging.info('Block_specs_json found.')
      tunable_block_specs = utils.read_tunable_block_specs(
          model_config.block_specs_json)
      block_specs = build_block_specs(tunable_block_specs)
  else:
    logging.info('Preset block_specs found for building model')
  logging.info('Build tunable model with block_specs:\n%s', block_specs)

  return TunablePointPillars(
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      image_size=image_size,
      pillars_size=pillars_size,
      anchor_sizes=anchor_sizes,
      train_batch_size=train_batch_size,
      eval_batch_size=eval_batch_size,
      num_classes=model_config.num_classes,
      num_anchors_per_location=num_anchors_per_location,
      num_params_per_anchor=4,
      attribute_heads=attribute_heads,
      detection_generator=detection_generator,
      block_specs=block_specs,
      use_sync_bn=model_config.norm_activation.use_sync_bn,
  )


def export_inference_graph(
    params,
    checkpoint_path,
    export_dir,
):
  """Exports inference graph for tunable PointPillars model."""
  export_module = TunablePointPillarsModule(params=params, batch_size=1)
  model_exporter.export_inference_graph(
      batch_size=1,
      params=params,
      checkpoint_path=checkpoint_path,
      export_dir=export_dir,
      export_module=export_module)


class TunablePointPillarsModule(model_exporter.PointPillarsModule):
  """Tunable PointPillars model export module."""

  def _build_model(self):
    model = build_tunable_pointpillars(
        model_config=self._params.task.model,
        train_batch_size=1,
        eval_batch_size=1)
    return model
