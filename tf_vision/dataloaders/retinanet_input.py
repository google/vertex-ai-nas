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

"""Data parser and processing for RetinaNet.

Parse image and ground truths in a dataset to training targets and package them
into (image, labels) tuple for RetinaNet.
"""
from typing import Optional
import tensorflow as tf

from nas_lib.augmentation_2d import policies_tf2 as policies
from official.vision.beta.dataloaders import retinanet_input
from official.vision.beta.ops import anchor
from official.vision.beta.ops import box_ops
from official.vision.beta.ops import preprocess_ops

NAMED_AUTOAUG_POLICIES = ('v0', 'v1', 'v2', 'v3')


class Parser(retinanet_input.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self, aug_policy = None, **kwargs):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      aug_policy: An optional augmentation policy to use. This can be a JSON
        string for an pyglove augmentation policy object. An empty string
        indicates no augmentation policy.
      **kwargs: Additional arguments for base class.
    """
    super().__init__(**kwargs)
    self._aug_policy = aug_policy

  def _parse_train_data(self, data):
    """Parses data for training and evaluation."""
    classes = data['groundtruth_classes']
    boxes = data['groundtruth_boxes']
    is_crowds = data['groundtruth_is_crowd']

    # Skips annotations with `is_crowd` = True.
    if self._skip_crowd_during_training:
      num_groundtrtuhs = tf.shape(input=classes)[0]
      with tf.control_dependencies([num_groundtrtuhs, is_crowds]):
        indices = tf.cond(
            pred=tf.greater(tf.size(input=is_crowds), 0),
            true_fn=lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
            false_fn=lambda: tf.cast(tf.range(num_groundtrtuhs), tf.int64))
      classes = tf.gather(classes, indices)
      boxes = tf.gather(boxes, indices)

    # Gets original image and its size.
    image = data['image']

    # Apply augmentation if aug_policy is present.
    if self._aug_policy:
      if self._aug_policy in NAMED_AUTOAUG_POLICIES:
        # Create a glove policy for certain named autoaugment policies.
        policy = policies.autoaugment_detection_policy(self._aug_policy)
      else:
        # Decode the policy from a glove object JSON str.
        policy = policies.get_policy_from_str(self._aug_policy)
      image, boxes = policy(image, bounding_boxes=boxes)

    image_shape = tf.shape(input=image)[0:2]

    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image)

    # Flips image randomly during training.
    if self._aug_rand_hflip:
      image, boxes, _ = preprocess_ops.random_horizontal_flip(image, boxes)

    # Converts boxes from normalized coordinates to pixel coordinates.
    boxes = box_ops.denormalize_boxes(boxes, image_shape)

    # Resizes and crops image.
    image, image_info = preprocess_ops.resize_and_crop_image(
        image,
        self._output_size,
        padded_size=preprocess_ops.compute_padded_size(self._output_size,
                                                       2**self._max_level),
        aug_scale_min=self._aug_scale_min,
        aug_scale_max=self._aug_scale_max)
    image_height, image_width, _ = image.get_shape().as_list()

    # Resizes and crops boxes.
    image_scale = image_info[2, :]
    offset = image_info[3, :]
    boxes = preprocess_ops.resize_and_crop_boxes(boxes, image_scale,
                                                 image_info[1, :], offset)
    # Filters out ground truth boxes that are all zeros.
    indices = box_ops.get_non_empty_box_indices(boxes)
    boxes = tf.gather(boxes, indices)
    classes = tf.gather(classes, indices)

    # Assigns anchors.
    input_anchor = anchor.build_anchor_generator(
        min_level=self._min_level,
        max_level=self._max_level,
        num_scales=self._num_scales,
        aspect_ratios=self._aspect_ratios,
        anchor_size=self._anchor_size)
    anchor_boxes = input_anchor(image_size=(image_height, image_width))
    anchor_labeler = anchor.AnchorLabeler(self._match_threshold,
                                          self._unmatched_threshold)
    (cls_targets, box_targets, _, cls_weights,
     box_weights) = anchor_labeler.label_anchors(
         anchor_boxes, boxes, tf.expand_dims(classes, axis=1))

    # Casts input image to desired data type.
    image = tf.cast(image, dtype=self._dtype)

    # Packs labels for model_fn outputs.
    labels = {
        'cls_targets': cls_targets,
        'box_targets': box_targets,
        'anchor_boxes': anchor_boxes,
        'cls_weights': cls_weights,
        'box_weights': box_weights,
        'image_info': image_info,
    }
    return image, labels
