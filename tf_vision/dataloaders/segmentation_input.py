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

"""Data parser and processing for segmentation datasets."""
from typing import Optional
import tensorflow as tf

from nas_lib.augmentation_2d import policies_tf2 as policies

from official.vision.beta.dataloaders import segmentation_input
from official.vision.beta.ops import preprocess_ops


class Parser(segmentation_input.Parser):
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

  def _prepare_image_and_label(self, data):
    """Prepare normalized image and label."""
    image = tf.io.decode_image(data['image/encoded'], channels=3)
    label = tf.io.decode_image(
        data['image/segmentation/class/encoded'], channels=1)
    height = data['image/height']
    width = data['image/width']
    image = tf.reshape(image, (height, width, 3))

    # Apply augmentation if aug_policy is present.
    if self._aug_policy:
      # Perform augmentation according to the pyglove policy.
      policy = policies.get_policy_from_str(self._aug_policy)
      image, label = policy(image, segmentation_mask=label)

    label = tf.reshape(label, (1, height, width))
    label = tf.cast(label, tf.float32)
    # Normalizes image with mean and std pixel values.
    image = preprocess_ops.normalize_image(image)
    return image, label
