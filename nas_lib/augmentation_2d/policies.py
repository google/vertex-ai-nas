# Lint: python2, python3
# Copyright 2020 Google Research. All Rights Reserved.
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
"""Image augmentation policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Callable, Dict, List, Optional, Text, Tuple, Union
import numpy as np
import pyglove as pg
import tensorflow.compat.v1 as tf

from nas_lib.augmentation_2d import ops
from nas_lib.augmentation_2d import policies_search_space

# Constants from research code.
MAX_AUGMENT_MAGNITUDE = 10
POSTERIZE_BITS = 4
ENHANCE_FACTOR_MULT = 1.8
ENHANCE_FACTOR_ADD = 0.1
SOLARIZE_FACTOR_ADD = 110
MAX_SOLARIZE_THRESHOLD = 256
MAX_PAD_SIZE_DEFAULT = 100
MAX_SHIFT_DEFAULT = 250
MAX_SHEAR = 0.3
MAX_ROTATE_DEGREES = 30.0

# Commonly used types.
TensorPair = Tuple[tf.Tensor, tf.Tensor]
TensorOrTensorPair = Union[tf.Tensor, TensorPair]


class AugmentPolicy(pg.Object):
  """Interface for augment policy.

  An AugmentPolicy defines how to transform data. Currently, policies
  can support the following tasks: Image classification, image deteciton,
  and imaged segmentation.

  A policy can be as simple as transforming the data through one augmentation
  function. A policy can also be composed of other policies, allowing for more
  complex funcionality, like AutoAugment or RandAugment policies.

  Policies are called by passing in an image plus any necssary ground truth
  data, depending on the task. Here is an example showing how to call a policy
  for classification, detection, and segmentation:

    >>> image, bboxes, seg_mask = ... # Your existing data.
    >>> p = MyAugmentationPolicy()  # A policy that supports all tasks.
    >>> image_aug = p(image)  # Classification, only augment the image.
    >>> image_aug, bboxes_aug = p(image, bounding_boxes=bboxes) # Detection.
    >>> image_aug, seg_mask_aug = p(image,
    >>>   segmentation_mask=seg_mask) # Segmentation.

  Sub-classes can implement the following methods to add support for a
  particular augmentation task. Otherwise, attempting to call a policy
  that does not support a particular task will raise an exception.
  * _augment - implementing this will add support for classification.
  * _augment_with_bounding_boxes - implementing this will add support
    for detection.
  * _augment_with_segmentation_mask - implementing this will add support
    for segmentation.
  """
  K_BB = 'bounding_boxes'  # Key for ground truth Bounding Boxes.
  K_SEG = 'segmentation_mask'  # Key for ground truth Segmentation Masks.

  def __call__(self,
               image,
               **kwargs):
    """Call the augmentation policy.

    Augmentation policies can support any combination of these tasks:
      Classification, Detection, or Segmentation.

    The task is determined by the inputs passed to a call.
      Classification: Only the image is passed in. The ground
        truth is assumed to not change and therefore does not
        need to be passed in.
      Detection: The image and ground truth bounding boxes are
        passed in. The bounding boxes must be passed in as
        a keyword argument "bounding_boxes=".
      Segmentation: The image and ground truth segmentation mask are
        passed in. The segmentation mask must be passed in as
        a keyword argument "segmentation_mask=". The mask must have the
        same width and heigh as the image as shape [H, W] or [H, W, 1]
        (The extra dimension to signify 1 channel is optional).
    If a policy does not support a specific task, then a NotImplementedError
    will be raised.

    Example:
      >>> image, bboxes, seg_mask = ...  # Your existing data.
      >>> p = MyAugmentationPolicy()  # A policy that supports all tasks.
      >>> image_aug = p(image)  # Classification, only augment the image.
      >>> image_aug, bboxes_aug = p(image, bounding_boxes=bboxes)  # Detection.
      >>> image_aug, seg_mask_aug = p(image,
      >>>   segmentation_mask=seg_mask)  # Segmentation.

    Args:
      image: Input image tensor.
      **kwargs: Ground truth can be passed in as 'bounding_boxes' for detection
       or 'segmentation_mask' for segmentation.

    Returns:
      The augmented data as one of: (image), (image, bounding_boxes),
        (image, segmentation_mask), depending on which ground truth
        format was passed in.

    Raises:
      ValueError: If more than one ground truth items are passed in.
    """
    # Collect any ground truth items and check that at most 1 is present.
    bounding_boxes = kwargs.pop(AugmentPolicy.K_BB, None)
    segmentation_mask = kwargs.pop(AugmentPolicy.K_SEG, None)
    all_ground_truth = [bounding_boxes, segmentation_mask]
    num_present_ground_truth = sum(x is not None for x in all_ground_truth)
    if num_present_ground_truth > 1:
      raise ValueError('Augment policy called with too many '
                       'ground truth items!')

    if bounding_boxes is not None:
      return self._augment_with_bounding_boxes(image, bounding_boxes)
    elif segmentation_mask is not None:
      return self._augment_with_segmentation_mask(image, segmentation_mask)
    else:
      # Classification.
      return self._augment(image)

  def _format_result_for_call(
      self, result,
      **orig_kwargs):
    """Reformat a result into the (image, **kwargs) call input format.

    For this method, the original kwargs are passed in to determine
    what keys to use (if any) for the resulting kwargs.

    This method is helpful to run multiple calls within a policy.

    Args:
      result: A result from a previous call to a policy. For classificaiton
        this is just the image. For detection this is a tuple of
        (image, bounding_boxes). For segmentation this is a tuple of
        (image, segmentation_mask).
      **orig_kwargs: The original kwargs used for the first call to a policy.

    Returns:
      A tuple of the original image tensor and the resulting kwargs dict.
      The resulting kwargs will be empty for classification, contain
      a 'bounding_boxes' entry for detection, or a 'segmentation_mask' entry
      for segmentation.
    """
    if AugmentPolicy.K_BB in orig_kwargs:
      image, bounding_boxes = result
      return image, {AugmentPolicy.K_BB: bounding_boxes}
    elif AugmentPolicy.K_SEG in orig_kwargs:
      image, segmentation_mask = result
      return image, {AugmentPolicy.K_SEG: segmentation_mask}
    else:
      return result, {}

  def _augment(self, image):
    """Augments an image for Classification.

    Sub-class policies should implement this method to add
    suport for classification.

    Args:
      image: Input image tensor.

    Returns:
      The resulting image after augmentation, which has the same
      shape as the input tensor.

    Raises:
      NotImplementedError: If classification augmentation is not supported
        for a policy.
    """
    raise NotImplementedError

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection.

    Sub-class policies should implement this method to add
    suport for detection.

    Args:
      image: Input image tensor.
      bounding_boxes: Input bounding boxes as (N, 4) shape tensor.

    Returns:
      The resulting image and bounding_boxes after augmentation, each with
      the same shape as the corresponding input.

    Raises:
      NotImplementedError: If detection augmentation is not supported
        for a policy.
    """
    raise NotImplementedError

  def _augment_with_segmentation_mask(
      self, image,
      segmentation_mask):
    """Augments an image and ground truth segmentation mask for Segmentation.

    Sub-class policies should implement this method to add
    suport for segmentation.

    Args:
      image: Input image tensor.
      segmentation_mask: Input segmentation mask tensor.

    Returns:
      The resulting image and segmentation mask after augmentation, each with
      the same shape as the corresponding input.

    Raises:
      NotImplementedError: If segmentation augmentation is not supported
        for a policy.
    """
    raise NotImplementedError


class GroundTruthPassThroughPolicy(AugmentPolicy):
  """Base class for policies that pass through ground truth data unchanged.

  This is helpful for augmentation operations that transform only the image,
  while any ground truth labels remain valid.

  A class inheriting from GroundTruthPassThroughPolicy should implement
  the `_augment` function. This function will be used to transform
  the image for all tasks, while simply passing through any ground truth
  data unchanged during augmentation.
  """

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection.

    Here the policiy's classification augmentation method is used
    to augment the image, while the bounding boxes are passed
    through unchanged.

    Args:
      image: Input image tensor.
      bounding_boxes: Input bounding boxes as (N, 4) shape tensor.

    Returns:
      The resulting image and bounding_boxes after augmentation, each with
      the same shape as the corresponding input. The bounding_boxes are
      passed through unchanged from the input.

    """
    image = self._augment(image)
    return image, bounding_boxes

  def _augment_with_segmentation_mask(
      self, image,
      segmentation_mask):
    """Augments an image and ground truth segmentation mask for Segmentation.

    Here the policiy's classification augmentation method is used
    to augment the image, while the segmentation mask is passed
    through unchanged.

    Args:
      image: Input image tensor.
      segmentation_mask: Input segmentation mask tensor.

    Returns:
      The resulting image and segmentation_mask after augmentation, each with
      the same shape as the corresponding input. The segmentation_mask is
      passed through unchanged from the input.
    """
    image = self._augment(image)
    return image, segmentation_mask


def declare_policy(init_arg_list, new_members=None):
  """Class decorator to declare augment policy."""
  return pg.members(
      new_members or [],
      metadata={'init_arg_list': init_arg_list})


class Identity(AugmentPolicy):
  """Identity policy."""

  def _augment(self, image):
    """Returns the image unchanged."""
    return image

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection."""
    return image, bounding_boxes

  def _augment_with_segmentation_mask(
      self, image,
      segmentation_mask):
    """Returns the image and segmentation_mask unchanged."""
    return image, segmentation_mask


#
# Policy mix-ins.
#


@pg.members([
    ('magnitude', pg.typing.Int(min_value=0, max_value=MAX_AUGMENT_MAGNITUDE),
     'Augment magnitude.')
])
class SupportAugmentMagnitude(pg.Object):
  """Mix-in for supporting augment magnitude."""


@pg.members([
    ('prob', pg.typing.Float(min_value=0., max_value=1.),
     'Augment probability.')
])
class SupportAugmentProbability(pg.Object):
  """Mix-in for supporting augment with probability."""


@pg.members([('max_pad_size',
              pg.typing.Int(min_value=0, default=MAX_PAD_SIZE_DEFAULT),
              'Max pad size.')])
class SupportMaxPadSize(pg.Object):
  """Mix-in for supporting max pad size."""


@pg.members([
    ('replace', pg.typing.List(
        pg.typing.Int(min_value=0), size=3, default=[128, 128, 128]),
     'Replaced value for images.'),
    ('mask_replace',
     pg.typing.Int(min_value=0, default=255),
     'Replaced value for segmentation masks.')
])
class NeedValueReplacement(pg.Object):
  """Mix-in for providing replacement values."""


#
# Augmentation policies.
#


@declare_policy(
    ['children'],
    [
        ('children', pg.typing.List(pg.typing.Object(AugmentPolicy)),
         'Child policies to apply sequentially.')
    ])
class Sequential(AugmentPolicy):
  """Sequential policy."""

  def __call__(self,
               image,
               **kwargs):
    """Sequentially calls all children augmentation policies."""
    for child in self.children:
      result = child(image, **kwargs)
      image, kwargs = self._format_result_for_call(result, **kwargs)
    return result


@declare_policy(
    ['candidates', 'num_choices', 'probs', 'seed'],
    [
        ('candidates', pg.typing.List(pg.typing.Object(AugmentPolicy)),
         'Candidate policies to sample.'),
        ('num_choices', pg.typing.Int(default=1, min_value=1),
         'Number of policies to sample.'),
        ('probs', pg.typing.List(
            pg.typing.Float(min_value=0.0, max_value=1.0)).noneable(),
         'Probabability distribution used to sample candidates. '
         'If provided, this list must be the same size as the candidates. '
         'If not provided, then a uniform distribution over all candidates '
         'will be used.'),
        ('replacement', pg.typing.Bool(default=True),
         'True to sample with replacement.'),
        ('seed', pg.typing.Int().noneable(),
         'Random seed for sampling.')
    ])
class Choice(AugmentPolicy):
  """A policy for sampling candidate policies."""

  def _on_bound(self):
    """Event that is triggered after init and when members are bound.

    For Choice, we validate the "probs" member.

    Raises:
      ValeuError: If probs is supplied and the length does not match
        the length of the candidates.
      ValueError: If probs is supplied and does not sum to 1.0.
      ValueError: If replacement=False and num_choices is greater
        than the number of candidates.
    """
    # Collect probability numpy array.
    num_candidates = len(self.candidates)
    if self.probs is not None:
      num_probs = len(self.probs)
      if num_probs != num_candidates:
        raise ValueError('Length of probs does not match length '
                         'of candidates. {} != {}'.format(
                             num_probs, num_candidates))
      if not np.isclose(np.sum(self.probs), 1.0):
        raise ValueError('Probs must sum to 1.0.')
      self._probs = self.probs
    else:
      num_probs = num_candidates
      self._probs = np.ones(num_candidates) / float(num_candidates)
    self._num_probs = num_probs
    self._probs = np.atleast_2d(self._probs)
    if not self.replacement and (self.num_choices > num_candidates):
      raise ValueError(
          'Using replacement=False, "num_choices" must be less than'
          ' or equal to the number of "candidates". {} vs {}.'.format(
              self.num_choices, num_candidates))

  def __call__(self,
               image,
               **kwargs):
    """Run selection policies chosen from candidates."""
    probs = tf.convert_to_tensor(self._probs, dtype=tf.float32)
    for _ in range(self.num_choices):
      # Make a random selection.
      selected_index = tf.random.categorical(
          logits=tf.math.log(probs),
          num_samples=1,
          dtype=tf.int32,
          seed=self.seed)
      selected_index = tf.squeeze(selected_index)
      # Apply selected policy and no-op (Identity) for others.
      for i, p in enumerate(self.candidates):
        result = tf.cond(
            tf.equal(i, selected_index),
            lambda selected_policy=p: selected_policy(image=image, **kwargs),
            lambda: Identity()(image=image, **kwargs))
        image, kwargs = self._format_result_for_call(result, **kwargs)
      if not self.replacement:
        # Update probabilities for sampling without replacement.
        selection_one_hot = tf.one_hot(
            indices=selected_index, depth=self._num_probs, dtype=tf.float32)
        probs *= (tf.constant(1.0, dtype=tf.float32) - selection_one_hot)
    return result


@declare_policy(
    ['policy', 'prob'],
    [('policy', pg.typing.Object(AugmentPolicy), 'The policy to run.')]
)
class ApplyWithProbability(AugmentPolicy, SupportAugmentProbability):
  """Apply an augmentation policy according to a given probability."""

  def __call__(self,
               image,
               **kwargs):
    """Apply the policy according to the probability."""
    should_apply_op = tf.cast(
        tf.floor(tf.random_uniform([], dtype=tf.float32) + self.prob), tf.bool)
    result = tf.cond(should_apply_op,
                     lambda: self.policy(image=image, **kwargs),
                     lambda: Identity()(image=image, **kwargs))
    return result


@declare_policy([])
class AutoContrast(GroundTruthPassThroughPolicy):
  """Auto contrast policy."""

  def _augment(self, image):
    """Augments an image for Classification."""
    return ops.autocontrast(image)


@declare_policy([])
class Equalize(GroundTruthPassThroughPolicy):
  """Equalize operation policy."""

  def _augment(self, image):
    """Augments an image for Classification."""
    return ops.equalize(image)


@declare_policy([])
class Invert(GroundTruthPassThroughPolicy):
  """Invert operation policy for images with values in range [0, 255].

  Inverts an image with pixel values in range [0, 255]
  by subtracting original pixel values from 255.

  The augmented image dtype will remain the same.
  """

  def _augment(self, image):
    """Augments an image for Classification."""
    return 255 - image


@declare_policy(['magnitude'])
class Posterize(GroundTruthPassThroughPolicy,
                SupportAugmentMagnitude):
  """Posterize policy."""

  def _augment(self, image):
    """Augments an image for Classification."""
    return ops.posterize(
        image,
        int((self.magnitude / MAX_AUGMENT_MAGNITUDE) * POSTERIZE_BITS))


@declare_policy(['prob'])
class EqualizeOnlyBBoxes(AugmentPolicy,
                         SupportAugmentProbability):
  """Equalize policy on each bbox in the image with a probability."""

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection."""
    return ops.equalize_only_bboxes(image, bounding_boxes, self.prob)


#
#  Image enhancement policies.
#


class ImageEnhancementPolicy(GroundTruthPassThroughPolicy,
                             SupportAugmentMagnitude):
  """Base class for image enhacement policies."""

  def _augment(self, image):
    """Augments an image for Classification."""
    enhance_magnitude = ((self.magnitude / MAX_AUGMENT_MAGNITUDE) *
                         ENHANCE_FACTOR_MULT)
    enhance_magnitude += ENHANCE_FACTOR_ADD
    return self._enhance(image, enhance_magnitude)

  @abc.abstractmethod
  def _enhance(self, image, enhance_magnitude):
    """Enhance image with an enhance magnitude."""


@declare_policy(['magnitude'])
class Color(ImageEnhancementPolicy):
  """Color policy."""

  def _enhance(self, image, enhance_magnitude):
    return ops.color(image, enhance_magnitude)


@declare_policy(['magnitude'])
class Contrast(ImageEnhancementPolicy):
  """Contrast policy."""

  def _enhance(self, image, enhance_magnitude):
    return ops.contrast(image, enhance_magnitude)


@declare_policy(['magnitude'])
class Brightness(ImageEnhancementPolicy):
  """Brightness policy."""

  def _enhance(self, image, enhance_magnitude):
    return ops.brightness(image, enhance_magnitude)


@declare_policy(['magnitude'])
class Sharpness(ImageEnhancementPolicy):
  """Sharpness policy."""

  def _enhance(self, image, enhance_magnitude):
    return ops.sharpness(image, enhance_magnitude)


#
#  Solarize policies.
#


@declare_policy(['magnitude'])
class Solarize(GroundTruthPassThroughPolicy, SupportAugmentMagnitude):
  """Solarize policy.

  For each pixel in the image, select the pixel if the value is less than the
  threshold. Otherwise, subtract 255 from the pixel.
  """

  def _augment(self, image):
    """Augments an image for Classification."""
    threshold = int(
        (self.magnitude / MAX_AUGMENT_MAGNITUDE) * MAX_SOLARIZE_THRESHOLD)
    return ops.solarize(image, threshold)


@declare_policy(['magnitude', 'prob'])
class SolarizeOnlyBBoxes(AugmentPolicy,
                         SupportAugmentMagnitude,
                         SupportAugmentProbability):
  """Solarize each bbox in the image with a probability."""

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection."""
    threshold = int(
        (self.magnitude / MAX_AUGMENT_MAGNITUDE) * MAX_SOLARIZE_THRESHOLD)
    return ops.solarize_only_bboxes(
        image, bounding_boxes, self.prob, threshold)


@declare_policy(['magnitude'])
class SolarizeAdd(GroundTruthPassThroughPolicy, SupportAugmentMagnitude):
  """Solarize add policy.

  For each pixel in the image less than threshold we add 'addition' amount to it
  and then clip the pixel value to be between 0 and 255. The value of 'addition'
  is between -128 and 128.
  """

  def _augment(self, image):
    """Augments an image for Classification."""
    return ops.solarize_add(
        image,
        int((self.magnitude / MAX_AUGMENT_MAGNITUDE) * SOLARIZE_FACTOR_ADD))


#
# Cutout policies
#


@declare_policy(['magnitude', 'max_pad_size', 'replace'])
class CutoutImage(GroundTruthPassThroughPolicy,
                  SupportAugmentMagnitude,
                  SupportMaxPadSize,
                  NeedValueReplacement):
  """Cutout policy: https://arxiv.org/abs/1708.04552.

  Policy to cutout rectangular patches from the image.
  This policy only affects the input image, passing
  through any extra ground truth unchanged (e.g. bounding
  boxes or segmentation mask).
  """

  def _augment(self, image):
    """Augments an image for Classification."""
    pad_size = int((self.magnitude / MAX_AUGMENT_MAGNITUDE) * self.max_pad_size)
    return ops.cutout(image, pad_size, self.replace)


@declare_policy(
    ['magnitude', 'cutout_image', 'cutout_mask', 'max_pad_size', 'replace'], [
        ('cutout_image', pg.typing.Bool(default=True), 'Whether or not to '
         'perform cutout on the image.'),
        ('cutout_mask', pg.typing.Bool(default=True), 'Whether or not to '
         'perform cutout on the segmentation mask.'),
    ])
class CutoutSegmentation(AugmentPolicy,
                         SupportAugmentMagnitude,
                         SupportMaxPadSize,
                         NeedValueReplacement):
  """Cutout policy: https://arxiv.org/abs/1708.04552.

  Policy to cutout rectangular patches from the image and segmentation mask.
  """

  def _augment_with_segmentation_mask(
      self, image, segmentation_mask):
    """Augments an image and ground truth segmentation mask for Segmentation."""
    pad_size = int((self.magnitude / MAX_AUGMENT_MAGNITUDE) * self.max_pad_size)
    return ops.cutout_image_and_mask(image=image,
                                     segmentation_mask=segmentation_mask,
                                     pad_size=pad_size,
                                     replace=self.replace,
                                     mask_replace=self.mask_replace,
                                     cutout_image=self.cutout_image,
                                     cutout_mask=self.cutout_mask)


@declare_policy(['magnitude', 'prob', 'max_pad_size', 'replace'])
class CutoutOnlyBBoxes(AugmentPolicy,
                       SupportAugmentMagnitude,
                       SupportAugmentProbability,
                       SupportMaxPadSize,
                       NeedValueReplacement):
  """Apply cutout to each bbox in the image with a probability."""

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection."""
    pad_size = int((self.magnitude / MAX_AUGMENT_MAGNITUDE) * self.max_pad_size)
    return ops.cutout_only_bboxes(
        image, bounding_boxes, self.prob, pad_size, self.replace)


@declare_policy(
    ['magnitude', 'max_pad_fraction', 'replace_with_mean'],
    [
        ('max_pad_fraction', pg.typing.Float(
            min_value=0, max_value=1.0, default=0.75)),
        ('replace_with_mean', pg.typing.Bool(default=False))
    ])
class BBoxCutout(AugmentPolicy, SupportAugmentMagnitude):
  """Applies cutout to the image according to bbox information.

  This is a cutout variant that using bbox information to make more informed
  decisions on where to place the cutout mask.
  """

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection."""
    pad_fraction = (
        self.magnitude / MAX_AUGMENT_MAGNITUDE) * self.max_pad_fraction
    return ops.bbox_cutout(
        image, bounding_boxes, pad_fraction, self.replace_with_mean)


#
#  Translation policies.
#


@pg.members([
    ('max_shift', pg.typing.Int(default=MAX_SHIFT_DEFAULT)),
])
class TranslatePolicy(AugmentPolicy,
                      SupportAugmentMagnitude,
                      NeedValueReplacement):
  """Base class for translation policices."""

  def pixels_to_shift(self):
    shift = (self.magnitude / MAX_AUGMENT_MAGNITUDE) * float(self.max_shift)
    return _randomly_negate_tensor(shift)


@declare_policy(['magnitude', 'replace', 'max_shift'])
class TranslateX(TranslatePolicy):
  """Translate (shift) in the X dimension."""

  def _augment(self, image):
    """Augments an image for Classification."""
    return ops.translate_x(image, self.pixels_to_shift(), self.replace)

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection."""
    return ops.translate_bbox(
        image,
        bounding_boxes,
        self.pixels_to_shift(),
        self.replace,
        shift_horizontal=True)

  def _augment_with_segmentation_mask(
      self, image, segmentation_mask):
    """Augments an image and ground truth segmentation mask for Segmentation."""
    return ops.translate_x_with_mask(image,
                                     segmentation_mask,
                                     self.pixels_to_shift(),
                                     self.replace,
                                     self.mask_replace)


@declare_policy(['magnitude', 'replace', 'max_shift'])
class TranslateY(TranslatePolicy):
  """Translate (shift) in the Y dimension."""

  def _augment(self, image):
    """Augments an image for Classification."""
    return ops.translate_y(image, self.pixels_to_shift(), self.replace)

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection."""
    return ops.translate_bbox(
        image,
        bounding_boxes,
        self.pixels_to_shift(),
        self.replace,
        shift_horizontal=False)

  def _augment_with_segmentation_mask(
      self, image, segmentation_mask):
    """Augments an image and ground truth segmentation mask for Segmentation."""
    return ops.translate_y_with_mask(image,
                                     segmentation_mask,
                                     self.pixels_to_shift(),
                                     self.replace,
                                     self.mask_replace)


@declare_policy(
    ['magnitude', 'prob', 'replace', 'max_shift'],
    [('max_shift', pg.typing.Int(default=120))])
class TranslateXOnlyBBoxes(TranslatePolicy,
                           SupportAugmentProbability):
  """Translate policy in X dim to each bbox in the image with a probability."""

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection."""
    return ops.translate_x_only_bboxes(
        image, bounding_boxes, self.prob, self.pixels_to_shift(), self.replace)


@declare_policy(
    ['magnitude', 'prob', 'replace', 'max_shift'],
    [('max_shift', pg.typing.Int(default=120))])
class TranslateYOnlyBBoxes(TranslatePolicy,
                           SupportAugmentProbability):
  """Translate policy in Y dim to each bbox in the image with a probability."""

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection."""
    return ops.translate_y_only_bboxes(
        image, bounding_boxes, self.prob, self.pixels_to_shift(), self.replace)


#
# Shear policies.
#


class ShearPolicy(AugmentPolicy,
                  SupportAugmentMagnitude,
                  NeedValueReplacement):
  """Base class for shear based bounding box policy."""

  def shear_magnitude(self):
    """Get rotation degrees."""
    magnitude = (self.magnitude / MAX_AUGMENT_MAGNITUDE) * MAX_SHEAR
    return _randomly_negate_tensor(magnitude)


@declare_policy(['magnitude', 'replace'])
class ShearX(ShearPolicy):
  """Apply Shear Transformation on X dimension."""

  def _augment(self, image):
    """Augments an image for Classification."""
    return ops.shear_x(image, self.shear_magnitude(), self.replace)

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection."""
    return ops.shear_with_bboxes(
        image,
        bounding_boxes,
        self.shear_magnitude(),
        self.replace,
        shear_horizontal=True)

  def _augment_with_segmentation_mask(
      self, image, segmentation_mask):
    """Augments an image and ground truth segmentation mask for Segmentation."""
    return ops.shear_x_with_mask(image,
                                 segmentation_mask,
                                 self.shear_magnitude(),
                                 self.replace,
                                 self.mask_replace)


@declare_policy(['magnitude', 'replace'])
class ShearY(ShearPolicy):
  """Apply Shear Transformation on Y dimension."""

  def _augment(self, image):
    """Augments an image for Classification."""
    return ops.shear_y(image, self.shear_magnitude(), self.replace)

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection."""
    return ops.shear_with_bboxes(
        image,
        bounding_boxes,
        self.shear_magnitude(),
        self.replace,
        shear_horizontal=False)

  def _augment_with_segmentation_mask(
      self, image, segmentation_mask):
    """Augments an image and ground truth segmentation mask for Segmentation."""
    return ops.shear_y_with_mask(image,
                                 segmentation_mask,
                                 self.shear_magnitude(),
                                 self.replace,
                                 self.mask_replace)


@declare_policy(['magnitude', 'prob', 'replace'])
class ShearXOnlyBBoxes(ShearPolicy,
                       SupportAugmentProbability):
  """Apply Shear Transformation on X dim of each bbox with a probability."""

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection."""
    return ops.shear_x_only_bboxes(
        image, bounding_boxes, self.prob, self.shear_magnitude(), self.replace)


@declare_policy(['magnitude', 'prob', 'replace'])
class ShearYOnlyBBoxes(ShearPolicy,
                       SupportAugmentProbability):
  """Apply Shear Transformation on Y dim of each bbox with a probability."""

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection."""
    return ops.shear_y_only_bboxes(
        image, bounding_boxes, self.prob, self.shear_magnitude(), self.replace)


#
# Rotation policies.
#


@declare_policy(['magnitude', 'replace'])
class RotatePolicy(AugmentPolicy,
                   SupportAugmentMagnitude,
                   NeedValueReplacement):
  """Base class for rotation policices."""

  def degrees(self):
    """Get rotation degrees."""
    degrees = (self.magnitude / MAX_AUGMENT_MAGNITUDE) * MAX_ROTATE_DEGREES
    return _randomly_negate_tensor(degrees)


@declare_policy(['magnitude', 'replace'])
class Rotate(RotatePolicy):
  """Rotate image and ground truth."""

  def _augment(self, image):
    """Augments an image for Classification."""
    return ops.rotate(image, self.degrees(), self.replace)

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection."""
    return ops.rotate_with_bboxes(
        image, bounding_boxes, self.degrees(), self.replace)

  def _augment_with_segmentation_mask(
      self, image, segmentation_mask):
    """Augments an image and ground truth segmentation mask for Segmentation."""
    return ops.rotate_with_mask(image,
                                segmentation_mask,
                                self.degrees(),
                                self.replace,
                                self.mask_replace)


@declare_policy(['magnitude', 'prob', 'replace'])
class RotateOnlyBBoxes(RotatePolicy,
                       SupportAugmentProbability):
  """Rotate only on images in bounding_boxes with a probability."""

  def degrees(self):
    """Get rotation degrees."""
    degrees = (self.magnitude / MAX_AUGMENT_MAGNITUDE) * MAX_ROTATE_DEGREES
    return _randomly_negate_tensor(degrees)

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection."""
    return ops.rotate_only_bboxes(
        image, bounding_boxes, self.prob, self.degrees(), self.replace)


@declare_policy(['prob'])
class FlipOnlyBBoxes(AugmentPolicy,
                     SupportAugmentProbability):
  """Left-to-right flip on each box in the image with a probability."""

  def _augment_with_bounding_boxes(
      self, image,
      bounding_boxes):
    """Augments an image and ground truth bounding boxes for Detection."""
    return ops.flip_only_bboxes(image, bounding_boxes, self.prob)


def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  should_flip = tf.cast(tf.floor(tf.random_uniform([]) + 0.5), tf.bool)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor


#
# AutoAugment detection policies.
#
# Barret, et al. Learning Data Augmentation Strategies for Object Detection.
# Arxiv: https://arxiv.org/abs/1906.11172
#


def policy_constants_spec():
  """Returns the value spec for policy constants."""
  return pg.typing.Dict([
      ('replace_value', pg.typing.List(
          pg.typing.Int(min_value=0), size=3, default=[128, 128, 128])),
      ('mask_replace_value', pg.typing.Int(min_value=0, default=255)),
      ('translate_max_shift', pg.typing.Int(default=250)),
      ('translate_bbox_max_shift', pg.typing.Int(default=120)),
      ('cutout_max_pad_fraction', pg.typing.Float(default=0.75)),
      ('cutout_max_pad_size', pg.typing.Int(default=100)),
      ('cutout_bbox_replace_with_mean', pg.typing.Bool(default=False)),
  ])


def default_policy_constants():
  """Return default values from policy constants spec."""
  return pg.Dict(value_spec=policy_constants_spec())


def apply_policy_constants(policy, constants):
  """Apply policy constants."""
  constants = constants or default_policy_constants()
  # Rebind policy constants.
  pg.patch_on_member(
      policy, (TranslateX, TranslateY),
      'max_shift', constants.translate_max_shift)
  pg.patch_on_member(
      policy, (TranslateXOnlyBBoxes, TranslateYOnlyBBoxes),
      'max_shift', constants.translate_bbox_max_shift)
  pg.patch_on_member(
      policy, (CutoutImage, CutoutSegmentation, CutoutOnlyBBoxes),
      'max_pad_size', constants.cutout_max_pad_size)
  pg.patch_on_member(
      policy, BBoxCutout,
      'max_pad_fraction', constants.cutout_max_pad_fraction)
  pg.patch_on_member(
      policy, BBoxCutout,
      'replace_with_mean', constants.cutout_bbox_replace_with_mean)
  pg.patch_on_member(
      policy, AugmentPolicy,
      'replace', constants.replace_value)
  pg.patch_on_member(
      policy, AugmentPolicy,
      'mask_replace', constants.mask_replace_value)
  return policy


@pg.functor([
    ('policy_constants', policy_constants_spec().noneable())
])
def autoaugment_detection_policy_v0(
    policy_constants=default_policy_constants()):
  """Augment policy used in AutoAugment detection paper."""
  policy = Choice([
      Sequential([ApplyWithProbability(TranslateX(4), prob=0.6),
                  ApplyWithProbability(Equalize(), prob=0.8)]),
      Sequential([TranslateYOnlyBBoxes(2, prob=0.2),
                  ApplyWithProbability(CutoutImage(8), prob=0.8)]),
      Sequential([ApplyWithProbability(Sharpness(8), prob=0.0),
                  ApplyWithProbability(ShearX(0), prob=0.4)]),
      Sequential([ApplyWithProbability(ShearY(2), prob=1.0),
                  TranslateYOnlyBBoxes(6, prob=0.6)]),
      Sequential([ApplyWithProbability(Rotate(10), prob=0.6),
                  ApplyWithProbability(Color(6), prob=1.0)]),
  ])
  return apply_policy_constants(policy, policy_constants)  # pytype: disable=wrong-arg-types


@pg.functor([
    ('policy_constants', policy_constants_spec().noneable())
])
def autoaugment_detection_policy_v1(
    policy_constants=default_policy_constants()):
  """Augment policy used in AutoAugment detection paper."""
  policy = Choice([
      Sequential([ApplyWithProbability(TranslateX(4), prob=0.6),
                  ApplyWithProbability(Equalize(), prob=0.8)]),
      Sequential([TranslateYOnlyBBoxes(2, prob=0.2),
                  ApplyWithProbability(CutoutImage(8), prob=0.8)]),
      Sequential([ApplyWithProbability(Sharpness(8), prob=0.0),
                  ApplyWithProbability(ShearX(0), prob=0.4)]),
      Sequential([ApplyWithProbability(ShearY(2), prob=1.0),
                  TranslateYOnlyBBoxes(6, prob=0.6)]),
      Sequential([ApplyWithProbability(Rotate(10), prob=0.6),
                  ApplyWithProbability(Color(6), prob=1.0)]),
      Sequential([ApplyWithProbability(Color(0), prob=0.0),
                  ShearXOnlyBBoxes(4, prob=0.8)]),
      Sequential([ShearYOnlyBBoxes(2, prob=0.8),
                  FlipOnlyBBoxes(prob=0.0)]),
      Sequential([ApplyWithProbability(Equalize(), prob=0.6),
                  ApplyWithProbability(TranslateX(2), prob=0.2)]),
      Sequential([ApplyWithProbability(Color(10), prob=1.0),
                  TranslateYOnlyBBoxes(6, prob=0.4)]),
      Sequential([ApplyWithProbability(Rotate(10), prob=0.8),
                  ApplyWithProbability(Contrast(10), prob=0.0)]),
      Sequential([ApplyWithProbability(CutoutImage(2), prob=0.2),
                  ApplyWithProbability(Brightness(10), prob=0.8)]),
      Sequential([ApplyWithProbability(Color(6), prob=1.0),
                  ApplyWithProbability(Equalize(), prob=1.0)]),
      Sequential([CutoutOnlyBBoxes(6, prob=0.4),
                  TranslateYOnlyBBoxes(2, prob=0.8)]),
      Sequential([ApplyWithProbability(Color(8), prob=0.2),
                  ApplyWithProbability(Rotate(10), prob=0.8)]),
      Sequential([ApplyWithProbability(Sharpness(4), prob=0.4),
                  TranslateYOnlyBBoxes(4, prob=0.0)]),
      Sequential([ApplyWithProbability(Sharpness(4), prob=1.0),
                  ApplyWithProbability(SolarizeAdd(4), prob=0.4)]),
      Sequential([ApplyWithProbability(Rotate(8), prob=1.0),
                  ApplyWithProbability(Sharpness(8), prob=0.2)]),
      Sequential([ApplyWithProbability(ShearY(10), prob=0.6),
                  EqualizeOnlyBBoxes(prob=0.6)]),
      Sequential([ApplyWithProbability(ShearX(6), prob=0.2),
                  TranslateYOnlyBBoxes(10, prob=0.2)]),
      Sequential([ApplyWithProbability(SolarizeAdd(8), prob=0.6),
                  ApplyWithProbability(Brightness(10), prob=0.8)]),
  ])
  return apply_policy_constants(policy, policy_constants)  # pytype: disable=wrong-arg-types


@pg.functor([
    ('policy_constants', policy_constants_spec().noneable())
])
def autoaugment_detection_policy_v2(
    policy_constants=default_policy_constants()):
  """Augment policy used in AutoAugment detection paper."""
  policy = Choice([
      Sequential([ApplyWithProbability(Color(6), prob=0.0),
                  ApplyWithProbability(CutoutImage(8), prob=0.6),
                  ApplyWithProbability(Sharpness(8), prob=0.4)]),
      Sequential([ApplyWithProbability(Rotate(8), prob=0.4),
                  ApplyWithProbability(Sharpness(2), prob=0.4),
                  ApplyWithProbability(Rotate(10), prob=0.8)]),
      Sequential([ApplyWithProbability(TranslateY(8), prob=1.0),
                  ApplyWithProbability(AutoContrast(), prob=0.8)]),
      Sequential([ApplyWithProbability(AutoContrast(), prob=0.4),
                  ApplyWithProbability(ShearX(8), prob=0.8),
                  ApplyWithProbability(Brightness(10), prob=0.0)]),
      Sequential([ApplyWithProbability(SolarizeAdd(6), prob=0.2),
                  ApplyWithProbability(Contrast(10), prob=0.0),
                  ApplyWithProbability(AutoContrast(), prob=0.6)]),
      Sequential([ApplyWithProbability(CutoutImage(0), prob=0.2),
                  ApplyWithProbability(Solarize(8), prob=0.8),
                  ApplyWithProbability(Color(4), prob=1.0)]),
      Sequential([ApplyWithProbability(TranslateY(4), prob=0.0),
                  ApplyWithProbability(Equalize(), prob=0.6),
                  ApplyWithProbability(Solarize(10), prob=0.0)]),
      Sequential([ApplyWithProbability(TranslateY(2), prob=0.2),
                  ApplyWithProbability(ShearY(8), prob=0.8),
                  ApplyWithProbability(Rotate(8), prob=0.8)]),
      Sequential([ApplyWithProbability(CutoutImage(8), prob=0.8),
                  ApplyWithProbability(Brightness(8), prob=0.8),
                  ApplyWithProbability(CutoutImage(2), prob=0.2)]),
      Sequential([ApplyWithProbability(Color(4), prob=0.8),
                  ApplyWithProbability(TranslateY(6), prob=1.0),
                  ApplyWithProbability(Rotate(6), prob=0.6)]),
      Sequential([ApplyWithProbability(Rotate(10), prob=0.6),
                  ApplyWithProbability(BBoxCutout(4), prob=1.0),
                  ApplyWithProbability(CutoutImage(8), prob=0.2)]),
      Sequential([ApplyWithProbability(Rotate(0), prob=0.0),
                  ApplyWithProbability(Equalize(), prob=0.6),
                  ApplyWithProbability(ShearY(8), prob=0.6)]),
      Sequential([ApplyWithProbability(Brightness(8), prob=0.8),
                  ApplyWithProbability(AutoContrast(), prob=0.4),
                  ApplyWithProbability(Brightness(2), prob=0.2)]),
      Sequential([ApplyWithProbability(TranslateY(8), prob=0.8),
                  ApplyWithProbability(Solarize(6), prob=0.4),
                  ApplyWithProbability(SolarizeAdd(10), prob=0.2)]),
      Sequential([ApplyWithProbability(Contrast(10), prob=1.0),
                  ApplyWithProbability(SolarizeAdd(8), prob=0.2),
                  ApplyWithProbability(Equalize(), prob=0.2)]),
  ])
  return apply_policy_constants(policy, policy_constants)  # pytype: disable=wrong-arg-types


@pg.functor([
    ('policy_constants', policy_constants_spec().noneable())
])
def autoaugment_detection_policy_v3(
    policy_constants=default_policy_constants()):
  """Autogument policy used in AutoAugment detection paper."""
  policy = Choice([
      Sequential([ApplyWithProbability(Posterize(2), prob=0.8),
                  ApplyWithProbability(TranslateX(8), prob=1.0)]),
      Sequential([ApplyWithProbability(BBoxCutout(10), prob=0.2),
                  ApplyWithProbability(Sharpness(8), prob=1.0)]),
      Sequential([ApplyWithProbability(Rotate(8), prob=0.6),
                  ApplyWithProbability(Rotate(10), prob=0.8)]),
      Sequential([ApplyWithProbability(Equalize(), prob=0.8),
                  ApplyWithProbability(AutoContrast(), prob=0.2)]),
      Sequential([ApplyWithProbability(SolarizeAdd(2), prob=0.2),
                  ApplyWithProbability(TranslateY(8), prob=0.2)]),
      Sequential([ApplyWithProbability(Sharpness(2), prob=0.0),
                  ApplyWithProbability(Color(8), prob=0.4)]),
      Sequential([ApplyWithProbability(Equalize(), prob=1.0),
                  ApplyWithProbability(TranslateY(8), prob=1.0)]),
      Sequential([ApplyWithProbability(Posterize(2), prob=0.6),
                  ApplyWithProbability(Rotate(10), prob=0.0)]),
      Sequential([ApplyWithProbability(AutoContrast(), prob=0.6),
                  ApplyWithProbability(Rotate(6), prob=1.0)]),
      Sequential([ApplyWithProbability(Equalize(), prob=0.0),
                  ApplyWithProbability(CutoutImage(10), prob=0.8)]),
      Sequential([ApplyWithProbability(Brightness(2), prob=1.0),
                  ApplyWithProbability(TranslateY(6), prob=1.0)]),
      Sequential([ApplyWithProbability(Contrast(2), prob=0.0),
                  ApplyWithProbability(ShearY(0), prob=0.8)]),
      Sequential([ApplyWithProbability(AutoContrast(), prob=0.8),
                  ApplyWithProbability(Contrast(10), prob=0.2)]),
      Sequential([ApplyWithProbability(Rotate(10), prob=1.0),
                  ApplyWithProbability(CutoutImage(10), prob=1.0)]),
      Sequential([ApplyWithProbability(SolarizeAdd(6), prob=1.0),
                  ApplyWithProbability(Equalize(), prob=0.8)]),
  ])
  return apply_policy_constants(policy, policy_constants)  # pytype: disable=wrong-arg-types


def autoaugment_detection_policy(name, policy_constants=None):
  """Returns AutoAugment detection policy by name."""
  if name == 'v0':
    return autoaugment_detection_policy_v0(policy_constants)()
  elif name == 'v1':
    return autoaugment_detection_policy_v1(policy_constants)()
  elif name == 'v2':
    return autoaugment_detection_policy_v2(policy_constants)()
  elif name == 'v3':
    return autoaugment_detection_policy_v3(policy_constants)()
  else:
    raise ValueError('Unsupported policy name: %r' % name)


@pg.functor([
    ('policy_constants', policy_constants_spec().noneable())
])
def autoaugment_classification_policy_v0(
    policy_constants=default_policy_constants()):
  """Autogument policy used in AutoAugment classification paper."""
  # The following policy is from Cloud TPU EfficientNet GitHub repo:
  # cs/google3/third_party/cloud_tpu/models/efficientnet/autoaugment.py
  policy = Choice([
      Sequential([ApplyWithProbability(Equalize(), prob=0.8),
                  ApplyWithProbability(ShearY(4), prob=0.8)]),
      Sequential([ApplyWithProbability(Color(9), prob=0.4),
                  ApplyWithProbability(Equalize(), prob=0.6)]),
      Sequential([ApplyWithProbability(Color(1), prob=0.4),
                  ApplyWithProbability(Rotate(8), prob=0.6)]),
      Sequential([ApplyWithProbability(Solarize(3), prob=0.8),
                  ApplyWithProbability(Equalize(), prob=0.4)]),
      Sequential([ApplyWithProbability(Solarize(2), prob=0.4),
                  ApplyWithProbability(Solarize(2), prob=0.6)]),
      Sequential([ApplyWithProbability(Color(0), prob=0.2),
                  ApplyWithProbability(Equalize(), prob=0.8)]),
      Sequential([ApplyWithProbability(Invert(), prob=0.4),
                  ApplyWithProbability(Rotate(0), prob=0.6)]),
      Sequential([ApplyWithProbability(Color(1), prob=0.6),
                  ApplyWithProbability(Equalize(), prob=1.0)]),
      Sequential([ApplyWithProbability(Invert(), prob=0.4),
                  ApplyWithProbability(Rotate(0), prob=0.9)]),
      Sequential([ApplyWithProbability(Equalize(), prob=1.0),
                  ApplyWithProbability(ShearY(3), prob=0.6)]),
      Sequential([ApplyWithProbability(Color(7), prob=0.4),
                  ApplyWithProbability(Equalize(), prob=0.6)]),
      Sequential([ApplyWithProbability(Posterize(6), prob=0.4),
                  ApplyWithProbability(AutoContrast(), prob=0.4)]),
      Sequential([ApplyWithProbability(Solarize(8), prob=0.6),
                  ApplyWithProbability(Color(9), prob=0.6)]),
      Sequential([ApplyWithProbability(Solarize(4), prob=0.2),
                  ApplyWithProbability(Rotate(9), prob=0.8)]),
      Sequential([ApplyWithProbability(Rotate(7), prob=1.0),
                  ApplyWithProbability(TranslateY(9), prob=0.8)]),
      Sequential([ApplyWithProbability(ShearX(0), prob=0.0),
                  ApplyWithProbability(Solarize(4), prob=0.8)]),
      Sequential([ApplyWithProbability(ShearY(0), prob=0.8),
                  ApplyWithProbability(Color(4), prob=0.6)]),
      Sequential([ApplyWithProbability(Color(0), prob=1.0),
                  ApplyWithProbability(Rotate(2), prob=0.6)]),
      Sequential([ApplyWithProbability(Equalize(), prob=0.8),
                  ApplyWithProbability(Equalize(), prob=0.0)]),
      Sequential([ApplyWithProbability(Equalize(), prob=1.0),
                  ApplyWithProbability(AutoContrast(), prob=0.6)]),
      Sequential([ApplyWithProbability(ShearY(7), prob=0.4),
                  ApplyWithProbability(SolarizeAdd(7), prob=0.6)]),
      Sequential([ApplyWithProbability(Posterize(2), prob=0.8),
                  ApplyWithProbability(Solarize(10), prob=0.6)]),
      Sequential([ApplyWithProbability(Solarize(8), prob=0.6),
                  ApplyWithProbability(Equalize(), prob=0.6)]),
      Sequential([ApplyWithProbability(Color(6), prob=0.8),
                  ApplyWithProbability(Rotate(5), prob=0.4)]),
  ])
  return apply_policy_constants(policy, policy_constants)  # pytype: disable=wrong-arg-types


def autoaugment_classification_policy(name, policy_constants=None):
  """Returns AutoAugment classification policy by name."""
  policies = {'v0': autoaugment_classification_policy_v0}
  if name not in policies:
    raise ValueError('Invalid augmentation name : {:}'.format(name))
  else:
    return policies[name](policy_constants)()


def create_randaugment_policy(
    num_ops, magnitude,
    candidate_op_fns):
  """Create a RandAugment Policy.

  Creates an augmentation policy defined by running `num_ops` augmentation
  policies sequentially, each parameterized by a `magnitude` value. The
  operations are chosen randomly from a uniform distribution over the
  `candidate_op_fns`. For example, num_ops=3 and magnitude=4 will
  run 3 (randomly selected) sequential augmentation operations,
  each with a magnitude of 4.

  For more details on RandAugment, see the publication:
    https://arxiv.org/abs/1909.13719

  Args:
    num_ops: Number of augmentation transformations to apply sequentially.
    magnitude: Magnitude to use for all the transformations. This method does
      not make any assumptions about magnitude value ranges.
    candidate_op_fns: List of candidate operation functions, where each
      item is a function taking an integer argument for `magnitude` and
      returning an AugmentPolicy.

  Returns:
    The resulting AugmentPolicy object for RandAugment.
  """
  # Materialize the ops by calling the op functions with the magnitude param.
  candidate_ops = [f(magnitude) for f in candidate_op_fns]
  return Choice(candidates=candidate_ops, num_choices=num_ops)  # pytype: disable=bad-return-type


@pg.functor(
    args=[('num_ops', pg.typing.Int(min_value=1)),
          ('magnitude', pg.typing.Int(min_value=0, max_value=10)),
          ('cutout_pad_size', pg.typing.Int(min_value=1, default=40)),
          ('translate_max_shift', pg.typing.Int(min_value=1, default=100))],
    returns=pg.typing.Object(AugmentPolicy))
def randaugment_classification_policy(num_ops,
                                      magnitude,
                                      cutout_pad_size = 40,
                                      translate_max_shift = 100
                                     ):
  """RandAugment Policy for Classification.

  Creates an augmentation policy defined by running `num_ops` augmentation
  policies sequentially, each parameterized by a `magnitude` value. The
  operations are chosen randomly from a uniform distribution. For example,
  num_ops=3 and magnitude=4 will run 3 (randomly selected) sequential
  augmentation operations, each with a magnitude of 4.

  The candidate operations are:
    AutoContrast, Equalize, Invert, Rotate, Posterize,
    Solarize, Color, Contrast, Sharpness, ShearX, ShearY,
    TranslateX, TranslateY, Cutout, SolarizeAdd.

  For more details on RandAugment, see the publication:
    https://arxiv.org/abs/1909.13719

  Args:
    num_ops: Positive number of augmentation transformations
      to apply sequentially.
    magnitude: Magnitude in range [0, 10] to use for all the transformations.
    cutout_pad_size: The maximum padding size for CutoutImage.
    translate_max_shift: The maximum shift for Translate.

  Returns:
    The resulting AugmentPolicy object for RandAugment.
  """

  candidate_op_fns = [
      lambda m: AutoContrast(),  # No magnitude input for AutoContrast.
      lambda m: Equalize(),  # No magnitude input for Equalize.
      lambda m: Invert(),  # No magnitude input for Invert.
      Rotate,
      Posterize,
      Solarize,
      Color,
      Contrast,
      Sharpness,
      ShearX,
      ShearY,
      lambda m: TranslateX(m, max_shift=translate_max_shift),
      lambda m: TranslateY(m, max_shift=translate_max_shift),
      lambda m: CutoutImage(m, max_pad_size=cutout_pad_size),
      SolarizeAdd,
  ]
  return create_randaugment_policy(num_ops, magnitude, candidate_op_fns)


@pg.functor(
    args=[('num_ops', pg.typing.Int(min_value=1)),
          ('magnitude', pg.typing.Int(min_value=0, max_value=10))],
    returns=pg.typing.Object(AugmentPolicy))
def randaugment_detection_policy(num_ops, magnitude):
  """RandAugment Policy for Detection.

  Creates an augmentation policy defined by running `num_ops` augmentation
  policies sequentially, each parameterized by a `magnitude` value. The
  operations are chosen randomly from a uniform distribution. For example,
  num_ops=3 and magnitude=4 will run 3 (randomly selected) sequential
  augmentation operations, each with a magnitude of 4.

  The candidate operations are:
    Equalize, Solarize, Color, CutoutImage, SolarizeAdd,
    TranslateX, TranslateY, ShearX, ShearY, Rotate.

  For more details on RandAugment, see the publication:
    https://arxiv.org/abs/1909.13719

  Args:
    num_ops: Positive number of augmentation transformations
      to apply sequentially.
    magnitude: Magnitude in range [0, 10] to use for all the transformations.

  Returns:
    The resulting AugmentPolicy object for RandAugment.
  """
  # TODO: Expose other ops params, like "max_shift".
  candidate_op_fns = [
      lambda m: Equalize(),  # No magnitude input for Equalize.
      Solarize,
      Color,
      CutoutImage,
      SolarizeAdd,
      TranslateX,
      TranslateY,
      ShearX,
      ShearY,
      Rotate,
  ]
  return create_randaugment_policy(num_ops, magnitude, candidate_op_fns)


@pg.functor(
    args=[('num_ops', pg.typing.Int(min_value=1)),
          ('magnitude', pg.typing.Int(min_value=0, max_value=10))],
    returns=pg.typing.Object(AugmentPolicy))
def randaugment_segmentation_policy(num_ops,
                                    magnitude):
  """RandAugment Policy for Segmentation.

  Creates an augmentation policy defined by running `num_ops` augmentation
  policies sequentially, each parameterized by a `magnitude` value. The
  operations are chosen randomly from a uniform distribution. For example,
  num_ops=3 and magnitude=4 will run 3 (randomly selected) sequential
  augmentation operations, each with a magnitude of 4.

  The candidate operations are:
    Equalize, Solarize, Color, CutoutImage, SolarizeAdd,
    TranslateX, TranslateY, ShearX, ShearY, Rotate.

  For more details on RandAugment, see the publication:
    https://arxiv.org/abs/1909.13719

  Args:
    num_ops: Positive number of augmentation transformations
      to apply sequentially.
    magnitude: Magnitude in range [0, 10] to use for all the transformations.

  Returns:
    The resulting AugmentPolicy object for RandAugment.
  """
  # TODO: Expose other ops params, like "max_shift".
  candidate_op_fns = [
      lambda m: Equalize(),  # No magnitude input for Equalize.
      Solarize,
      Color,
      CutoutSegmentation,
      SolarizeAdd,
      TranslateX,
      TranslateY,
      ShearX,
      ShearY,
      Rotate,
  ]
  return create_randaugment_policy(num_ops, magnitude, candidate_op_fns)


def get_policy_from_str(aug_str):
  """Converts json augmentation string to policy.

  Args:
    aug_str: The json str could be an AugmentPolicy
      or RandAugmentSpecBuilder.

  Returns:
    The resulting AugmentPolicy object.
  """
  policy_or_spec = pg.from_json_str(aug_str)
  if isinstance(policy_or_spec,
                policies_search_space.RandAugmentDetectionSpecBuilder):
    spec = policy_or_spec
    policy = randaugment_detection_policy(spec.num_ops, spec.magnitude)()
    return policy
  elif isinstance(policy_or_spec,
                  policies_search_space.RandAugmentSegmentationSpecBuilder):
    spec = policy_or_spec
    policy = randaugment_segmentation_policy(spec.num_ops, spec.magnitude)()
    return policy
  else:
    # NOTE: This function is generic so that it supports the
    # "AugmentPolicy" PyGlove json string as well. The "autoaugment" policies
    # are of type "AugmentPolicy" which is also derived from PyGlove.
    policy = policy_or_spec
    return policy
