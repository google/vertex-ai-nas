# Lint: python3
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
"""3D augmentation policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from lingvo.core import py_utils
from lingvo.tasks.car import input_preprocessors
import numpy as np
import pyglove as pg
import tensorflow.compat.v1 as tf

from nas_lib.augmentation_3d import policies_search_space


def _nested_map_to_dict(obj):
  """Converts a NestedMap to a dict."""
  if isinstance(obj, py_utils.NestedMap):
    return {k: _nested_map_to_dict(v) for k, v in obj.items()}
  else:
    return obj


def _dict_to_nested_map(obj):
  """Converts a dict to a NestedMap."""
  if isinstance(obj, dict):
    return py_utils.NestedMap(
        {k: _dict_to_nested_map(v) for k, v in obj.items()})
  else:
    return obj


def _convert_dict_items_to_tensor(obj):
  """Converts items in a dict or NestedMap to tensors."""
  if isinstance(obj, (dict, py_utils.NestedMap)):
    return type(obj)(
        {k: _convert_dict_items_to_tensor(v) for k, v in obj.items()})
  else:
    return tf.convert_to_tensor(obj)


class AugmentPolicy3D(pg.Object):
  """Interface for a 3D augment policy."""

  def __call__(self, features):
    """Runs data augmentation on the input features."""
    return self._augment(features)

  def _augment(self, features):
    """Performs data augmentation on the input features."""
    raise NotImplementedError


class Identity(AugmentPolicy3D):
  """Identity policy."""

  def _augment(self, features):
    """Returns the features unchanged."""
    return features


@pg.members(
    [('policy', pg.typing.Object(AugmentPolicy3D), 'The policy to run.'),
     ('prob', pg.typing.Float(min_value=0.,
                              max_value=1.), 'Augment probability.')],
    metadata={'init_arg_list': ['policy', 'prob']})
class ApplyWithProbability(AugmentPolicy3D):
  """Apply an augmentation policy according to a given probability."""

  def __call__(self, features):
    """Apply the policy according to the probability."""
    should_apply_op = tf.cast(
        tf.floor(tf.random_uniform([], dtype=tf.float32) + self.prob), tf.bool)
    result = tf.cond(should_apply_op, lambda: self.policy(features),
                     lambda: Identity()(features))
    return result


@pg.members([('children', pg.typing.List(pg.typing.Object(AugmentPolicy3D)),
              'Child policies to apply sequentially.')],
            metadata={'init_arg_list': ['children']})
class Sequential(AugmentPolicy3D):
  """Sequential policy."""

  def __call__(self, features):
    """Sequentially calls all children augmentation policies."""
    for child in self.children:
      features = child(features)
    return features


@pg.members(
    [('candidates', pg.typing.List(
        pg.typing.Object(AugmentPolicy3D)), 'Candidate policies to sample.'),
     ('num_choices', pg.typing.Int(
         default=1, min_value=1), 'Number of policies to sample.'),
     ('probs', pg.typing.List(pg.typing.Float(min_value=0.0,
                                              max_value=1.0)).noneable(),
      'Probabability distribution used to sample candidates. '
      'If provided, this list must be the same size as the candidates. '
      'If not provided, then a uniform distribution over all candidates '
      'will be used.'),
     ('replacement', pg.typing.Bool(default=True),
      'True to sample with replacement.'),
     ('seed', pg.typing.Int().noneable(), 'Random seed for sampling.')],
    metadata={
        'init_arg_list': [
            'candidates', 'num_choices', 'probs', 'replacement', 'seed'
        ]
    })
class Choice(AugmentPolicy3D):
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

  def __call__(self, features):
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
        features = tf.cond(
            tf.equal(i, selected_index),
            lambda selected_policy=p: selected_policy(features),
            lambda: Identity()(features))
      if not self.replacement:
        # Update probabilities for sampling without replacement.
        selection_one_hot = tf.one_hot(
            indices=selected_index, depth=self._num_probs, dtype=tf.float32)
        probs *= (tf.constant(1.0, dtype=tf.float32) - selection_one_hot)
    return features


class SingleOpPolicy3D(AugmentPolicy3D):
  """Base class for policys that perform a single augmentation operation."""

  def _on_bound(self):
    if self.is_deterministic:
      self._preprocessor = self._init_preprocessor()

  def _augment(self, features):
    """Performs data augmentation on the input features."""
    inputs_as_nested_map = _convert_dict_items_to_tensor(
        _dict_to_nested_map(features))
    transformed_features = self._preprocessor.TransformFeatures(
        inputs_as_nested_map)
    return _nested_map_to_dict(transformed_features)

  @abc.abstractmethod
  def _init_preprocessor(self):
    """Returns the input_preprocessors.Preprocessor used for this Policy."""
    raise NotImplementedError


class RandomFlip(SingleOpPolicy3D):
  """Flip the world along axis Y as a form of data augmentation.

  When there are leading dimensions, this will flip the boxes with the same
  transformation across all the frames. This is useful when the input is a
  sequence of frames from the same run segment.

  This preprocessor expects features to contain the following keys:
  - features['lasers']['points_xyz'] of shape [..., 3]
  - features['labels']['bboxes_3d'] of shape [..., 7]

  Modifies the following features:
    features['lasers']['points_xyz'], features['labels']['bboxes_3d']
    with the same flipping applied to both.
  """

  def _init_preprocessor(self):
    preprocessor_p = input_preprocessors.RandomFlipY.Params()
    return preprocessor_p.Instantiate()


@pg.members([('max_rotation', pg.typing.Float(min_value=0.),
              'The rotation amount will be randomly picked from '
              '[-max_rotation, max_rotation).')],
            metadata={'init_arg_list': ['max_rotation']})
class RandomRotation(SingleOpPolicy3D):
  """Rotates the world randomly as a form of data augmentation.

  Rotations are performed around the *z-axis*. This assumes that the car is
  always level. In general, we'd like to instead rotate the car on the spot,
  this would then make sense for cases where the car is on a slope.

  When there are leading dimensions, this will rotate the boxes with the same
  transformation across all the frames. This is useful when the input is a
  sequence of frames from the same run segment.

  This preprocessor expects features to contain the following keys:
  - features['lasers']['points_xyz'] of shape [..., 3]
  - features['labels']['bboxes_3d'] of shape [..., 7]

  Modifies the following features:
    features['lasers']['points_xyz'], features['labels']['bboxes_3d']
    with the same rotation applied to both.

  Adds the following features:
    features['world_rot_z'] which contains the rotation applied to the example.
  """

  def _init_preprocessor(self):
    preprocessor_p = input_preprocessors.RandomWorldRotationAboutZAxis.Params()
    preprocessor_p.max_rotation = self.max_rotation
    return preprocessor_p.Instantiate()


@pg.members(
    [('scale_min', pg.typing.Float(min_value=0.), 'Min value for scaling.'),
     ('scale_max', pg.typing.Float(min_value=0.), 'Max value for scaling.')],
    metadata={'init_arg_list': ['scale_min', 'scale_max']})
class WorldScaling(SingleOpPolicy3D):
  """Scale the world randomly as a form of data augmentation.

  This preprocessor expects features to contain the following keys:
  - features['lasers']['points_xyz'] of shape [P, 3]
  - features['labels']['bboxes_3d'] of shape [L, 7]

  Modifies the following features:
    features['lasers']['points_xyz'], features['labels']['bboxes_3d']
    with the same scaling applied to both.
  """

  def _init_preprocessor(self):
    if self.scale_min > self.scale_max:
      raise ValueError('Invalid scaling values min: {} and max {}'.format(
          self.scale_min, self.scale_max))
    preprocessor_p = input_preprocessors.WorldScaling.Params()
    preprocessor_p.scaling = [self.scale_min, self.scale_max]
    return preprocessor_p.Instantiate()


@pg.members([('std_x', pg.typing.Float(min_value=0., default=0.2),
              'Standard deviation of transation noise for X axis.'),
             ('std_y', pg.typing.Float(min_value=0., default=0.2),
              'Standard deviation of transation noise for Y axis.'),
             ('std_z', pg.typing.Float(min_value=0., default=0.2),
              'Standard deviation of transation noise for Z axis.')],
            metadata={'init_arg_list': ['std_x', 'std_y', 'std_z']})
class GlobalTranslateNoise(SingleOpPolicy3D):
  """Add global translation noise of xyz coordinates to points and boxes.

  This preprocessor expects features to contain the following keys:
  - features['lasers']['points_xyz'] of shape [P, 3]
  - features['labels']['bboxes_3d'] of shape [L, 7]

  Modifies the following features:
    features['lasers']['points_xyz'], features['labels']['bboxes_3d']
    with the same random translation noise applied to both.
  """

  def _init_preprocessor(self):
    preprocessor_p = input_preprocessors.GlobalTranslateNoise.Params()
    preprocessor_p.noise_std = [self.std_x, self.std_y, self.std_z]
    return preprocessor_p.Instantiate()


@pg.members(
    [('theta_width', pg.typing.Float(min_value=0., default=0.03),
      'Theta angle width for dropping points.'),
     ('phi_width', pg.typing.Float(min_value=0., default=0.0),
      'Phi angle width for dropping points.'),
     ('distance', pg.typing.Float(min_value=0., default=0.0),
      'Drop points that have larger distance to the'
      'origin than the value given here.'),
     ('keep_prob', pg.typing.Float(min_value=0., default=0.0),
      'keep_prob: 1. = drop no points in the Frustum,'
      '0 = drop all points, between 0 and 1 = down sample the points.'),
     ('drop_union', pg.typing.Bool(default=True),
      'If True, this will drop the union of phi width and theta width. '
      'Otherwise this will drop the intersection.')],
    metadata={
        'init_arg_list': [
            'theta_width', 'phi_width', 'distance', 'keep_prob', 'drop_union'
        ]
    })
class FrustumDropout(SingleOpPolicy3D):
  """Randomly drops out points in a frustum.

  All points are first converted to spherical coordinates, and then a point
  is randomly selected. All points in the frustum around that point within
  a given phi, theta angle width and distance to the original greater than
  a given value are dropped with probability = 1 - keep_prob.

  Here, we can specify whether the dropped frustum is the union or intersection
  of the phi and theta angle filters.

  This preprocessor expects features to contain the following keys:
  - features['lasers']['points_xyz'] of shape [P, 3]
  - features['lasers']['points_feature'] of shape [P, K]

  Optionally points_padding of shape [P] corresponding to the padding.
  if points_padding is None, then all points are considered valid.

  Modifies the following features:
    features['lasers']['points_xyz'], features['lasers']['points_feature'],
    features['lasers']['points_padding'] with points randomly dropped out.
  """

  def _init_preprocessor(self):
    preprocessor_p = input_preprocessors.FrustumDropout.Params()
    preprocessor_p.theta_width = self.theta_width
    preprocessor_p.phi_width = self.phi_width
    preprocessor_p.distance = self.distance
    preprocessor_p.keep_prob = self.keep_prob
    preprocessor_p.drop_type = 'union' if self.drop_union else 'intersection'
    return  preprocessor_p.Instantiate()


@pg.members(
    [('keep_prob', pg.typing.Float(min_value=0., default=0.95),
      'Probability for keeping points.')],
    metadata={'init_arg_list': ['keep_prob']})
class RandomDropLaserPoints(SingleOpPolicy3D):
  """Randomly dropout laser points and the corresponding features.

  This preprocessor expects features to contain the following keys:
  - features['lasers']['points_xyz'] of shape [P, 3]
  - features['lasers']['points_feature'] of shape [P, K]

  Modifies the following features:
    features['lasers']['points_xyz'], features['lasers']['points_feature'].
  """

  def _init_preprocessor(self):
    preprocessor_p = input_preprocessors.RandomDropLaserPoints.Params()
    preprocessor_p.keep_prob = self.keep_prob
    return  preprocessor_p.Instantiate()


@pg.members(
    [
        ('max_rotation', pg.typing.Float(min_value=0.),
         'The rotation amount will be randomly picked from '
         '[-max_rotation, max_rotation).'),
        ('max_scaling_x', pg.typing.Float(min_value=0.).noneable(),
         'If defined, delta s_x parameter is drawn from '
         '[-max_scaling_x, max_scaling_x].'),
        ('max_scaling_y', pg.typing.Float(min_value=0.).noneable(),
         'If defined, delta s_y parameter is drawn from '
         '[-max_scaling_y, max_scaling_y].'),
        ('max_scaling_z', pg.typing.Float(min_value=0.).noneable(),
         'If defined, delta s_z parameter is drawn from '
         '[-max_scaling_z, max_scaling_z].'),
        ('max_shearing_xy', pg.typing.Float(min_value=0.).noneable(),
         'If defined, shearing parameter sh_x^y is drawn from '
         '[-max_shearing_xy, max_shearing_xy].'),
        ('max_shearing_xz', pg.typing.Float(min_value=0.).noneable(),
         'If defined, shearing parameter sh_x^z is drawn from '
         '[-max_shearing_xz, max_shearing_xz].'),
        ('max_shearing_yx', pg.typing.Float(min_value=0.).noneable(),
         'If defined, shearing parameter sh_y^x is drawn from '
         '[-max_shearing_yx, max_shearing_yx].'),
        ('max_shearing_yz', pg.typing.Float(min_value=0.).noneable(),
         'If defined, shearing parameter sh_y^z is drawn from '
         '[-max_shearing_yz, max_shearing_yz].'),
        ('max_shearing_zx', pg.typing.Float(min_value=0.).noneable(),
         'If defined, shearing parameter sh_z^x is drawn from '
         '[-max_shearing_zx, max_shearing_zx].'),
        ('max_shearing_zy', pg.typing.Float(min_value=0.).noneable(),
         'If defined, shearing parameter sh_z^y is drawn from '
         '[-max_shearing_zy, max_shearing_zy].'),
        ('max_num_points_per_bbox', pg.typing.Int(default=16384),
         'The maximum number of points that fall within a bounding box. '
         'Bounding boxes with more points than this value will '
         'have some points droppped.')
    ],
    metadata={
        'init_arg_list': [
            'max_rotation', 'max_scaling_x', 'max_scaling_y', 'max_scaling_z',
            'max_shearing_xy', 'max_shearing_xz', 'max_shearing_yx',
            'max_shearing_yz', 'max_shearing_zx', 'max_shearing_zy',
            'max_num_points_per_bbox'
        ]
    })
class RandomBBoxTransform(SingleOpPolicy3D):
  """Randomly transform bounding boxes and the points inside them.

  This preprocessor expects features to contain the following items:
  - features['lasers']['points_xyz'] of shape [P, 3]
  - features['lasers']['points_feature'] of shape [P, K]
  - features['lasers']['points_padding'] of shape [P]
  - features['labels']['bboxes_3d'] of shape [L, 7]
  - features['labels']['bboxes_3d_mask'] of shape [L]

  Modifies the following features:
    features['lasers']['points_{xyz,feature,padding}'],
    features['labels']['bboxes_3d'] with the
    transformed bounding boxes and points.
  """

  def _init_preprocessor(self):
    preprocessor_p = input_preprocessors.RandomBBoxTransform.Params()

    all_scalings = [self.max_scaling_x, self.max_scaling_y, self.max_scaling_z]
    if not (all(p is None for p in all_scalings) or
            all(p is not None for p in all_scalings)):
      raise ValueError(
          'Either zero or all max_scaling_{x, y, z} params must be defined.')
    max_scaling = all_scalings if self.max_scaling_x is not None else None

    all_shearings = [self.max_shearing_xy, self.max_shearing_xz,
                     self.max_shearing_yx, self.max_shearing_yz,
                     self.max_shearing_zx, self.max_shearing_zy]
    if not (all(p is None for p in all_shearings) or
            all(p is not None for p in all_shearings)):
      raise ValueError(
          'Either zero or all max_shearing_{xy, xz, yx, yz, zx, xy} params '
          'must be defined.')
    max_shearing = all_shearings if self.max_shearing_xy is not None else None

    preprocessor_p.max_rotation = self.max_rotation
    preprocessor_p.max_scaling = max_scaling
    preprocessor_p.max_shearing = max_shearing
    preprocessor_p.max_num_points_per_bbox = self.max_num_points_per_bbox
    return preprocessor_p.Instantiate()


def create_basic_policy(
    bbox_rotation=np.pi / 20,
    world_rotation=np.pi / 4,
    world_scale_min=0.95,
    world_scale_max=1.05,
    frustum_dropout_theta=0.03,
    frustum_dropout_phi=0.0):
  """Creates a basic 3D augmentation policy."""
  return Sequential(children=[
      RandomBBoxTransform(bbox_rotation),
      RandomFlip(),
      RandomRotation(world_rotation),
      WorldScaling(scale_min=world_scale_min, scale_max=world_scale_max),
      FrustumDropout(theta_width=frustum_dropout_theta,
                     phi_width=frustum_dropout_phi)])


def get_policy_from_str(aug_str):
  """Converts json augmentation string to AugmentPolicy3D object.

  The trainer code should pass the `nas_params_str` sent by the nas-service
  as 'aug_str' to this function :-

    policy = get_policy_from_str(aug_str=nas_params_str)
    augmented_features = policy(features)


  Args:
    aug_str: The json str for the policy-spec-builder.

  Returns:
    The resulting AugmentPolicy3D object.
  """
  spec = pg.from_json_str(aug_str)
  if isinstance(spec, policies_search_space.BasicSearchSpaceSpecBuilder):
    policy = create_basic_policy(
        bbox_rotation=spec.bbox_rotation,
        world_rotation=spec.world_rotation,
        world_scale_min=spec.world_scale_min,
        world_scale_max=spec.world_scale_max,
        frustum_dropout_theta=spec.frustum_dropout_theta,
        frustum_dropout_phi=spec.frustum_dropout_phi)
    return policy
  else:
    raise ValueError('Could not find matching policy-spec-builder for %s!' %
                     aug_str)
