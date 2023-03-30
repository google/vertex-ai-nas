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
"""Tunable AutoAugment search.

Cubuk, et al. AutoAugment: Learning Augmentation Policies from Data
https://arxiv.org/abs/1805.09501

Zoph, et al. Learning Data Augmentation Strategies for Object Detection.
Arxiv: https://arxiv.org/abs/1906.11172
"""

import pyglove as pg

from nas_lib.augmentation_2d import policies_tf2 as policies
from nas_architecture import tunable_autoaugment_search_space  # pylint: disable=unused-import


@pg.members(
    [('auto_aug_policy',
      pg.typing.List(pg.typing.Any()),
      'AutoAugment policy as list of three lists containing the '
      'operation_ind choices, magnitude choices, and probability choices.'
      )],
    metadata={'init_arg_list': ['auto_aug_policy']})
class AutoAugmentBuilderBase(pg.Object):
  """Base Class AutoAugment search policy."""

  _op_fns = [
      lambda m, p: policies.ApplyWithProbability(policies.Equalize(), p),
      lambda m, p: policies.ApplyWithProbability(policies.Solarize(m), p),
      lambda m, p: policies.ApplyWithProbability(policies.Color(m), p),
      lambda m, p: policies.ApplyWithProbability(policies.CutoutImage(m), p),
      lambda m, p: policies.ApplyWithProbability(policies.SolarizeAdd(m), p),
      lambda m, p: policies.ApplyWithProbability(policies.TranslateX(m), p),
      lambda m, p: policies.ApplyWithProbability(policies.TranslateY(m), p),
      lambda m, p: policies.ApplyWithProbability(policies.ShearX(m), p),
      lambda m, p: policies.ApplyWithProbability(policies.ShearY(m), p),
      lambda m, p: policies.ApplyWithProbability(policies.Rotate(m), p),
      lambda m, p: policies.ApplyWithProbability(policies.Sharpness(m), p),
      lambda m, p: policies.ApplyWithProbability(policies.CutoutImage(m), p),
  ]

  @classmethod
  def num_ops(cls):
    """Returns the number of possible augmentation operations."""
    return len(cls._op_fns)

  def __call__(self):
    """Builds the augmentation policy.

    Returns:
      The augmentation policy.

    Raises:
      ValueError: If this object is not deterministic.
      ValueError: If the number of choices differ between ops
        magnitudes, and probabilities.
    """
    if not self.is_deterministic:
      raise ValueError(
          'Object must be deterministic to build the augmentation Policy.')

    op_ind_choices, m_choices, p_choices = self.auto_aug_policy
    if not len(op_ind_choices) == len(m_choices) == len(p_choices):
      raise ValueError('Different number of choices.')

    sub_policies = []
    for op_inds, ms, ps in zip(op_ind_choices, m_choices, p_choices):
      if not len(op_inds) == len(ms) == len(ps):
        raise ValueError('Different number of choices.')
      ops = []
      for op_ind, m, p in zip(op_inds, ms, ps):
        op_fn = self._op_fns[op_ind]
        ops.append(op_fn(m, p))
      sub_policies.append(policies.Sequential(ops))

    return policies.Choice(sub_policies)


class AutoAugmentClassificationBuilder(AutoAugmentBuilderBase):
  """AutoAugment search policy for classification."""

  _op_fns = AutoAugmentBuilderBase._op_fns + [
      lambda m, p: policies.ApplyWithProbability(policies.Invert(), p),
      lambda m, p: policies.ApplyWithProbability(policies.Posterize(m), p),
      lambda m, p: policies.ApplyWithProbability(policies.Contrast(m), p),
      lambda m, p: policies.ApplyWithProbability(policies.Brightness(m), p),
      lambda m, p: policies.ApplyWithProbability(policies.Sharpness(m), p),
  ]


class AutoAugmentDetectionBuilder(AutoAugmentBuilderBase):
  """AutoAugment search policy for detection."""

  _op_fns = AutoAugmentBuilderBase._op_fns + [
      policies.SolarizeOnlyBBoxes,
      policies.CutoutOnlyBBoxes,
      policies.TranslateXOnlyBBoxes,
      policies.TranslateYOnlyBBoxes,
      policies.ShearXOnlyBBoxes,
      policies.ShearYOnlyBBoxes,
      policies.RotateOnlyBBoxes,
      lambda m, p: policies.EqualizeOnlyBBoxes(p),
      lambda m, p: policies.FlipOnlyBBoxes(p),
    ]


class AutoAugmentSegmentationBuilder(AutoAugmentBuilderBase):
  """AutoAugment search policy for segmentation."""

  _op_fns = AutoAugmentBuilderBase._op_fns + [
      lambda m, p: policies.ApplyWithProbability(  # pylint: disable=g-long-lambda
          policies.CutoutSegmentation(m), p)
  ]
