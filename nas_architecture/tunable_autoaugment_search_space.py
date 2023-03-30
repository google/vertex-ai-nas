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
"""AutoAugment search space definition.

Cubuk, et al. AutoAugment: Learning Augmentation Policies from Data
https://arxiv.org/abs/1805.09501

Zoph, et al. Learning Data Augmentation Strategies for Object Detection.
Arxiv: https://arxiv.org/abs/1906.11172
"""

import pyglove as pg

# The number of augmentation operations for `prebuilt` AutoAugmentXXXBuilder in
# tunable_auto_augment.py. Their values are validated through unit tests.
BASE_AUG_OPS_COUNT = 12
CLASSIFICATION_OPS_COUNT = BASE_AUG_OPS_COUNT + 5
DETECTION_OPS_COUNT = BASE_AUG_OPS_COUNT + 9
SEGMENTATION_OPS_COUNT = BASE_AUG_OPS_COUNT + 1


def autoaugment_search_space(
    total_num_ops,
    num_ops_per_sub_policy = 2,
    num_sub_policies = 3):
  """Creates an AutoAugment search space.

  AutoAugment policies.

  Cubuk, et al. AutoAugment: Learning Augmentation Policies from Data
  https://arxiv.org/abs/1805.09501

  Zoph, et al. Learning Data Augmentation Strategies for Object Detection.
  Arxiv: https://arxiv.org/abs/1906.11172

  Args:
    total_num_ops: The total number of all ops.
    num_ops_per_sub_policy: The number of operations to choose
      for each sub-policy.
    num_sub_policies: The total number of sub-policies to search for.

  Returns:
    A list representing the search space of operation_ind choices, magnitude
    choices, and probability choices. It should be used with
    AutoAugment{Classification, Detection, Segmentation}Builder.

  Raises:
    ValueError: If task value is not one of the supported tasks.
  """
  op_ind_candidates = list(range(total_num_ops))
  m_candidates = list(range(11))
  p_candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

  op_ind_choices = []
  m_choices = []
  p_choices = []
  for _ in range(num_sub_policies):
    op_ind_choices.append(pg.manyof(
        k=num_ops_per_sub_policy,
        candidates=op_ind_candidates,
        choices_distinct=False,
        choices_sorted=False))
    m_choices.append(pg.manyof(
        k=num_ops_per_sub_policy,
        candidates=m_candidates,
        choices_distinct=False,
        choices_sorted=False))
    p_choices.append(pg.manyof(
        k=num_ops_per_sub_policy,
        candidates=p_candidates,
        choices_distinct=False,
        choices_sorted=False))
  return pg.List([op_ind_choices, m_choices, p_choices])
