# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Evaluator factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf1.detection.evaluation import coco_evaluator


def evaluator_generator(params):
  """Generator function for various evaluators."""
  if params.type == 'box':
    evaluator = coco_evaluator.COCOEvaluator(
        annotation_file=params.val_json_file,
        include_mask=False,
        per_category_metrics=params.per_category_metrics)
  elif params.type == 'box_and_mask':
    evaluator = coco_evaluator.COCOEvaluator(
        annotation_file=params.val_json_file,
        include_mask=True,
        per_category_metrics=params.per_category_metrics)
  elif params.type == 'shapemask_box_and_mask':
    evaluator = coco_evaluator.ShapeMaskCOCOEvaluator(
        mask_eval_class=params.mask_eval_class,
        annotation_file=params.val_json_file,
        include_mask=True,
        per_category_metrics=params.per_category_metrics)
  elif params.type == 'lvis_box':
    evaluator = coco_evaluator.LVISEvaluator(
        annotation_file=params.val_json_file, include_mask=False)
  elif params.type == 'lvis_box_and_mask':
    evaluator = coco_evaluator.LVISEvaluator(
        annotation_file=params.val_json_file, include_mask=True)
  else:
    raise ValueError('The detection evaluation type `{}` is not supported.'
                     .format(params.type))

  return evaluator
