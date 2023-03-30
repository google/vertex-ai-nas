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

"""Image segmentation task definition."""
from typing import Optional

from tf_vision.configs import semantic_segmentation as semantic_segmentation_cfg
from tf_vision.dataloaders import segmentation_input
import tensorflow as tf

from official.common import dataset_fn
from official.core import task_factory
from official.vision.beta.configs import semantic_segmentation as base_semantic_segmentation_cfg
from official.vision.beta.dataloaders import input_reader_factory
from official.vision.beta.dataloaders import segmentation_input as base_segmentation_input
from official.vision.beta.dataloaders import tfds_factory
from official.vision.beta.tasks import semantic_segmentation as base_semantic_segmentation_task


@task_factory.register_task_cls(
    semantic_segmentation_cfg.SemanticSegmentationTask)
class SemanticSegmentationTask(
    base_semantic_segmentation_task.SemanticSegmentationTask):
  """A task for semantic segmentation."""

  def build_inputs(self,
                   params,
                   input_context = None):
    """Builds classification input."""

    ignore_label = self.task_config.losses.ignore_label

    if params.tfds_name:
      decoder = tfds_factory.get_segmentation_decoder(params.tfds_name)
    else:
      decoder = base_segmentation_input.Decoder()

    parser = segmentation_input.Parser(
        aug_policy=params.aug_policy,
        output_size=params.output_size,
        crop_size=params.crop_size,
        ignore_label=ignore_label,
        resize_eval_groundtruth=params.resize_eval_groundtruth,
        groundtruth_padded_size=params.groundtruth_padded_size,
        aug_scale_min=params.aug_scale_min,
        aug_scale_max=params.aug_scale_max,
        aug_rand_hflip=params.aug_rand_hflip,
        dtype=params.dtype)

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    return reader.read(input_context=input_context)
