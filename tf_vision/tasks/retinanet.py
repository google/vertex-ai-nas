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

"""RetinaNet task definition."""
from typing import Optional

from tf_vision.configs import retinanet as retinanet_cfg
from tf_vision.dataloaders import retinanet_input
import tensorflow as tf

from official.common import dataset_fn
from official.core import task_factory
from official.vision.beta.configs import retinanet as base_retinanet_cfg
from official.vision.beta.dataloaders import input_reader_factory
from official.vision.beta.dataloaders import tf_example_decoder
from official.vision.beta.dataloaders import tf_example_label_map_decoder
from official.vision.beta.tasks import retinanet as base_retinanet_task


@task_factory.register_task_cls(retinanet_cfg.RetinaNetTask)
class RetinaNetTask(base_retinanet_task.RetinaNetTask):
  """A single-replica view of training procedure.

  RetinaNet task provides artifacts for training/evalution procedures, including
  loading/iterating over Datasets, initializing the model, calculating the loss,
  post-processing, and customized metrics with reduction.
  """

  def build_inputs(self,
                   params,
                   input_context = None):
    """Build input dataset."""
    decoder_cfg = params.decoder.get()
    if params.decoder.type == 'simple_decoder':
      decoder = tf_example_decoder.TfExampleDecoder(
          regenerate_source_id=decoder_cfg.regenerate_source_id)
    elif params.decoder.type == 'label_map_decoder':
      decoder = tf_example_label_map_decoder.TfExampleDecoderLabelMap(
          label_map=decoder_cfg.label_map,
          regenerate_source_id=decoder_cfg.regenerate_source_id)
    else:
      raise ValueError('Unknown decoder type: {}!'.format(params.decoder.type))

    parser = retinanet_input.Parser(
        aug_policy=params.parser.aug_policy,
        output_size=self.task_config.model.input_size[:2],
        min_level=self.task_config.model.min_level,
        max_level=self.task_config.model.max_level,
        num_scales=self.task_config.model.anchor.num_scales,
        aspect_ratios=self.task_config.model.anchor.aspect_ratios,
        anchor_size=self.task_config.model.anchor.anchor_size,
        dtype=params.dtype,
        match_threshold=params.parser.match_threshold,
        unmatched_threshold=params.parser.unmatched_threshold,
        aug_rand_hflip=params.parser.aug_rand_hflip,
        aug_scale_min=params.parser.aug_scale_min,
        aug_scale_max=params.parser.aug_scale_max,
        skip_crowd_during_training=params.parser.skip_crowd_during_training,
        max_num_instances=params.parser.max_num_instances)

    reader = input_reader_factory.input_reader_generator(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))
    dataset = reader.read(input_context=input_context)

    return dataset
