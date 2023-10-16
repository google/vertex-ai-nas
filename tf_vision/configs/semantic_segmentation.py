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

"""Semantic segmentation configuration definition."""
# pytype: disable=wrong-keyword-args
import dataclasses
import os
from tf_vision.configs import backbones
from tf_vision.configs import decoders
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.vision.beta.configs import common
from official.vision.beta.configs import decoders as base_decoders
from official.vision.beta.configs import semantic_segmentation as semantic_segmentation_cfg


@dataclasses.dataclass
class SemanticSegmentationModel(
    semantic_segmentation_cfg.SemanticSegmentationModel):
  """Semantic segmentation model config."""
  backbone: backbones.Backbone = dataclasses.field(
      default_factory=lambda: backbones.Backbone(  # pylint: disable=g-long-lambda
          type='tunable_spinenet', tunable_spinenet=backbones.TunableSpineNet()
      )
  )
  decoder: decoders.Decoder = dataclasses.field(
      default_factory=lambda: decoders.Decoder(  # pylint: disable=g-long-lambda
          type='identity', identity=decoders.Identity()
      )
  )


@dataclasses.dataclass
class SemanticSegmentationTask(
    semantic_segmentation_cfg.SemanticSegmentationTask):
  """Definition of semantic segmentation task."""


# PASCAL VOC 2012 Dataset
PASCAL_INPUT_PATH_BASE = 'pascal_voc_seg'


@exp_factory.register_config_factory('semantic_segmentation_nas')
def semantic_segmentation():
  """Image segmentation on imagenet with resnet deeplabv3."""
  config = cfg.ExperimentConfig(
      task=SemanticSegmentationTask(
          model=SemanticSegmentationModel(
              num_classes=21,
              input_size=[None, None, 3],
              backbone=backbones.Backbone(
                  type='tunable_spinenet',
                  tunable_spinenet=backbones.TunableSpineNet(
                      model_id='49', init_stochastic_depth_rate=0.2)),
              decoder=decoders.Decoder(
                  type='aspp',
                  aspp=base_decoders.ASPP(level=3, dilation_rates=[12, 24,
                                                                   36])),
              head=semantic_segmentation_cfg.SegmentationHead(
                  level=3, num_convs=0, prediction_kernel_size=3),
              norm_activation=common.NormActivation(
                  activation='swish',
                  norm_momentum=0.9997,
                  norm_epsilon=1e-3,
                  use_sync_bn=True)),
          losses=semantic_segmentation_cfg.Losses(
              l2_weight_decay=1e-4, top_k_percent_pixels=1.0),
          train_data=semantic_segmentation_cfg.DataConfig(
              input_path=os.path.join(PASCAL_INPUT_PATH_BASE, 'train_aug*'),
              output_size=[640, 640],
              is_training=True,
              global_batch_size=32,
              aug_rand_hflip=True,
              aug_scale_min=0.5,
              aug_scale_max=2.0),
          validation_data=semantic_segmentation_cfg.DataConfig(
              input_path=os.path.join(PASCAL_INPUT_PATH_BASE, 'val*'),
              output_size=[640, 640],
              is_training=False,
              global_batch_size=16,
              resize_eval_groundtruth=False,
              groundtruth_padded_size=[512, 512],
              drop_remainder=False)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=100,
          summary_interval=500,
          checkpoint_interval=500,
          train_steps=20000,
          validation_steps=90,
          validation_interval=500,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'cosine',
                  'cosine': {
                      'initial_learning_rate': 0.01,
                      'decay_steps': 20000,
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 500,
                      'warmup_learning_rate': 0
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])

  return config
