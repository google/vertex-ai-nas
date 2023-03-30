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

"""RetinaNet for detection configuration definition."""
# pytype: disable=wrong-keyword-args
import dataclasses
from tf_vision.configs import backbones
from tf_vision.configs import decoders

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.vision.beta.configs import common
from official.vision.beta.configs import retinanet as retinanet_cfg


@dataclasses.dataclass
class RetinaNet(retinanet_cfg.RetinaNet):
  """Definition of RetinaNet, containing backbone and decoder."""
  backbone: backbones.Backbone = backbones.Backbone(
      type='tunable_spinenet', tunable_spinenet=backbones.TunableSpineNet())
  decoder: decoders.Decoder = decoders.Decoder(
      type='identity', identity=decoders.Identity())


@dataclasses.dataclass
class RetinaNetTask(retinanet_cfg.RetinaNetTask):
  """Definition of RetinaNet task."""


# Use shard 0-239 as training set, 110895 examples.
COCO_TRAIN = 'coco/train-00([0-1][0-9][0-9]|2[0-3][0-9])-of-?????.tfrecord'
# Use shard 240-255 as validation set, 7392 samples.
COCO_VAL = 'coco/train-002[4-5]?-of-?????.tfrecord'
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64


@exp_factory.register_config_factory('retinanet_nas')
def retinanet():
  """COCO object detection with RetinaNet using TunableSpineNet backbone."""

  return cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='float32'),
      task=RetinaNetTask(
          model=RetinaNet(
              backbone=backbones.Backbone(
                  type='tunable_spinenet',
                  tunable_spinenet=backbones.TunableSpineNet(
                      init_stochastic_depth_rate=0.2,
                      filter_size_scale=0.5,
                      resample_alpha=0.25,
                      endpoints_num_filters=64,
                      min_level=3,
                      max_level=7)),
              decoder=decoders.Decoder(
                  type='identity', identity=decoders.Identity()),
              head=retinanet_cfg.RetinaNetHead(num_filters=64),
              anchor=retinanet_cfg.Anchor(anchor_size=3),
              norm_activation=common.NormActivation(
                  norm_momentum=0.997,
                  norm_epsilon=0.0001,
                  use_sync_bn=True,
                  activation='relu'),
              num_classes=91,
              input_size=[512, 512, 3],
              min_level=3,
              max_level=7),
          losses=retinanet_cfg.Losses(l2_weight_decay=4e-5),
          train_data=retinanet_cfg.DataConfig(
              input_path=COCO_TRAIN,
              is_training=True,
              global_batch_size=TRAIN_BATCH_SIZE,
              parser=retinanet_cfg.Parser(
                  aug_rand_hflip=True, aug_scale_min=0.5, aug_scale_max=2.0)),
          validation_data=retinanet_cfg.DataConfig(
              input_path=COCO_VAL,
              is_training=False,
              global_batch_size=EVAL_BATCH_SIZE)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=1733,
          summary_interval=1733,
          checkpoint_interval=1733,
          train_steps=17330,  # 10 epochs.
          validation_steps=116,
          validation_interval=1733,
          optimizer_config=optimization.OptimizationConfig({
              'optimizer': {
                  'type': 'sgd',
                  'sgd': {
                      'momentum': 0.9
                  }
              },
              'learning_rate': {
                  'type': 'stepwise',
                  'stepwise': {
                      'boundaries': [13864],
                      'values': [0.08, 0.008]
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 500
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
      ])
