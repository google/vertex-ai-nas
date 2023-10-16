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

"""Image classification configuration definition."""
# pytype: disable=wrong-keyword-args
import dataclasses
from tf_vision.configs import backbones

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from official.vision.beta.configs import common
from official.vision.beta.configs import image_classification as image_classification_cfg


@dataclasses.dataclass
class ImageClassificationModel(image_classification_cfg.ImageClassificationModel
                              ):
  """The model config."""
  min_level: int = 3  # Only used for SpineNet.
  max_level: int = 7  # Only used for SpineNet.
  backbone: backbones.Backbone = dataclasses.field(
      default_factory=lambda: backbones.Backbone(  # pylint: disable=g-long-lambda
          type='tunable_mnasnet',
          tunable_mnasnet=backbones.TunableMnasNet(endpoints_num_filters=256),
      )
  )


# ImageNet training set contains 1251139 samples.
IMAGENET_TRAIN = 'imagenet-2012-tfrecord/train-00???-of-01024'
# ImageNet validation set contains 30028 samples.
IMAGENET_VAL = 'imagenet-2012-tfrecord/train-01???-of-01024'


@exp_factory.register_config_factory('image_classification_nas')
def image_classification():
  """Image classification general config."""
  return cfg.ExperimentConfig(
      task=image_classification_cfg.ImageClassificationTask(
          model=ImageClassificationModel(
              num_classes=1001,
              input_size=[224, 224, 3],
              norm_activation=common.NormActivation(
                  norm_momentum=0.99, norm_epsilon=0.001, use_sync_bn=True),
              dropout_rate=0.2),
          losses=image_classification_cfg.Losses(label_smoothing=0.1),
          train_data=image_classification_cfg.DataConfig(
              input_path=IMAGENET_TRAIN,
              is_training=True,
              global_batch_size=1024),
          validation_data=image_classification_cfg.DataConfig(
              input_path=IMAGENET_VAL,
              is_training=False,
              global_batch_size=1024)),
      trainer=cfg.TrainerConfig(
          steps_per_loop=1222,
          summary_interval=1222,
          checkpoint_interval=1222,
          train_steps=12218,  # 10 epochs.
          validation_steps=30,
          validation_interval=1222,
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
                      'boundaries': [3666, 6721, 9776],
                      'values': [0.064, 0.06208, 0.0602, 0.0584]
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 611
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None'
      ])
