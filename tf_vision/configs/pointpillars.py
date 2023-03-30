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

"""PointPillars for detection configuration definition."""
# pytype: disable=wrong-keyword-args
import dataclasses
from typing import Optional

from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling import optimization
from tf_vision.pointpillars.configs import pointpillars as pointpillars_cfg


@dataclasses.dataclass
class TunablePointPillarsModel(pointpillars_cfg.PointPillarsModel):
  """Definition of tunable PointPillars model."""
  block_specs_json: Optional[str] = None


@dataclasses.dataclass
class PointPillarsTask(pointpillars_cfg.PointPillarsTask):
  """Definition of RetinaNet task."""
  # A placeholder for task registry.


# pylint: disable=unexpected-keyword-arg
@exp_factory.register_config_factory('pointpillars_nas')
def pointpillars_nas():
  """PointPillars baseline config."""
  return cfg.ExperimentConfig(
      runtime=cfg.RuntimeConfig(mixed_precision_dtype='float32'),
      task=PointPillarsTask(
          model=TunablePointPillarsModel(
              classes='vehicle',
              num_classes=2,
              min_level=1,
              max_level=3
          ),
      ),
      trainer=cfg.TrainerConfig(
          train_steps=100,
          validation_steps=100,
          validation_interval=10,
          steps_per_loop=10,
          summary_interval=10,
          checkpoint_interval=10,
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
                      'decay_steps': 100,
                      'initial_learning_rate': 0.16,
                  }
              },
              'warmup': {
                  'type': 'linear',
                  'linear': {
                      'warmup_steps': 10,
                      'warmup_learning_rate': 0.016
                  }
              }
          })),
      restrictions=[
          'task.train_data.is_training != None',
          'task.validation_data.is_training != None',
      ])
# pylint: enable=unexpected-keyword-arg
