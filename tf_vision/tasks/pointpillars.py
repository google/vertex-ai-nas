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

"""PointPillars task definition."""

from typing import Any, Optional

from absl import logging
from tf_vision import utils
from tf_vision.configs import pointpillars as cfg
from tf_vision.modeling import tunable_pointpillars
import tensorflow as tf

from official.core import task_factory
from tf_vision.pointpillars.tasks import pointpillars as base_task


@task_factory.register_task_cls(cfg.PointPillarsTask)
class PointPillarsTask(base_task.PointPillarsTask):
  """A single-replica view of training procedure."""

  def build_model(self):
    """Build tunable model."""
    logging.info('Start building tunable model.')
    train_batch_size = base_task.get_batch_size_per_replica(
        self.task_config.train_data.global_batch_size)
    eval_batch_size = base_task.get_batch_size_per_replica(
        self.task_config.validation_data.global_batch_size)
    logging.info('Per replica batch size: train %d, eval %d',
                 train_batch_size, eval_batch_size)
    model = tunable_pointpillars.build_tunable_pointpillars(
        model_config=self.task_config.model,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )
    return model

  def initialize(self, model):
    """Loading pretrained checkpoint."""
    if not self.task_config.init_checkpoint:
      logging.info('No checkpoint specified, initialize model.')
      return

    ckpt_dir_or_file = self.task_config.init_checkpoint
    if tf.io.gfile.isdir(ckpt_dir_or_file):
      ckpt_dir_or_file = tf.train.latest_checkpoint(ckpt_dir_or_file)

    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.read(ckpt_dir_or_file)
    status.expect_partial().assert_existing_objects_matched()
    logging.info('Finished loading pretrained checkpoint from %s',
                 ckpt_dir_or_file)

  def reduce_aggregated_logs(self,
                             aggregated_logs,
                             global_step = None):
    """Called after eval_end to calculate metrics."""
    logs = super().reduce_aggregated_logs(aggregated_logs, global_step)
    lite_logs = {}
    for k in ['loss', utils.MODEL_MAP, utils.MODEL_MAPH]:
      if k in logs:
        lite_logs[k] = logs[k]
    return lite_logs
