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

"""Library containing training functions using TF Vision trainer."""
import enum
import json
import os
import time
from typing import Dict, Optional, Sequence

from absl import logging
import cloud_nas_utils

from proxy_task import proxy_task_utils

from tf_vision import latency_utils
from tf_vision import utils
import orbit
import six
import tensorflow as tf

from official.common import distribute_utils
from official.core import actions
from official.core import config_definitions
from official.core import task_factory
from official.core import train_utils
from official.modeling import performance


# pylint: disable=protected-access
def run_train_and_eval_in_loop(
    params,
    controller,
    model_dir = "",
    accuracy_key = ""):
  """Runs train and eval in for loop and returns the list of all the metrics.

  Args:
    params: An input ExperimentConfig config.
    controller: An orbit Controller instance.
    model_dir: Model directory.
    accuracy_key: Key for accuracy metric.

  Returns:
    A list of dictionary of metrics.
  """
  logging.warning("run_train_and_eval_in_loop ignores params.steps_per_loop "
                  "and params.checkpoint_interval, it uses "
                  "params.validation_interval as the steps for one cycle of "
                  "training, evaluation and saving checkpoints.")
  metrics_list = []
  train_steps = params.trainer.train_steps
  eval_steps = params.trainer.validation_steps
  eval_interval = params.trainer.validation_interval
  current_step = int(controller.global_step.numpy())
  logging.info("Start run_train_and_eval_in_loop from step: %d", current_step)

  # If a training job aborts during the loop and gets restarted from the latest
  # checkpoint, the current_step will be restored from the checkpoint. The loop
  # will start from the checkpointed step.
  while current_step < train_steps:

    if proxy_task_utils.get_stop_training(
        model_dir,
        end_training_cycle_step=current_step - 1,
        total_training_steps=train_steps):
      break

    interval = min(train_steps - current_step, eval_interval)
    next_step = current_step + interval
    logging.info(
        "Train from step: %d to step: %d, num steps: %d, "
        "total num steps: %d", current_step, next_step, interval, train_steps)
    tic = time.time()
    controller._train_n_steps(interval)
    toc = time.time()
    metrics = controller.evaluate(steps=eval_steps)
    if not metrics:
      raise ValueError("Evaluation metrics are unavailable.")
    metrics_list.append(metrics)

    proxy_task_utils.update_trial_training_accuracy_metric(
        model_dir=model_dir,
        accuracy=float(metrics[accuracy_key]),
        begin_training_cycle_step=current_step,
        end_training_cycle_step=next_step - 1,
        training_cycle_time_in_secs=toc - tic,
        total_training_steps=train_steps)

    # Read current step from controller for next cycle, it should be correctly
    # incremented during training.
    current_step = int(controller.global_step.numpy())

    # Save checkpoints at the end of the loop, to ensure that model states
    # should only be checkpointed after all works above have been done. If
    # the job aborts at anywhere before this in a cycle, the uncompleted cycle
    # won't be checkpointed.
    controller._maybe_save_checkpoint(check_interval=False)
  return metrics_list  # pytype: disable=bad-return-type  # numpy-scalars


# pylint: enable=protected-access


def run_training(params,
                 session,
                 mode,
                 model,
                 model_dir,
                 save_summary = True,
                 constraint_type = utils.LatencyConstraintType.NONE,
                 target_latency_or_flops = 0.0,
                 target_memory = 0,
                 skip_eval = False,
                 multiple_eval_during_search = True,
                 retrain_search_job_trials = None):
  """Runs model training."""
  logging.info("Starting training.")

  # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
  # can have significant impact on model speeds by utilizing float16 in case
  # of GPUs, and bfloat16 in the case of TPUs. loss_scale takes effect only
  # when dtype is float16.
  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype)

  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu)

  with distribution_strategy.scope():
    task = task_factory.get_task(params.task, logging_dir=model_dir)
    trainer = train_utils.create_trainer(
        params,
        task,
        train="train" in mode,
        evaluate=("eval" in mode) or not skip_eval,
        checkpoint_exporter=train_utils.maybe_create_best_ckpt_exporter(
            params, model_dir))

  model_stats = utils.get_model_flops_and_params(task.build_model(), params)

  if trainer.checkpoint:
    checkpoint_manager = tf.train.CheckpointManager(
        trainer.checkpoint,
        directory=model_dir,
        max_to_keep=params.trainer.max_to_keep,
        step_counter=trainer.global_step,
        checkpoint_interval=params.trainer.checkpoint_interval,
        init_fn=trainer.initialize)
  else:
    checkpoint_manager = None

  controller = orbit.Controller(
      strategy=distribution_strategy,
      trainer=trainer if "train" in mode else None,
      evaluator=trainer,
      global_step=trainer.global_step,
      steps_per_loop=params.trainer.steps_per_loop,
      checkpoint_manager=checkpoint_manager,
      summary_dir=os.path.join(model_dir, "train") if (save_summary) else None,
      eval_summary_dir=os.path.join(model_dir,
                                    params.trainer.validation_summary_subdir) if
      (save_summary) else None,
      summary_interval=params.trainer.summary_interval if
      (save_summary) else None,
      train_actions=actions.get_train_actions(
          params, trainer, model_dir, checkpoint_manager=checkpoint_manager),
      eval_actions=actions.get_eval_actions(params, trainer, model_dir))

  # Save configurations to trial directory.
  train_utils.serialize_config(params, model_dir)

  # This key is used to:
  # 1. sort multiple evaluation metrics list;
  # 2. be the name of the final score, like combining accuracy and latency.
  accuracy_key = utils.get_accuracy_key(task=params.task)

  # This dict is used to store the best metric chosen from multiple evaluations.
  # Take classification for example, it includes like 'top_1_accuracy',
  # 'top_5_accuracy' and 'validation_loss'.
  metrics = {}

  # This dict is used to store anything we want the reporter to report,
  # other than the metrics[accuracy_key]. Like latency, model memory.
  model_analysis = {}

  if mode == "train":
    if skip_eval:
      # This is only used during debugging.
      with distribution_strategy.scope():
        controller.train(steps=params.trainer.train_steps)
      return

    if multiple_eval_during_search or retrain_search_job_trials:
      # We want to enable multiple evaluations when either the flag
      # '--multiple_eval_during_search' is explicitly set to true, or it's a
      # stage-2 train job (which also gets triggered similar to a search job
      # with multiple trials).
      # For proxy task with large number of epochs
      # (for ex: augmentation-search)
      # we need to see eval-metrics in Tensorboard. So we need to run
      # multiple evaluations and choose the best metric to report.
      with distribution_strategy.scope():
        metrics_list = run_train_and_eval_in_loop(
            params=params,
            controller=controller,
            model_dir=model_dir,
            accuracy_key=accuracy_key)
      logging.info("Metrics list is: %s.", metrics_list)
      metrics = max(metrics_list, key=lambda x: x[accuracy_key])
      logging.info("Best metrics is: %s.", metrics)
    else:
      # For model-search, the proxy task is a small training job which should
      # run fast. So we just run one evaluation.
      with distribution_strategy.scope():
        controller.train(steps=params.trainer.train_steps)
        metrics = controller.evaluate(steps=params.trainer.validation_steps)
      if not metrics:
        raise ValueError("Evaluation metrics are unavailable.")

    # For classification models, TF1 trainer defines the accuracy key to be
    # 'top_1_accuracy', and TF-vision trainer defines it to be 'accuracy',
    # while nas-client only uses 'top_1_accuracy' as the nasTargetRewardMetric.
    # So we use 'top_1_accuracy' as the only key to be consistent.
    if model == "classification":
      accuracy = float(metrics[accuracy_key])
      metrics.pop(accuracy_key)
      accuracy_key = "top_1_accuracy"
      metrics[accuracy_key] = accuracy
    elif model.startswith("pointpillars"):
      metrics = {
          utils.MODEL_MAP: metrics[utils.MODEL_MAP],
          utils.MODEL_MAPH: metrics[utils.MODEL_MAPH]
      }
      model_analysis[utils.MODEL_MAPH] = float(metrics[utils.MODEL_MAPH])

    # Since metrics[accuracy_key] will be updated below by combining it with
    # latency or flops, we store the original value into model_analysis to
    # report and make a new name to tell it apart form accuracy_key.
    # TODO: b/228496449
    model_analysis[accuracy_key + "_without_latency"] = float(
        metrics[accuracy_key])
    logging.info("Loaded train model: %s multi-adds billion FLOPS.",
                 model_stats["multi_add_flops_billion"])
    model_flops = model_stats["multi_add_flops_billion"]
    model_memory = 0

    # Get inference latency.
    if constraint_type == utils.LatencyConstraintType.DEVICE_MILLI_SEC:
      latency_millisec, model_memory = latency_utils.get_model_latency(
          params=params,
          model_dir=model_dir,
          latest_checkpoint=tf.train.latest_checkpoint(model_dir))
      measured_latency_or_flops = latency_millisec
      model_analysis[utils.MODEL_LATENCY_MILLI] = measured_latency_or_flops
      model_analysis[utils.MODEL_MEMORY] = model_memory
      logging.info("Model latency is %s milli-seconds.",
                   measured_latency_or_flops)
    elif constraint_type == utils.LatencyConstraintType.FLOPS_HALF_ADDS_BILLION:
      measured_latency_or_flops = model_stats["multi_add_flops_billion"]

    # Update metrics-reward with latency/FLOPS before reporting to controller.
    if constraint_type != utils.LatencyConstraintType.NONE:

      proxy_task_utils.update_trial_training_latency_metric(
          model_dir=model_dir, latency=measured_latency_or_flops)

      original_accuracy = float(metrics[accuracy_key])
      metrics[accuracy_key] = cloud_nas_utils.compute_reward(
          accuracy=original_accuracy,
          target_latency_or_flops=target_latency_or_flops,
          measured_latency_or_flops=measured_latency_or_flops,
          target_memory=target_memory,
          measured_memory=model_memory)
      logging.info(
          ("Original accuracy [%s], accuracy_with_constraint [%s] with "
           "target_latency_or_flops [%s] and measured_latency_or_flops [%s]"),
          original_accuracy, metrics[accuracy_key], target_latency_or_flops,
          measured_latency_or_flops)

    # Report measurement to the NAS controller.
    # NOTE: If training stops earlier due to proxy-task stopping condition
    # then the reported step should be less than total training steps.
    measurement = utils.Measurement(
        step=int(controller.global_step.numpy()),
        metrics={k: float(v) for k, v in six.iteritems(metrics)})

    if retrain_search_job_trials:
      model_analysis[
          "nas_trial_id"] = cloud_nas_utils.get_search_trial_id_to_finetune(
              retrain_search_job_trials)

    # Other metrics can also be reported for analysis,
    session.report(
        measurement, model_flops=model_flops, other_metrics=model_analysis)

    # Save model-analysis to disk.
    logging.info("Writing model_analysis.json: %s", model_analysis)
    with tf.io.gfile.GFile(os.path.join(model_dir, "model_analysis.json"),
                           "w") as fp:
      json.dump(model_analysis, fp, indent=2, sort_keys=True)

  elif mode == "train_and_eval":
    # In NAS stage 2, we need to evaluate continuously so that Tensorboard
    # can show the curves of eval metrics. Besides, we do not need to report
    # rewards for `train_and_eval` mode since `reward` is only needed by NAS
    # controller in NAS stage 1 jobs.
    with distribution_strategy.scope():
      controller.train_and_evaluate(
          train_steps=params.trainer.train_steps,
          eval_steps=params.trainer.validation_steps,
          eval_interval=params.trainer.validation_interval)
  else:
    raise ValueError("Unsupported job_mode: {}".format(mode))
