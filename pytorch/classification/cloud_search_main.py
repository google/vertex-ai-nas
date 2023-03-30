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
# ==============================================================================
"""Main function for PyTorch mnasnet classification."""

from __future__ import absolute_import

import argparse
import functools
import itertools
import json
import logging
import math
import os
import sys
import time
from typing import Any, Mapping, Tuple

import cloud_nas_utils
import metrics_reporter
from gcs_utils import gcs_path_utils
from pytorch.classification import base_config
from pytorch.classification import mnasnet
from pytorch.classification import params_dict
from pytorch.classification import search_space as classification_search_space
import pyglove as pg
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchmetrics
from torchvision.transforms import autoaugment
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
import webdataset as wds

# Configure logging globally instead of in main(), to have it take effect
# in main process and spawned processes.
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# pylint: disable=logging-fstring-interpolation

# The file name of task params.
PARAMS_FILE = "params.yaml"
# The file name of saved nas_params_str.
NAS_PARAMS_STR_FILE = "nas_params.json"
# The file name of saved tunable object of search space.
TUNABLE_OBJECT_FILE = "serialized_tunable_object.json"
# The file name of saved training metric.
METRICS_FILE = "metrics.json"
# The file name of saved checkpoints.
CKPT_FILE = "checkpoint.pth"

# Train data transform: the image size of crop and resize.
T_TRAIN_RESIZE = 224
# Eval data transform: the image size of resize.
T_EVAL_RESIZE = 256
# Eval data transform: the image size of crop.
T_EVAL_CROP = 224
# Data transform: normalization mean.
T_NORM_MEAN = [0.485, 0.456, 0.406]
# Data transform: normalization std.
T_NORM_STD = [0.229, 0.224, 0.225]


def create_args():
  """Creates arg parser."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # Nas-service required flags.
  parser.add_argument(
      "--nas_params_str",
      type=str,
      help="Nas args serialized in JSON string.")
  parser.add_argument(
      "--job-dir",
      type=str,
      default="tmp",
      help="Job output directory.")
  parser.add_argument(
      "--retrain_search_job_dir",
      type=str,
      help="The job dir of the NAS search job to retrain.")
  parser.add_argument(
      "--retrain_search_job_trials",
      type=str,
      help="A list of trial IDs of the NAS search job to retrain, "
      "separated by comma.")

  # Task specific flags.
  parser.add_argument(
      "--config_file",
      type=str,
      help="Configuration file path.")
  parser.add_argument(
      "--train_data_path",
      type=str,
      help="Path to training data.")
  parser.add_argument(
      "--eval_data_path",
      type=str,
      help="Path to evaluation data.")
  parser.add_argument(
      "--skip_eval",
      type=cloud_nas_utils.str_2_bool,
      default=False,
      help="True to skip evaluation.")
  parser.add_argument(
      "--search_space",
      type=str,
      default="mnasnet",
      choices=["mnasnet"],
      help="The choice of NAS search space, e.g., mnasnet.")
  args = parser.parse_args()
  return args


def create_params(flags,
                  serialized_tunable_object):
  """Create params from base config."""
  params = base_config.BASE_CONFIG
  if flags.config_file:
    params = params_dict.override_params_dict(
        params, flags.config_file, is_strict=True)
  if flags.train_data_path:
    train_dict = {"train": {"data_path": flags.train_data_path}}
    params = params_dict.override_params_dict(
        params, train_dict, is_strict=True)
  if flags.eval_data_path:
    eval_dict = {"eval": {"data_path": flags.train_data_path}}
    params = params_dict.override_params_dict(
        params, eval_dict, is_strict=True)

  if flags.search_space == "mnasnet":
    params.override({
        "tunable_mnasnet": {
            "block_specs": serialized_tunable_object
        },
    }, is_strict=True)
  else:
    raise ValueError("Unexpected search_space {}".format(flags.search_space))

  params.train.data_path = gcs_path_utils.gcs_fuse_path(params.train.data_path)
  params.eval.data_path = gcs_path_utils.gcs_fuse_path(params.eval.data_path)

  params.validate()
  params.lock()
  return params


def get_device_and_world_size():
  """Get device and world size of DDP."""
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device == "cuda":
    count = torch.cuda.device_count()
    return device, count
  return device, 1


def get_dataloader_num_workers(world_size,
                               params):
  """Get num workers for dataloader."""
  if params.runtime.dataloader_num_workers is None:
    return int(os.cpu_count() / world_size)
  return params.runtime.dataloader_num_workers


def wds_split(src, rank, world_size):
  """Shards split function for webdataset."""
  # The context of caller of this function is within multiple processes
  # (by DDP world_size) and multiple workers (by dataloader_num_workers).
  # So we totally have (world_size * num_workers) workers for processing data.
  # NOTE: Raw data should be sharded to enough shards to make sure one process
  # can handle at least one shard, otherwise the process may hang.
  worker_id = 0
  num_workers = 1
  worker_info = torch.utils.data.get_worker_info()
  if worker_info:
    worker_id = worker_info.id
    num_workers = worker_info.num_workers
  for s in itertools.islice(src, rank * num_workers + worker_id, None,
                            world_size * num_workers):
    yield s


def identity(x):
  return x


def create_wds_dataloader(rank,
                          world_size,
                          params,
                          mode = "train"):
  """Create webdataset dataset and dataloader."""
  if mode == "train":
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=T_TRAIN_RESIZE),
        transforms.RandomHorizontalFlip(),
        autoaugment.TrivialAugmentWide(
            interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=T_NORM_MEAN, std=T_NORM_STD),
        transforms.RandomErasing(params.train.random_erase)
    ])
    data_path = params.train.data_path
    data_size = params.train.data_size
    batch_size_global = params.train.batch_size
    batch_size_local = int(batch_size_global / world_size)
    # Since webdataset disallows partial batch, we pad the last batch for train.
    batches = int(math.ceil(data_size / batch_size_global))
    shard_shuffle_size = params.train.shard_shuffle_size
    sample_shuffle_size = params.train.sample_shuffle_size
  else:
    transform = transforms.Compose([
        transforms.Resize(T_EVAL_RESIZE),
        transforms.CenterCrop(T_EVAL_CROP),
        transforms.ToTensor(),
        transforms.Normalize(mean=T_NORM_MEAN, std=T_NORM_STD),
    ])
    data_path = params.eval.data_path
    data_size = params.eval.data_size
    batch_size_global = params.eval.batch_size
    batch_size_local = int(batch_size_global / world_size)
    # Since webdataset disallows partial batch, we drop the last batch for eval.
    batches = int(data_size / batch_size_global)
    shard_shuffle_size = 0
    sample_shuffle_size = 0

  dataset = wds.DataPipeline(
      wds.SimpleShardList(data_path),
      wds.shuffle(shard_shuffle_size),
      functools.partial(wds_split, rank=rank, world_size=world_size),
      wds.tarfile_to_samples(),
      wds.shuffle(sample_shuffle_size),
      wds.decode("pil"),
      wds.to_tuple("jpg;png;jpeg cls"),
      wds.map_tuple(transform, identity),
      wds.batched(batch_size_local, partial=False),
  )
  num_workers = get_dataloader_num_workers(world_size, params)
  dataloader = wds.WebLoader(
      dataset=dataset,
      batch_size=None,
      shuffle=False,
      num_workers=num_workers,
      persistent_workers=True if num_workers > 0 else False,
      pin_memory=True).repeat(nbatches=batches)
  logging.info(f"{mode} dataloader | samples: {data_size}, "
               f"num_workers: {num_workers}, "
               f"local batch size: {batch_size_local}, "
               f"global batch size: {batch_size_global}, "
               f"batches: {batches}")
  return dataloader


def train(model, device,
          dataloader, optimizer):
  """Run model training loop."""
  start = time.time()
  model.train()
  step = 0
  for image, target in dataloader:
    image = image.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    pred = model(image)
    loss = nn.functional.cross_entropy(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step += 1
  end = time.time()
  duration = end - start
  return step, duration, loss.item()


def evaluate(model, device,
             dataloader, metric):
  """Run model evaluation loop."""
  start = time.time()
  model.eval()
  step = 0
  with torch.no_grad():
    for image, target in dataloader:
      image = image.to(device, non_blocking=True)
      target = target.to(device, non_blocking=True)
      pred = model(image)
      loss = nn.functional.cross_entropy(pred, target)
      metric.update(pred, target)
      step += 1
  # In DDP mode, the internal state of the metric is synced and reduced across
  # each process.
  accuracy = metric.compute()
  metric.reset()
  end = time.time()
  duration = end - start
  return step, duration, loss.item(), accuracy.item()


def save_checkpoint(state,
                    save_prefix):
  latest_file = save_prefix + ".latest.pth"
  torch.save(state, latest_file)


def load_checkpoint(load_path, device):
  state = torch.load(load_path, map_location=device)
  return state


def worker(rank, world_size,
           device, flags, params):
  """Runs model training in single process view."""
  # Initiate process group.
  dist.init_process_group(
      backend="nccl" if device == "cuda" else "gloo",
      init_method="env://",
      world_size=world_size,
      rank=rank)
  if not dist.is_initialized():
    raise ValueError("Failed to initialize process group.")
  else:
    logging.info(
        f"Initialized process {dist.get_rank()} / {dist.get_world_size()}")

  # Create model.
  model = mnasnet.build_mnasnet_model(params)
  if device == "cuda":
    torch.cuda.set_device(rank)
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
  logging.info(f"Process {rank} converts model to {device}")

  # Create dataloader.
  train_dataloader = create_wds_dataloader(rank, world_size, params, "train")
  eval_dataloader = create_wds_dataloader(rank, world_size, params, "eval")

  # Create or load optimizer.
  optimizer = torch.optim.SGD(params=model.parameters(),
                              lr=params.train.optimizer.learning_rate,
                              momentum=params.train.optimizer.momentum,
                              weight_decay=params.train.optimizer.weight_decay)
  # Create learning rate scheduler.
  main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer,
      T_max=params.train.num_epochs - params.train.lr_scheduler.warmup_epochs,
      eta_min=params.train.lr_scheduler.min_lr)
  warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
      optimizer,
      start_factor=params.train.lr_scheduler.warmup_decay,
      total_iters=params.train.lr_scheduler.warmup_epochs)
  lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
      optimizer,
      schedulers=[warmup_lr_scheduler, main_lr_scheduler],
      milestones=[params.train.lr_scheduler.warmup_epochs])

  # Try to load checkpoint.
  epochs = 0
  steps = 0
  accuracy = 0
  if params.checkpoint.load_path:
    state = load_checkpoint(params.checkpoint.load_path, device)
    epochs = state["epochs"]
    steps = state["steps"]
    accuracy = state["accuracy"]
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])

  # Run main loop.
  metric = torchmetrics.classification.Accuracy(top_k=1).to(device)
  while epochs < params.train.num_epochs:
    if rank == 0:
      logging.info(f"Running epoch {epochs}")
    # Train loop.
    train_steps, duration, loss = train(
        model, device, train_dataloader, optimizer)
    if rank == 0:
      logging.info(f"Train epoch: {epochs} | steps: {train_steps} | "
                   f"duration: {duration:>0.2f} | "
                   f"seconds/step: {(duration / train_steps):>0.2f} | "
                   f"lr: {lr_scheduler.get_last_lr()} | loss: {loss:>0.2f}")
    steps += train_steps
    lr_scheduler.step()

    # Eval loop.
    if not flags.skip_eval:
      eval_steps, duration, loss, accuracy = evaluate(
          model, device, eval_dataloader, metric)
      if rank == 0:
        logging.info(f"Eval epoch: {epochs} | steps: {eval_steps} | "
                     f"duration: {duration:>0.2f} | "
                     f"loss: {loss:>0.2f} | accuracy: {accuracy}")
    # Save checkpoint.
    if rank == 0:
      state = {
          "epochs": epochs,
          "steps": steps,
          "accuracy": accuracy,
          "model": model.state_dict(),
          "optimizer": optimizer.state_dict(),
          "lr_scheduler": lr_scheduler.state_dict(),
      }
      save_checkpoint(
          state, os.path.join(flags.job_dir,
                              params.checkpoint.save_file_prefix))
    epochs += 1

  # Dump metrics to file.
  if rank == 0:
    final_metrics = {
        "steps": steps,
        "loss": loss,
        "accuracy": accuracy,
    }
    logging.info(final_metrics)
    with open(os.path.join(flags.job_dir, METRICS_FILE), "w") as f:
      json.dump(final_metrics, f, indent=2)


def main():
  flags = create_args()

  # Find trial id for this job.
  trial_id = cloud_nas_utils.get_trial_id_from_environment()
  logging.info(f"Starting trial {trial_id}.")

  # Create job dir.
  flags.job_dir = cloud_nas_utils.get_job_dir_from_environment_if_exist(
      current_job_dir=flags.job_dir)
  logging.info(f"Job dir: {flags.job_dir}")
  flags.job_dir = gcs_path_utils.gcs_fuse_path(flags.job_dir)
  if not os.path.isdir(flags.job_dir):
    os.makedirs(flags.job_dir)

  # Get nas_params_str.
  if flags.retrain_search_job_trials:
    # Reset `nas_params_str` if this job is to retrain a previous NAS trial.
    flags.nas_params_str = cloud_nas_utils.get_finetune_nas_params_str(
        retrain_search_job_trials=flags.retrain_search_job_trials,
        retrain_search_job_dir=flags.retrain_search_job_dir)
  # Save nas_params_str, so that it can be reused by stage-2 jobs.
  with open(os.path.join(flags.job_dir, NAS_PARAMS_STR_FILE), "w") as f:
    f.write(flags.nas_params_str)

  # Process nas_params_str to give one instance of the search-space to be used
  # for this trial.
  logging.info(f"search_space: {flags.search_space}")
  logging.info(f"nas_params_str: {flags.nas_params_str}")
  tunable_object_or_functor = cloud_nas_utils.parse_and_save_nas_params_str(
      classification_search_space.mnasnet_search_space(
          reference="mobilenet_v2"), flags.nas_params_str, flags.job_dir)
  tunable_object = tunable_object_or_functor()
  # Create a serialized `tunable_object` which will be used to build the model.
  serialized_tunable_object = pg.to_json_str(
      tunable_object, json_indent=2, hide_default_values=False)
  logging.info(f"serialized_tunable_object: {serialized_tunable_object}")
  with open(os.path.join(flags.job_dir, TUNABLE_OBJECT_FILE), "w") as f:
    f.write(serialized_tunable_object)

  # Create params for the training task.
  params = create_params(flags, serialized_tunable_object)
  logging.info(f"Trainer params: {params.as_dict()}")
  params_dict.save_params_dict_to_yaml(
      params, os.path.join(flags.job_dir, PARAMS_FILE))

  # Set env for DDP usage.
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "8888"
  # Uncomment the following two lines to log details of DDP setup.
  # os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
  # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

  # Spawn multiple processes. If one of the processes exits with a non-zero exit
  # status, the remaining processes are killed and an exception is raised.
  # https://pytorch.org/docs/stable/notes/ddp.html
  device, world_size = get_device_and_world_size()
  logging.info(f"Ready to spawn {world_size} processes for {device}")
  mp.spawn(fn=worker,
           args=(world_size, device, flags, params),
           nprocs=world_size)

  # Read metrics from file.
  with open(os.path.join(flags.job_dir, METRICS_FILE), "r") as f:
    metrics = json.load(f)
  accuracy = metrics["accuracy"]
  training_steps = metrics["steps"]
  # TODO Add latency constraint support.

  # Report metrics back to NAS-service after training.
  metric_tag = os.environ.get("CLOUD_ML_HP_METRIC_TAG", "")
  other_metrics = {}
  if flags.retrain_search_job_trials:
    other_metrics[
        "nas_trial_id"] = cloud_nas_utils.get_search_trial_id_to_finetune(
            flags.retrain_search_job_trials)
  if metric_tag:
    nas_metrics_reporter = metrics_reporter.NasMetricsReporter()
    nas_metrics_reporter.report_metrics(
        hyperparameter_metric_tag=metric_tag,
        metric_value=accuracy,
        global_step=training_steps,
        other_metrics=other_metrics)


if __name__ == "__main__":
  main()

# pylint: enable=logging-fstring-interpolation
