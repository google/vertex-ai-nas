# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Monai Trainer."""

import logging
import os
import tempfile

from monai.data import DataLoader
from monai.data.utils import list_data_collate
from monai.data.utils import pad_list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.optimizers import Novograd
from monai.transforms import (  # pylint: disable=g-multiple-import
    Activations,
    AsDiscrete,
    Compose,
)
import tensorflow.compat.v1 as tf
import torch
from torch.optim.swa_utils import AveragedModel
from torch.optim.swa_utils import update_bn
from torch.utils.tensorboard import SummaryWriter

_TENSORBOARD_DIR = "tensorboard"


def get_metric(val_outputs,
               val_labels,
               metric_fn,
               metric_count,
               metric_sum,
               class_idx=-1):
  """Updates metric statistics for a class."""
  if class_idx == -1:
    value, not_nans = metric_fn(y_pred=val_outputs, y=val_labels)
  else:
    value, not_nans = metric_fn(
        y_pred=val_outputs[:, class_idx:class_idx + 1],
        y=val_labels[:, class_idx:class_idx + 1])

  not_nans = not_nans.item()
  metric_count += not_nans
  metric_sum += value.item() * not_nans
  return metric_count, metric_sum


def update_plot(metric_tag, metric_values, tensorboard_writer):
  """Updates plot for an input metric."""
  epoch = metric_values[-1]["epoch"]
  metric_val = metric_values[-1]["val"]
  logging.info("current epoch: %d current %s: %.4f", epoch, metric_tag,
               metric_val)
  tensorboard_writer.add_scalar(metric_tag, metric_val, epoch)
  tensorboard_writer.flush()


def get_val_outputs(val_inputs, sliding_window_val_roi_size, model):
  """Returns validation outputs given validation inputs."""
  if sliding_window_val_roi_size:
    val_outputs = sliding_window_inference(
        inputs=val_inputs,
        roi_size=sliding_window_val_roi_size,
        sw_batch_size=1,
        predictor=model)
  else:
    val_outputs = model(val_inputs)
  return val_outputs


def train(train_ds,
          val_ds,
          sliding_window_val_roi_size,
          output_dir,
          model,
          num_gpus,
          epoch_num,
          val_interval,
          batch_size,
          moving_average_decay,
          learning_rate,
          learning_rate_scheduler,
          fast_monai=True):
  """Runs training."""
  # NOTE: Determinism should set in the main file.

  if fast_monai:
    # Uses cached data so does not need more workers here.
    data_loader_workers = 1
  else:
    data_loader_workers = 4
  train_loader = DataLoader(
      train_ds,
      batch_size=batch_size,
      shuffle=True,
      num_workers=data_loader_workers)
  # NOTE: For validation, allow images to be of different size by
  # using 'collate_fn'. The sliding-window inference will use a common
  # smaller patch-size even if images are of different sizes.
  if sliding_window_val_roi_size:
    collate_fn = pad_list_data_collate
  else:
    collate_fn = list_data_collate
  val_loader = DataLoader(
      val_ds,
      batch_size=batch_size,
      shuffle=False,
      num_workers=data_loader_workers,
      collate_fn=collate_fn)

  # Set up device.
  if num_gpus > 0:
    device = torch.device("cuda")
    logging.info("Using %d GPUs for training.", torch.cuda.device_count())
  else:
    device = torch.device("cpu")
    logging.info("Using CPU for training.")

  # Create model, loss, and optimizer.
  if num_gpus > 1:
    model = torch.nn.DataParallel(model)
  model = model.to(device)

  use_swa = (moving_average_decay > 0.0)
  if use_swa:
    # Add swa_model for stochastic-weighted-averaging.
    logging.info("moving_average_decay is %f", moving_average_decay)
    def _avg_fun(averaged_model_parameter, model_parameter, num_averaged):
      del num_averaged
      return moving_average_decay * averaged_model_parameter + (
          1 - moving_average_decay) * model_parameter
    swa_model = AveragedModel(model, avg_fn=_avg_fun)

  loss_function = DiceLoss(to_onehot_y=False, sigmoid=True, squared_pred=True)

  if fast_monai:
    # Novograd paper suggests to use a bigger LR than Adam,
    # because Adam does normalization by element-wise second moments.
    optimizer = Novograd(model.parameters(), learning_rate * 10)
    scaler = torch.cuda.amp.GradScaler()
  else:
    optimizer = torch.optim.Adam(
        model.parameters(), learning_rate, weight_decay=1e-5, amsgrad=True)

  if learning_rate_scheduler == "linear_decay":
    lr_lambda = lambda epoch: (1.0 - float(epoch) / epoch_num)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_lambda)
  else:
    scheduler = None

  # Initialize metrics.
  best_metric = -1
  best_metric_epoch = -1
  epoch_loss_values = list()
  metric_values = list()
  lr_values = list()
  num_classes = val_ds[0]["label"].shape[0]
  logging.info("Num-classes is %d", num_classes)
  per_class_metric_values = []
  for _ in range(num_classes):
    per_class_metric_values.append(list())

  # Initialize tensorboard writer.
  tensorboard_dir = os.path.join(output_dir, _TENSORBOARD_DIR)
  tf.gfile.MakeDirs(tensorboard_dir)
  tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)

  # Run training loop.
  for epoch in range(epoch_num):
    logging.info("-" * 10)
    logging.info("epoch %d/%d", epoch + 1, epoch_num)
    model.train()
    epoch_loss = 0
    step = 0
    logging.info("train_loader.batch_size is %d", train_loader.batch_size)
    for batch_data in train_loader:
      step += 1
      inputs, labels = (
          batch_data["image"].to(device),
          batch_data["label"].to(device),
      )
      optimizer.zero_grad()
      if fast_monai:
        # Set AMP for MONAI training.
        with torch.cuda.amp.autocast():
          outputs = model(inputs)
          loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
      else:
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
      if use_swa:
        swa_model.update_parameters(model)

      epoch_loss += loss.item()
      logging.info("%d/%d, train_loss: %.4f", step,
                   len(train_ds) // train_loader.batch_size, loss.item())
    epoch_loss /= step
    epoch_loss_values.append({"val": epoch_loss, "epoch": epoch + 1})

    # Plot loss.
    update_plot(
        metric_tag="Loss/Train",
        metric_values=epoch_loss_values,
        tensorboard_writer=tensorboard_writer)

    if scheduler:
      scheduler.step()

    # Plot learning rate.
    lr_values.append({
        "val": optimizer.param_groups[0]["lr"],
        "epoch": epoch + 1
    })
    update_plot(
        metric_tag="LearningRate",
        metric_values=lr_values,
        tensorboard_writer=tensorboard_writer)

    if use_swa:
      update_bn(train_loader, swa_model)

    # Run Validation.
    if (epoch + 1) % val_interval == 0 or (epoch+1) == epoch_num:
      logging.info("-" * 10)
      logging.info("Running validation.")
      model.eval()
      if use_swa:
        model_for_eval = swa_model
      else:
        model_for_eval = model
      with torch.no_grad():
        dice_metric = DiceMetric(include_background=True, reduction="mean")
        post_trans = Compose(
            [Activations(sigmoid=True),
             AsDiscrete(threshold_values=True)])
        metric_sum = 0.0
        metric_count = 0
        per_class_metric_sum = [0.0] * num_classes
        per_class_metric_count = [0] * num_classes
        for val_data in val_loader:
          val_inputs, val_labels = (
              val_data["image"].to(device),
              val_data["label"].to(device),
          )
          if fast_monai:
            # Set AMP for MONAI validation.
            with torch.cuda.amp.autocast():
              val_outputs = get_val_outputs(
                  val_inputs=val_inputs,
                  sliding_window_val_roi_size=sliding_window_val_roi_size,
                  model=model_for_eval)
          else:
            val_outputs = get_val_outputs(
                val_inputs=val_inputs,
                sliding_window_val_roi_size=sliding_window_val_roi_size,
                model=model_for_eval)
          val_outputs = post_trans(val_outputs)

          # Compute overall mean dice for this batch.
          metric_count, metric_sum = get_metric(
              val_outputs=val_outputs,
              val_labels=val_labels,
              metric_fn=dice_metric,
              metric_count=metric_count,
              metric_sum=metric_sum,
              class_idx=-1)

          # Compute per-class dice for this batch.
          for class_idx in range(num_classes):
            per_class_metric_count[class_idx], per_class_metric_sum[
                class_idx] = get_metric(
                    val_outputs=val_outputs,
                    val_labels=val_labels,
                    metric_fn=dice_metric,
                    metric_count=per_class_metric_count[class_idx],
                    metric_sum=per_class_metric_sum[class_idx],
                    class_idx=class_idx)

        # Compute and plot mean and per-class dice at the end of the epoch.
        metric = metric_sum / metric_count
        metric_values.append({"val": metric, "epoch": epoch + 1})
        update_plot(
            metric_tag="MeanDice/Val",
            metric_values=metric_values,
            tensorboard_writer=tensorboard_writer)
        for class_idx in range(num_classes):
          per_class_metric = per_class_metric_sum[
              class_idx] / per_class_metric_count[
                  class_idx] if per_class_metric_count[class_idx] else 0.0
          per_class_metric_values[class_idx].append({
              "val": per_class_metric,
              "epoch": epoch + 1
          })
          update_plot(
              metric_tag="MeanDiceClass{}/Val".format(class_idx),
              metric_values=per_class_metric_values[class_idx],
              tensorboard_writer=tensorboard_writer)

        # Update best metric model.
        if metric > best_metric:
          best_metric = metric
          best_metric_epoch = epoch + 1
          # NOTE: torch.save does not work with tf.gfile file handle.
          local_model_filepath = os.path.join(tempfile.gettempdir(),
                                              "best_metric_model.pth")
          gcs_model_filepath = os.path.join(output_dir, "best_metric_model.pth")
          checkpoint = {
              "epoch": epoch + 1,
              "state_dict": model.state_dict(),
              "optimizer": optimizer.state_dict()
          }
          if scheduler:
            checkpoint["scheduler"] = scheduler.state_dict()
          torch.save(checkpoint, local_model_filepath)
          tf.gfile.Copy(
              local_model_filepath, gcs_model_filepath, overwrite=True)
          logging.info("best mean dice: %.4f at epoch: %d", best_metric,
                       best_metric_epoch)
          logging.info("Saved new best metric model.")

  tensorboard_writer.close()
  return metric_values
