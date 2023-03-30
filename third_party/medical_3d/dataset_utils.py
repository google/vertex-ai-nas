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
"""Utility functions for Datasets."""

import logging
import os
import tempfile

import matplotlib
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
from monai.data import CacheDataset
from monai.data import Dataset
import tensorflow.compat.v1 as tf

# NOTE: This is needed for non-interactive use of matplotlib.
matplotlib.use("Agg")

_GCS_DIR_TAG = "gcs_dir"
_IMAGE_PATTERN_TAG = "image_pattern"
_LABEL_PATTERN_TAG = "label_pattern"


def get_images_using_glob(image_pattern, label_pattern):
  """Returns images using glob."""
  images = sorted(tf.gfile.Glob(image_pattern))
  labels = sorted(tf.gfile.Glob(label_pattern))
  logging.info("Found %d images for pattern %s", len(images), image_pattern)
  logging.info("Found %d labels for pattern %s", len(labels), label_pattern)
  return images, labels


def get_images_using_indices(image_pattern, label_pattern, begin_idx, end_idx):
  """Returns images using indices."""
  images = []
  labels = []
  for idx in range(begin_idx, end_idx+1):
    image = image_pattern.format(idx)
    label = label_pattern.format(idx)
    if tf.gfile.Exists(image) and tf.gfile.Exists(label):
      images.append(image)
      labels.append(label)
  logging.info("Found %d images for pattern %s", len(images), image_pattern)
  logging.info("Found %d labels for pattern %s", len(labels), label_pattern)
  return images, labels


def parse_str_arg(str_arg):
  """Parses string to return a dictionary of key value pairs.

  Args:
    str_arg: A string of the form '<key>=<val>,<key>=<val>'.

  Returns:
    A dictionary of key-value pairs.
  """
  item_list = [item.strip() for item in str_arg.split(",")]
  parsed_dict = {}
  for item in item_list:
    key, val = item.split("=")
    parsed_dict[key.strip()] = val.strip()
  return parsed_dict


def get_dataset(images, labels, transform, fast_monai=True):
  """Returns dataset."""
  data_dicts = [
      {"image": image_name, "label": label_name}
      for image_name, label_name in zip(images, labels)
  ]
  if fast_monai:
    return CacheDataset(
        data=data_dicts,
        transform=transform,
        cache_rate=1.0,
        num_workers=8,
    )
  else:
    return Dataset(data=data_dicts, transform=transform)


def save_data_plots(save_folder, val_ds, slice_number, in_channels,
                    out_channels):
  """Saves sample plots for validation data."""
  # Pick one image to visualize and check the 4 channels.
  data_idx = 0
  logging.info("image shape: %s", val_ds[data_idx]["image"].shape)
  plt.figure("image", (24, 6))
  for i in range(in_channels):
    plt.subplot(1, in_channels, i + 1)
    plt.title(f"image channel {i}")
    plt.imshow(
        val_ds[data_idx]["image"][i, :, :, slice_number].detach().cpu(),
        cmap="gray")
  plot_filename = os.path.join(save_folder, "sample_image.png")
  with tf.gfile.Open(plot_filename, "wb") as plot_file:
    plt.savefig(plot_file)
  # Also visualize the 3 channels label corresponding to this image.
  logging.info("label shape: %s", val_ds[data_idx]["label"].shape)
  plt.figure("label", (18, 6))
  for i in range(out_channels):
    plt.subplot(1, out_channels, i + 1)
    plt.title(f"label channel {i}")
    plt.imshow(val_ds[data_idx]["label"][i, :, :, slice_number].detach().cpu())
  plot_filename = os.path.join(save_folder, "sample_label.png")
  with tf.gfile.Open(plot_filename, "wb") as plot_file:
    plt.savefig(plot_file)


def fast_dataset(str_arg, val_roi_size, train_transform, val_transform,
                 in_channels, out_channels):
  """Returns small MSD dataset.

  Args:
    str_arg: A string of the form 'image_pattern=<path>, label_pattern=<path>'.
    val_roi_size: 3d patch-size for the model-input. Ex: [96, 96, 64]
    train_transform: data-transforms for training.
    val_transform: data-transforms for training.
    in_channels: number of input channels for the model.
    out_channels: number of output channels for the model.
  """
  parsed_dict = parse_str_arg(str_arg)
  image_pattern = parsed_dict["image_pattern"]
  label_pattern = parsed_dict["label_pattern"]
  images, labels = get_images_using_glob(image_pattern, label_pattern)
  train_ds = get_dataset(
      images=images, labels=labels, transform=train_transform)
  val_ds = get_dataset(
      images=images, labels=labels, transform=val_transform)
  return train_ds, val_ds, in_channels, out_channels, val_roi_size


def cloud_dataset(str_arg, val_roi_size, train_transform, val_transform,
                  in_channels, out_channels, validation_idx, end_idx):
  """Returns MSD dataset for cloud training.

  NOTE: It copies the data files from GCS location to the local VM first.

  Args:
    str_arg: A string of the form 'gcs_dir=<path>, image_pattern=<path>,
      label_pattern=<path>'
    val_roi_size: 3d patch-size for the model-input. Ex: [96, 96, 64]
    train_transform: data-transforms for training.
    val_transform: data-transforms for training.
    in_channels: number of input channels for the model.
    out_channels: number of output channels for the model.
    validation_idx: begining index for the validation data.
    end_idx: end_idx for the entire data.
  """
  parsed_dict = parse_str_arg(str_arg)

  gcs_dir = parsed_dict[_GCS_DIR_TAG]
  image_pattern = parsed_dict[_IMAGE_PATTERN_TAG]
  label_pattern = parsed_dict[_LABEL_PATTERN_TAG]

  # Copy training data from GCS location to the local VM.
  local_dir = tempfile.gettempdir()
  command = "gsutil -m cp -R {} {}/".format(gcs_dir, local_dir)
  os.system(command)

  image_pattern = local_dir + "/" + image_pattern
  label_pattern = local_dir + "/" + label_pattern
  train_images, train_labels = get_images_using_indices(
      image_pattern=image_pattern,
      label_pattern=label_pattern,
      begin_idx=0,
      end_idx=validation_idx-1)
  val_images, val_labels = get_images_using_indices(
      image_pattern=image_pattern,
      label_pattern=label_pattern,
      begin_idx=validation_idx,
      end_idx=end_idx)
  train_ds = get_dataset(
      images=train_images, labels=train_labels, transform=train_transform)
  val_ds = get_dataset(
      images=val_images, labels=val_labels, transform=val_transform)
  return train_ds, val_ds, in_channels, out_channels, val_roi_size
