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
"""MSD Pancreas dataset configuration."""

from monai.transforms import (  # pylint: disable=g-multiple-import
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandZoomd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from third_party.medical_3d import dataset_utils
import numpy as np

_IN_CHANNELS = 1
_OUT_CHANNELS = 2

# NOTE: Mean image shape at the spacing below is 512, 512, 96.
_PIX_DIM = (0.8, 0.8, 2.5)
_ROI_SIZE = [96, 96, 64]
_FAST_ROI_SIZE = [32, 32, 32]

# This index marks the begining of validation images.
_VALIDATION_INDEX = 296
_END_INDEX = 450


class ConvertToMultiChannelClassesd(MapTransform):
  """Converts labels to multi-channel."""

  def __call__(self, data):
    d = dict(data)
    for key in self.keys:
      result = list()
      for class_id in range(1, _OUT_CHANNELS+1):
        result.append(d[key] == class_id)
      d[key] = np.stack(result, axis=0).astype(np.float32)
    return d


def get_transforms(roi_size, pixdim):
  """Returns training and validation transforms."""
  # The min_hu and max_hu clips the CT image intensity (HU values) to the
  # region-of-interest (pancreas organ and tumor region).
  min_hu = -96
  max_hu = 215

  # When cropping, a large margin is used around the
  # region-of-interest. The validation image is cropped to reduce the
  # memory and compute time.
  # NOTE: this value is currently chosen empirically. Ideally, one should
  # use the entire training and validation image.
  margin = 50

  # zoom_prob decides the probability of the zoom augmentation.
  # NOTE: this value is currently chosen empirically. Ideally, augmentation
  # search can be used to search for the best parameter.
  zoom_prob = 0.8

  # Operations:
  # - Load images.
  # - Make image and labels channel-first.
  # - Resample images to the desired pixel-spacing: 'pixdim'.
  # - Align images to a common orientation.
  # - Clip intensity to min-hu and max-hu and rescale to 0 and 1.
  # - Crop region around the region-of-interest but keep a large margin around.
  # - If the image is smaller than 'roi_size' then pad it else do nothing.
  # - Take random crops of size 'roi_size'.
  # - Randomly flip along axis.
  # - Randomly scale and shift intensities.
  # - Convert to tensor.
  train_transform = Compose([
      LoadImaged(keys=["image", "label"]),
      AddChanneld(keys="image"),
      ConvertToMultiChannelClassesd(keys="label"),
      Spacingd(
          keys=["image", "label"],
          pixdim=pixdim,
          mode=("bilinear", "nearest")),
      Orientationd(keys=["image", "label"], axcodes="RAS"),
      ScaleIntensityRanged(
          keys=["image"],
          a_min=min_hu,
          a_max=max_hu,
          b_min=0.0,
          b_max=1.0,
          clip=True,
      ),
      CropForegroundd(
          keys=["image", "label"], source_key="label", margin=margin),
      # NOTE: Pad atleast to the roi_size. Will do nothing if image is
      # already larger.
      SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
      RandSpatialCropd(
          keys=["image", "label"], roi_size=roi_size, random_size=False),
      RandZoomd(
          keys=["image", "label"],
          prob=zoom_prob,
          min_zoom=1.0,
          max_zoom=1.5,
          mode=("trilinear", "nearest")),
      RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
      RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
      RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
      ToTensord(keys=["image", "label"]),
  ])

  # NOTE: We will ask the trainer to use sliding-window inference
  # to process larger validation image size patch-by-patch.
  # The trainer allows loading validation images of different sizes.
  #
  # Operations:
  # - Load images.
  # - Make image and labels channel-first.
  # - Resample images to the desired pixel-spacing: 'pixdim'.
  # - Align images to a common orientation.
  # - Clip intensity to min-hu and max-hu and rescale to 0 and 1.
  # - Crop region around the region-of-interest but keep a large margin around.
  # - Convert to tensor.
  val_transform = Compose([
      LoadImaged(keys=["image", "label"]),
      AddChanneld(keys="image"),
      ConvertToMultiChannelClassesd(keys="label"),
      Spacingd(
          keys=["image", "label"],
          pixdim=pixdim,
          mode=("bilinear", "nearest")),
      Orientationd(keys=["image", "label"], axcodes="RAS"),
      ScaleIntensityRanged(
          keys=["image"],
          a_min=min_hu,
          a_max=max_hu,
          b_min=0.0,
          b_max=1.0,
          clip=True,
      ),
      CropForegroundd(
          keys=["image", "label"], source_key="label", margin=margin),
      ToTensord(keys=["image", "label"]),
  ])
  return train_transform, val_transform


def fast_dataset(str_arg):
  """Returns small dataset for local testing.

  Args:
    str_arg: A string of the form 'image_pattern=<path>, label_pattern=<path>'.
  """
  train_transform, val_transform = get_transforms(
      roi_size=_FAST_ROI_SIZE, pixdim=_PIX_DIM)
  return dataset_utils.fast_dataset(
      str_arg=str_arg,
      val_roi_size=_FAST_ROI_SIZE,
      train_transform=train_transform,
      val_transform=val_transform,
      in_channels=_IN_CHANNELS,
      out_channels=_OUT_CHANNELS)


def cloud_dataset(str_arg):
  """Returns dataset for cloud training.

  Args:
    str_arg: A string of the form 'gcs_dir=<path>, image_pattern=<path>,
      label_pattern=<path>'
  """
  train_transform, val_transform = get_transforms(
      roi_size=_ROI_SIZE, pixdim=_PIX_DIM)
  return dataset_utils.cloud_dataset(
      str_arg=str_arg,
      val_roi_size=_ROI_SIZE,
      train_transform=train_transform,
      val_transform=val_transform,
      in_channels=_IN_CHANNELS,
      out_channels=_OUT_CHANNELS,
      validation_idx=_VALIDATION_INDEX,
      end_idx=_END_INDEX)
