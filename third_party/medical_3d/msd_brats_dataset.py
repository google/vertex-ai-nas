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
"""MSD BRATS dataset configuration."""

from monai.transforms import (  # pylint: disable=g-multiple-import
    AsChannelFirstd,
    CropForegroundd,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ResizeWithPadOrCropd,
    Spacingd,
    ToTensord,
)
from third_party.medical_3d import dataset_utils
import numpy as np

_IN_CHANNELS = 4
_OUT_CHANNELS = 3
_PIX_DIM = (1.0, 1.0, 1.0)
_ROI_SIZE = [96, 96, 64]
_FAST_ROI_SIZE = [32, 32, 32]

# This index marks the begining of validation images.
_VALIDATION_INDEX = 324
_END_INDEX = 500


# NOTE: This creates classes like the MSD challenge.
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
  """Converts to multi-channel."""

  def __call__(self, data):
    d = dict(data)
    for key in self.keys:
      result = list()
      # Label 1
      result.append(d[key] == 1)
      # Label 2
      result.append(d[key] == 2)
      # Label 3
      result.append(d[key] == 3)
      d[key] = np.stack(result, axis=0).astype(np.float32)
    return d


def get_transforms(roi_size, pixdim):
  """Returns training and validation transforms."""
  train_transform = Compose([
      # load 4 Nifti images and stack them together
      LoadImaged(keys=["image", "label"]),
      AsChannelFirstd(keys="image"),
      ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
      Spacingd(
          keys=["image", "label"],
          pixdim=pixdim,
          mode=("bilinear", "nearest")),
      Orientationd(keys=["image", "label"], axcodes="RAS"),
      CropForegroundd(keys=["image", "label"], source_key="image"),
      RandSpatialCropd(
          keys=["image", "label"], roi_size=roi_size, random_size=False),
      RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
      NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
      RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
      RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
      ToTensord(keys=["image", "label"]),
  ])

  # NOTE: We will ask the trainer to use sliding-window inference
  # to process larger validation image size patch-by-patch.
  val_size = [150, 150, 150]
  val_transform = Compose([
      LoadImaged(keys=["image", "label"]),
      AsChannelFirstd(keys="image"),
      ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
      Spacingd(
          keys=["image", "label"],
          pixdim=pixdim,
          mode=("bilinear", "nearest")),
      Orientationd(keys=["image", "label"], axcodes="RAS"),
      CropForegroundd(keys=["image", "label"], source_key="image"),
      ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=val_size),
      NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
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
