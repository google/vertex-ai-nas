# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unet Encoder derived from MONAI BasicUnet."""

from third_party.medical_3d import basic_unet
import torch.nn as nn


class UNetEncoder(nn.Module):
  """Unet-Encoder implementation."""

  def __init__(
      self,
      dimensions,
      in_channels,
      features,
      act=("LeakyReLU", {
          "negative_slope": 0.1,
          "inplace": True
      }),
      norm=("instance", {
          "affine": True
      }),
      dropout=0.0
  ):
    """A UNet-Encoder implementation with 1D/2D/3D supports.

    Args:
        dimensions: number of spatial dimensions. Defaults to 3 for spatial
          3D inputs.
        in_channels: number of input channels. Defaults to 1.
        features: six integers as numbers of features. Defaults to ``(32,
          32, 64, 128, 256)``, - the five values correspond to the
          five-level encoder feature sizes.
        act: activation type and arguments. Defaults to LeakyReLU.
        norm: feature normalization type and arguments. Defaults to instance
          norm.
        dropout: dropout ratio. Defaults to no dropout.
    """
    super().__init__()

    self.features = list(features)
    self.num_levels = len(features)
    if self.num_levels < 2:
      raise ValueError("The Unet should atleast have a depth of two.")
    fea = list(features)
    print(f"BasicUNet features: {fea}.")

    self.down_ops = basic_unet.build_down_ops(
        dimensions=dimensions,
        in_channels=in_channels,
        features=fea,
        act=act,
        norm=norm,
        dropout=dropout,
        num_levels=self.num_levels)

  def get_channels(self):
    return self.features

  def forward(self, x):
    """Returns a list of features at every level starting with topmost level."""
    # Down path.
    down_features = []
    for level in range(self.num_levels):
      x = self.down_ops[level](x)
      down_features.append(x)
    return down_features
