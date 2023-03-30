# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Model definition for the tunable UNet-NASFPN."""

from gcs_utils import gcs_path_utils
from third_party.medical_3d import nasfpn
from third_party.medical_3d import unet_encoder
import pyglove as pg
import torch


def _encoder_lev_idx_to_nasfpn_level(lev_idx):
  return lev_idx + 1


class UNet3dNASFPN(torch.nn.Module):
  """Unet-3D encoder with NASFPN-3D decoder."""

  def __init__(self, in_channels, out_channels, features, nasfpn_min_level,
               nasfpn_max_level, nasfpn_num_filters, nasfpn_num_repeats,
               nasfpn_block_specs, nasfpn_norm):
    super().__init__()
    self.unet_encoder = unet_encoder.UNetEncoder(
        dimensions=3,
        in_channels=in_channels,
        features=features,
    )
    encoder_channels = self.unet_encoder.get_channels()

    # Add offset to encoder levels for nasfpn.
    backbone_level_to_channels = {}
    for lev_idx, channel in enumerate(encoder_channels):
      nasfpn_level = _encoder_lev_idx_to_nasfpn_level(lev_idx)
      backbone_level_to_channels[nasfpn_level] = channel

    self.nasfpn = nasfpn.NASFPN(
        backbone_level_to_channels=backbone_level_to_channels,
        min_level=nasfpn_min_level,
        max_level=nasfpn_max_level,
        num_filters=nasfpn_num_filters,
        num_repeats=nasfpn_num_repeats,
        block_specs=nasfpn_block_specs,
        norm=nasfpn_norm)

    # Fusion applied to NASFPN output for segmentation.
    self.fusion_op = nasfpn.FuseOutputFeatures(
        nasfpn_min_level=nasfpn_min_level,
        nasfpn_max_level=nasfpn_max_level,
        nasfpn_num_filters=nasfpn_num_filters,
        nasfpn_norm=nasfpn_norm,
        backbone_level_to_channels=backbone_level_to_channels)

    # Final-Op applied to output feature.
    self.final_op = torch.nn.Conv3d(
        nasfpn_num_filters,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False)

  def forward(self, x):
    encoder_features = self.unet_encoder(x)

    # Add offset to encoder levels for nasfpn.
    nasfpn_input_features = {}
    for lev_idx, feature in enumerate(encoder_features):
      nasfpn_level = _encoder_lev_idx_to_nasfpn_level(lev_idx)
      nasfpn_input_features[nasfpn_level] = feature

    nasfpn_features = self.nasfpn(nasfpn_input_features)
    fused_output = self.fusion_op(
        nasfpn_features=nasfpn_features,
        backbone_features=nasfpn_input_features)
    output = self.final_op(fused_output)
    return output


def tunable_unet_nasfpn(params):
  """Returns tunable-unet-nasfpn model."""
  block_specs_json = gcs_path_utils.gcs_fuse_path(params["block_specs_json"])
  if block_specs_json.endswith(".json"):
    with open(block_specs_json, "r") as f:
      block_specs_json = f.read()
  block_specs = pg.from_json_str(block_specs_json)

  # NOTE: Unet encoder by default uses instance normalization. So will use the
  # same for NASFPN too.
  return UNet3dNASFPN(
      in_channels=params["in_channels"],
      out_channels=params["out_channels"],
      features=params["features"],
      nasfpn_min_level=params["min_level"],
      nasfpn_max_level=params["max_level"],
      nasfpn_num_filters=params["num_filters"],
      nasfpn_num_repeats=params["num_repeats"],
      nasfpn_block_specs=block_specs,
      nasfpn_norm="instance")
