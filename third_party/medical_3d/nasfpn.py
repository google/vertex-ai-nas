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
"""3D NAS-FPN implementation in PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# The fixed NAS-FPN architecture discovered by NAS.
# Each element represents a specification of a building block:
#   (block_level, combine_fn, (input_offset0, input_offset1), is_output).
NASFPN_BLOCK_SPECS = [
    (4, "attention", (1, 3), False),
    (4, "sum", (1, 5), False),
    (3, "sum", (0, 6), True),
    (4, "sum", (6, 7), True),
    (5, "attention", (7, 8), True),
    (7, "attention", (6, 9), True),
    (6, "attention", (9, 10), True),
]


def _get_activation_op(act):
  """Returns op for activation.

  Args:
    act: string 'leaky_relu' or '' for no activation.
  """
  if not act:
    return nn.Identity()

  if act.lower() == "leaky_relu":
    return nn.LeakyReLU(negative_slope=0.1, inplace=True)
  else:
    raise ValueError("%s activation is not supported." % act)


def _get_norm_op(norm, num_filters):
  """Returns op for normalization.

  Args:
    norm: string 'instance' or 'batch'.
    num_filters: number of channels.
  """
  if norm.lower() == "instance":
    return nn.InstanceNorm3d(num_filters, affine=True)
  elif norm.lower() == "batch":
    return nn.BatchNorm3d(num_filters)
  else:
    raise ValueError("%s norm is not supported." % norm)


class BlockSpec(object):
  """A container class that specifies the block configuration for NAS-FPN."""

  def __init__(self, level, combine_fn, input_offsets, is_output):
    self.level = level
    self.combine_fn = combine_fn
    self.input_offsets = input_offsets
    self.is_output = is_output


def build_block_specs(block_specs=None):
  """Builds the list of BlockSpec objects for NAS-FPN."""
  if not block_specs:
    block_specs = NASFPN_BLOCK_SPECS
  print("Building NAS-FPN block specs: %s", block_specs)
  return [BlockSpec(*b) for b in block_specs]


class ResampleFeatureMap(nn.Module):
  """Resample Feature Map."""

  def __init__(self, input_num_filters, input_level, target_level,
               target_num_filters, norm):
    super(ResampleFeatureMap, self).__init__()
    self.input_level = input_level
    self.target_level = target_level
    if input_num_filters != target_num_filters:
      self.conv_op = nn.Sequential(
          nn.Conv3d(
              input_num_filters,
              target_num_filters,
              kernel_size=1,
              stride=1,
              padding=0,
              bias=False), _get_norm_op(norm, target_num_filters))
    else:
      self.conv_op = nn.Identity()

  def forward(self, x):
    x = self.conv_op(x)

    input_level = self.input_level
    target_level = self.target_level
    if input_level < target_level:
      stride = int(2 ** (target_level - input_level))
      x = F.max_pool3d(x, kernel_size=stride, stride=stride)
    elif input_level > target_level:
      scale = int(2 ** (input_level - target_level))
      x = F.interpolate(x, scale_factor=scale, mode="nearest")
    return x


class MergeCell(nn.Module):
  """Merge Cell."""

  def __init__(self,
               target_level,
               input_level1,
               input_level2,
               combine_fn,
               num_filters,
               norm=""):
    super(MergeCell, self).__init__()
    self.input_level1 = input_level1
    self.input_level2 = input_level2
    self.combine_fn = combine_fn
    self.resample_layer1 = ResampleFeatureMap(
        input_num_filters=num_filters,
        input_level=input_level1,
        target_level=target_level,
        target_num_filters=num_filters,
        norm=norm)
    self.resample_layer2 = ResampleFeatureMap(
        input_num_filters=num_filters,
        input_level=input_level2,
        target_level=target_level,
        target_num_filters=num_filters,
        norm=norm)
    self.conv_op = nn.Sequential(
        nn.Conv3d(
            num_filters,
            num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False), _get_norm_op(norm, num_filters))

  def forward(self, xs):
    x1 = self.resample_layer1(xs[0])
    x2 = self.resample_layer2(xs[1])
    if self.combine_fn == "sum":
      x = x1 + x2
    elif self.combine_fn == "attention":
      if self.input_level1 < self.input_level2:
        x1, x2 = x2, x1
      m = F.adaptive_max_pool3d(x1, output_size=1)
      x = x1 + x2 * F.sigmoid(m)
    if len(xs) > 2:
      x = x + sum(xs[2:])
    x = F.relu(x)
    x = self.conv_op(x)
    return x


class NASFPN(nn.Module):
  """This module implements :paper:`NASFPN`."""

  def __init__(self,
               backbone_level_to_channels,
               min_level,
               max_level,
               num_filters,
               num_repeats,
               block_specs,
               norm):
    """NASFPN initialization function.

    NASFPN attaches to an input pyramid of backbone-features at different
    levels. The topmost level corresponding to the input image size has
    level 1. The subsequent smaller feature levels correspond to level 2,
    3, 4, etc. NASFPN uses levels from min_level to max_level
    for input and produces the same level-range for output as well.

    Args:
      backbone_level_to_channels: A dictionary mapping int level to number
        of channels at corresponding level of the backbone features. For ex,
        {1: 8, 2: 16, 3: 32, 4: 64}.
      min_level: Minimum level index for NASFPN input and output.
      max_level: Maximum level index for NASFPN input and output.
      num_filters: The number of channels for the NASFPN blocks.
      num_repeats: The number of times the NASFPN blocks are repeated.
      block_specs: NASFPN block-specs.
      norm: string 'instance' or 'batch' for normalization choice.
    """
    super(NASFPN, self).__init__()
    backbone_min_level = min(backbone_level_to_channels.keys())
    if backbone_min_level > min_level:
      raise ValueError(
          "Backbone min level %d should be less or equal to FPN min level %d."%
          backbone_min_level, min_level)

    self.min_level = min_level
    self.max_level = max_level
    self.num_repeats = num_repeats
    self.block_specs = block_specs
    self.backbone_level_to_channels = backbone_level_to_channels

    self.resample_layers = nn.ModuleList()
    for level in range(min_level, max_level + 1):
      if level in self.backbone_level_to_channels:
        self.resample_layers.append(
            ResampleFeatureMap(
                input_num_filters=self.backbone_level_to_channels[level],
                input_level=level,
                target_level=level,
                target_num_filters=num_filters,
                norm=norm))
      else:
        self.resample_layers.append(
            ResampleFeatureMap(
                input_num_filters=num_filters,
                input_level=level - 1,
                target_level=level,
                target_num_filters=num_filters,
                norm=norm))

    offset_to_level = []
    for level in range(min_level, max_level + 1):
      offset_to_level.append(level)
    for block_spec in block_specs:
      offset_to_level.append(block_spec.level)

    self.feature_pyramid_networks = nn.ModuleList()
    for _ in range(num_repeats):
      fpn = nn.ModuleList()
      for _, block_spec in enumerate(block_specs):
        fpn.append(
            MergeCell(block_spec.level,
                      offset_to_level[block_spec.input_offsets[0]],
                      offset_to_level[block_spec.input_offsets[1]],
                      block_spec.combine_fn, num_filters, norm))
      self.feature_pyramid_networks.append(fpn)

  def forward(self, x):
    """Forward function for NASFPN module.

    Args:
      x: A dictionary of input backbone-features. For example,
          {1: input_feature_level1 (topmost), 2: input_feature_level2, ...}

    Returns:
      A dictionary of output-features starting from NASFPN-min-level to
      NASFPN-max-level. For example, {min_level: output_feature_min_level, ...
      , max_level: output_feature_max_level}
    """
    features = []
    for i, level in enumerate(range(self.min_level, self.max_level + 1)):
      if level in self.backbone_level_to_channels:
        features.append(self.resample_layers[i](x[level]))
      else:
        features.append(self.resample_layers[i](features[-1]))

    for rep in range(self.num_repeats):
      num_output_connections = [0] * len(features)
      num_output_levels = self.max_level - self.min_level + 1
      feat_levels = list(range(self.min_level, self.max_level + 1))
      for i, block_spec in enumerate(self.block_specs):
        feats = [
            features[block_spec.input_offsets[0]],
            features[block_spec.input_offsets[1]]
        ]
        if block_spec.is_output:
          for j, (feat, feat_level, num_output) in enumerate(
              zip(features, feat_levels, num_output_connections)):
            if num_output == 0 and feat_level == block_spec.level:
              num_output_connections[j] += 1
              feats.append(feat)
        new_node = self.feature_pyramid_networks[rep][i](feats)
        features.append(new_node)
        feat_levels.append(block_spec.level)
        num_output_connections.append(0)
      output_features = {}
      for i in range(len(features) - num_output_levels, len(features)):
        level = feat_levels[i]
        output_features[level] = features[i]
      features = []
      for level in range(self.min_level, self.max_level + 1):
        features.append(output_features[level])

    output_dict = {}
    for idx, feature in enumerate(features):
      level = idx + self.min_level
      output_dict[level] = feature
    return output_dict


class FuseOutputFeatures(nn.Module):
  """Fuse NASFPN output features for segmentation."""

  def __init__(self, nasfpn_min_level, nasfpn_max_level, nasfpn_num_filters,
               nasfpn_norm, backbone_level_to_channels):
    """FuseOutputFeatures initialization function.

    This module fuses the (a) an input pyramid of backbone-features at
    different levels and (b) the NAS-FPN feture outputs at different levels.
    The topmost level corresponding to the input image size has
    level 1. The subsequent smaller feature levels correspond to level 2,
    3, 4, etc. NASFPN uses levels from min_level to max_level
    for input and produces the same level-range for output as well.

    NOTE: For segmentation, we need nasfpn-levels to be same as backbone-levels.
    The fusion is done over pyramid levels too to generate one final
    segementation output at the topmost layer.

    Args:
      nasfpn_min_level: Minimum level index for NASFPN input and output.
      nasfpn_max_level: Maximum level index for NASFPN input and output.
      nasfpn_num_filters: The number of channels for the NASFPN blocks.
      nasfpn_norm: string 'instance' or 'batch' for normalization choice.
      backbone_level_to_channels: A dictionary mapping int level to number
        of channels at corresponding level of the backbone features. For ex,
        {1: 8, 2: 16, 3: 32, 4: 64}.
    """
    super(FuseOutputFeatures, self).__init__()

    self.nasfpn_min_level = nasfpn_min_level
    self.nasfpn_max_level = nasfpn_max_level
    self.nasfpn_num_filters = nasfpn_num_filters
    self.nasfpn_norm = nasfpn_norm

    # For segmentation, we need nasfpn-levels to be same as backbone-levels.
    if nasfpn_min_level != 1 or nasfpn_max_level != len(
        backbone_level_to_channels):
      raise ValueError(
          "For segmentation, nasfpn-levels should be same as backbone-levels.")

    upsampling_ops = {}
    merging_ops = {}
    for level in range(self.nasfpn_max_level, self.nasfpn_min_level - 1,
                       -1):
      if level == self.nasfpn_max_level:
        upsampling_ops[str(level)] = nn.Identity()
      else:
        upsampling_ops[str(level)] = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=self.nasfpn_num_filters,
                out_channels=self.nasfpn_num_filters,
                kernel_size=2,
                stride=2),
            _get_norm_op(self.nasfpn_norm, self.nasfpn_num_filters),
            _get_activation_op("leaky_relu"))
      # Combines encoder_feature_i, nasfpn_feature_i, and upsampled_feature_i+1.
      merging_input_channels = backbone_level_to_channels[
          level] + self.nasfpn_num_filters
      if level < self.nasfpn_max_level:
        # Use features from below too.
        merging_input_channels += self.nasfpn_num_filters
      merging_ops[str(level)] = nn.Sequential(
          nn.Conv3d(
              merging_input_channels,
              self.nasfpn_num_filters,
              kernel_size=3,
              stride=1,
              padding=1,
              bias=False),
          _get_norm_op(self.nasfpn_norm, self.nasfpn_num_filters),
          _get_activation_op("leaky_relu"))
    self.upsampling_op_dict = nn.ModuleDict(upsampling_ops)
    self.merging_op_dict = nn.ModuleDict(merging_ops)

    self.post_fusion_conv_op = nn.Sequential(
        nn.Conv3d(
            self.nasfpn_num_filters,
            self.nasfpn_num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False), _get_norm_op(self.nasfpn_norm,
                                      self.nasfpn_num_filters),
        _get_activation_op("leaky_relu"))

  def forward(self, nasfpn_features, backbone_features):
    """Forward function for FuseOutputFeatures module.

    Args:
      nasfpn_features: A dictionary of output-features starting from
        NASFPN-min-level to NASFPN-max-level.
        For example, {min_level: output_feature_min_level, ...
        , max_level: output_feature_max_level}
      backbone_features: A dictionary of input backbone-features. For example,
          {1: input_feature_level1 (topmost), 2: input_feature_level2, ...}

    Returns:
      Single fused_feature corresponding to min_level which is a combination
      of the features at all the levels.
    """
    level = self.nasfpn_max_level
    fused_output = self.merging_op_dict[str(level)](
        torch.cat([backbone_features[level], nasfpn_features[level]], dim=1))

    for level in range(self.nasfpn_max_level - 1, self.nasfpn_min_level - 1,
                       -1):
      upsampled_features = self.upsampling_op_dict[str(level)](fused_output)
      fused_output = self.merging_op_dict[str(level)](
          torch.cat([
              backbone_features[level], nasfpn_features[level],
              upsampled_features
          ],
                    dim=1))

    output = self.post_fusion_conv_op(fused_output)
    return output

