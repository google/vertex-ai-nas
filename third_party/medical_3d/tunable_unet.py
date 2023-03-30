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
"""Model definition for the tunable UNet.

NOTE: only the encoder is tunable.
"""

import functools

from gcs_utils import gcs_path_utils
from third_party.medical_3d import basic_unet
import pyglove as pg
import torch.nn as nn

_MONAI_INSTANCE_NORMALIZATION = ("instance", {"affine": True})

# Factor by which number of channels are scaled for the
# basic_unet.BottleneckBlock.
_BOTTLENECK_RATIO = 4


def _get_monai_activation(activation):
  # NOTE: Monai activation is created as a tuple of
  # (activation_name_str, activation_params_dict).
  if activation == "relu":
    return ("relu", {})
  elif activation == "swish":
    return ("swish", {})
  else:
    raise ValueError(
        "Activation function {} is not supported.".format(activation))


def _get_encoder_output_features(block_specs):
  """Returns encoder-output-features at each level based on the block type."""
  encoder_output_features = []
  for block_spec in block_specs.blocks:
    if block_spec.block_type == "basic_block":
      encoder_output_features.append(block_spec.num_filters)
    elif block_spec.block_type == "bottleneck_block":
      encoder_output_features.append(block_spec.num_filters * _BOTTLENECK_RATIO)
    else:
      raise ValueError("Unknown block_type %s." % block_spec.block_type)
  return encoder_output_features


def tunable_unet(params):
  """Returns a tunable-Unet model."""
  block_specs_json = gcs_path_utils.gcs_fuse_path(params["block_specs_json"])
  if block_specs_json.endswith(".json"):
    with open(block_specs_json, "r") as f:
      block_specs_json = f.read()
  block_specs = pg.from_json_str(block_specs_json)

  activation = block_specs.activation
  encoder_output_features = _get_encoder_output_features(block_specs)

  # Create customized downblock-ops based on block_specs to replace
  # the Unet-encoder.
  down_ops = []
  for level_idx, block_spec in enumerate(block_specs.blocks):
    if block_spec.block_type == "basic_block":
      block_fn = basic_unet.BasicBlock
    elif block_spec.block_type == "bottleneck_block":
      block_fn = functools.partial(
          basic_unet.BottleneckBlock, bottleneck_ratio=_BOTTLENECK_RATIO)
    else:
      raise ValueError("Unknown block_type %s." % block_spec.block_type)

    # Create conv-block (with repeats) at the current level.
    repeated_conv_blocks = nn.Sequential()
    for repeat_idx in range(block_spec.num_repeats):
      if repeat_idx == 0:
        # First block in repeats.
        in_channels = params[
            "in_channels"] if level_idx == 0 else encoder_output_features[
                level_idx - 1]
      else:
        in_channels = encoder_output_features[level_idx]
      conv_block = block_fn(
          dim=3,
          in_chns=in_channels,
          out_chns=block_spec.num_filters,
          kernel=block_spec.kernel_size,
          act=_get_monai_activation(activation),
          norm=_MONAI_INSTANCE_NORMALIZATION,
          id_skip=block_spec.id_skip,
          se_ratio=block_spec.se_ratio)
      repeated_conv_blocks.add_module(
          "conv_block_lev{}_idx{}".format(level_idx, repeat_idx), conv_block)

    # Create down-op at this level.
    # NOTE: First level does not do downsampling.
    down_op = repeated_conv_blocks if level_idx == 0 else basic_unet.CustomDown(
        dim=3, conv_block=repeated_conv_blocks)
    down_ops.append(down_op)
  down_ops = nn.ModuleList(down_ops)

  # Pass down-ops created from block-specs to the Unet.
  return basic_unet.BasicUNet(
      dimensions=3,
      in_channels=params["in_channels"],
      out_channels=params["out_channels"],
      features=encoder_output_features,
      act=("LeakyReLU", {
          "negative_slope": 0.1,
          "inplace": True
      }),
      norm=_MONAI_INSTANCE_NORMALIZATION,
      dropout=0.0,
      upsample="deconv",
      custom_down_ops=down_ops)
