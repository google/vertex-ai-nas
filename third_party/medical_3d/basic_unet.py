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
"""BasicUnet module from MONAI modified to input varying depth.

In addition, it adds support for tunable-encoder blocks too.
"""

from monai.networks.blocks import ADN
from monai.networks.blocks import Convolution
from monai.networks.blocks import UpSample
from monai.networks.blocks.squeeze_and_excitation import ChannelSELayer
from monai.networks.layers.factories import Conv
from monai.networks.layers.factories import Pool

import torch
import torch.nn as nn


def _get_padding_for_kernel(kernel):
  if kernel % 2 == 0:
    raise ValueError("Only odd kernel size is supported: %d" % kernel)
  return kernel // 2


class BasicBlock(nn.Module):
  """Basic block for the tunable encoder.

  Residual module enhanced with Squeeze-and-Excitation:

        ----+- conv1 -- conv2 (no act) -- SE -o--- act ---
            |                                 |
            +--------------(residual)---------+
  """

  def __init__(
      self,
      dim,
      in_chns,
      out_chns,
      kernel,
      act,
      norm,
      id_skip,
      se_ratio=0.0,
      dropout=0.0,
  ):
    """Initialization function.

    Args:
        dim: number of spatial dimensions.
        in_chns: number of input channels.
        out_chns: number of output channels.
        kernel: kernel size of the convolution. NOTE: currently only odd
          values are supported.
        act: activation type and arguments.
        norm: feature normalization type and arguments.
        id_skip: True to use skip connection.
        se_ratio: the reduction ratio of SE layer.
        dropout: dropout ratio. Defaults to no dropout.
    """
    super().__init__()
    self.id_skip = id_skip
    padding = _get_padding_for_kernel(kernel)

    # First convolution has normalization and activation.
    self.conv0 = Convolution(
        dim,
        in_chns,
        out_chns,
        kernel_size=kernel,
        act=act,
        norm=norm,
        dropout=dropout,
        padding=padding)
    # Second convolution has just normalization.
    self.conv1 = Convolution(
        dim,
        out_chns,
        out_chns,
        kernel_size=kernel,
        act=None,
        norm=norm,
        dropout=dropout,
        padding=padding)

    if se_ratio:
      r = int(1.0/se_ratio)
      self.se_layer = ChannelSELayer(
          spatial_dims=dim, in_channels=out_chns, r=r)
    else:
      self.se_layer = nn.Identity()
    if in_chns != out_chns and self.id_skip:
      self.residual_conv = Convolution(
          dim,
          in_chns,
          out_chns,
          kernel_size=1,
          act=None,
          norm=None,
          dropout=None,
          padding=0)
    else:
      self.residual_conv = nn.Identity()
    self.last_act = ADN(in_channels=out_chns, act=act, norm=None, dropout=None)

  def forward(self, x):
    residual = x
    x = self.conv0(x)
    x = self.conv1(x)
    x = self.se_layer(x)
    if self.id_skip:
      x += self.residual_conv(residual)
    x = self.last_act(x)
    return x


class BottleneckBlock(nn.Module):
  """Bottleneck block for the tunable encoder.

  Residual module enhanced with bottleneck and Squeeze-and-Excitation:

        ----+- conv1 -- conv2 -- conv3 (no act) -- SE -o--- act ---
            |                                          |
            +------------------(residual)--------------+

  NOTE: For this block, final-channels = out_chns * bottleneck_ratio
  """

  def __init__(
      self,
      dim,
      in_chns,
      out_chns,
      kernel,
      act,
      norm,
      id_skip,
      se_ratio=0.0,
      dropout=0.0,
      bottleneck_ratio=4
  ):
    """Initialization function.

    Args:
        dim: number of spatial dimensions.
        in_chns: number of input channels.
        out_chns: number of output channels of conv1. The final number
          of output channels is out_chns * bottleneck_ratio.
        kernel: kernel size of the convolution. NOTE: currently only odd
          values are supported.
        act: activation type and arguments.
        norm: feature normalization type and arguments.
        id_skip: True to use skip connection.
        se_ratio: the reduction ratio of SE layer.
        dropout: dropout ratio. Defaults to no dropout.
        bottleneck_ratio: Factor by which number of channels are scaled.
    """
    super().__init__()
    self.id_skip = id_skip
    padding = _get_padding_for_kernel(kernel)

    # First convolution has normalization and activation.
    # Kernel size 1.
    self.conv0 = Convolution(
        dim,
        in_chns,
        out_chns,
        kernel_size=1,
        act=act,
        norm=norm,
        dropout=dropout,
        padding=0)
    # Second convolution has normalization and activation.
    self.conv1 = Convolution(
        dim,
        out_chns,
        out_chns,
        kernel_size=kernel,
        act=act,
        norm=norm,
        dropout=dropout,
        padding=padding)
    # Thrid convolution has no activation and expands the channels.
    # Kernel size 1.
    bottleneck_chns = bottleneck_ratio * out_chns
    self.conv2 = Convolution(
        dim,
        out_chns,
        bottleneck_chns,
        kernel_size=1,
        act=None,
        norm=norm,
        dropout=dropout,
        padding=0)

    if se_ratio:
      r = int(1.0/se_ratio)
      self.se_layer = ChannelSELayer(
          spatial_dims=dim, in_channels=bottleneck_chns, r=r)
    else:
      self.se_layer = nn.Identity()
    if in_chns != bottleneck_chns and self.id_skip:
      self.residual_conv = Convolution(
          dim,
          in_chns,
          bottleneck_chns,
          kernel_size=1,
          act=None,
          norm=None,
          dropout=None,
          padding=0)
    else:
      self.residual_conv = nn.Identity()
    self.last_act = ADN(
        in_channels=bottleneck_chns, act=act, norm=None, dropout=None)

  def forward(self, x):
    residual = x
    x = self.conv0(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.se_layer(x)
    if self.id_skip:
      x += self.residual_conv(residual)
    x = self.last_act(x)
    return x


class TwoConv(nn.Sequential):
  """two convolutions."""

  def __init__(
      self,
      dim,
      in_chns,
      out_chns,
      act,
      norm,
      dropout=0.0,
  ):
    """Initialization function.

    Args:
        dim: number of spatial dimensions.
        in_chns: number of input channels.
        out_chns: number of output channels.
        act: activation type and arguments.
        norm: feature normalization type and arguments.
        dropout: dropout ratio. Defaults to no dropout.
    """
    super().__init__()

    conv_0 = Convolution(
        dim, in_chns, out_chns, act=act, norm=norm, dropout=dropout, padding=1)
    conv_1 = Convolution(
        dim, out_chns, out_chns, act=act, norm=norm, dropout=dropout, padding=1)
    self.add_module("conv_0", conv_0)
    self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
  """maxpooling downsampling and two convolutions."""

  def __init__(
      self,
      dim,
      in_chns,
      out_chns,
      act,
      norm,
      dropout=0.0,
  ):
    """Initialization function.

    Args:
        dim: number of spatial dimensions.
        in_chns: number of input channels.
        out_chns: number of output channels.
        act: activation type and arguments.
        norm: feature normalization type and arguments.
        dropout: dropout ratio. Defaults to no dropout.
    """
    super().__init__()

    max_pooling = Pool["MAX", dim](kernel_size=2)
    convs = TwoConv(dim, in_chns, out_chns, act, norm, dropout)
    self.add_module("max_pooling", max_pooling)
    self.add_module("convs", convs)


class CustomDown(nn.Sequential):
  """Sequence of maxpooling-downsampling and a custom convolution block."""

  def __init__(self, dim, conv_block):
    """Initialization function.

    Args:
        dim: number of spatial dimensions.
        conv_block: a custom convolution block to replace TwoConv.
    """
    super().__init__()
    max_pooling = Pool["MAX", dim](kernel_size=2)
    self.add_module("max_pooling", max_pooling)
    self.add_module("convs", conv_block)


class UpCat(nn.Module):
  """upsampling, concatenation with the encoder feature map, two convolutions."""

  def __init__(
      self,
      dim,
      in_chns,
      cat_chns,
      out_chns,
      act,
      norm,
      dropout=0.0,
      upsample="deconv",
      halves=True,
  ):
    """Initialization function.

    Args:
        dim: number of spatial dimensions.
        in_chns: number of input channels to be upsampled.
        cat_chns: number of channels from the decoder.
        out_chns: number of output channels.
        act: activation type and arguments.
        norm: feature normalization type and arguments.
        dropout: dropout ratio. Defaults to no dropout.
        upsample: upsampling mode, available options are ``"deconv"``,
          ``"pixelshuffle"``, ``"nontrainable"``.
        halves: whether to halve the number of channels during upsampling.
    """
    super().__init__()

    up_chns = in_chns // 2 if halves else in_chns
    self.upsample = UpSample(dim, in_chns, up_chns, 2, mode=upsample)
    self.convs = TwoConv(dim, cat_chns + up_chns, out_chns, act, norm, dropout)

  def forward(self, x, x_e):
    """Forward function for upward path.

    Args:
        x: features to be upsampled.
        x_e: features from the encoder.

    Returns:
      Output of upward-path block.
    """
    x_0 = self.upsample(x)

    # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
    dimensions = len(x.shape) - 2
    sp = [0] * (dimensions * 2)
    for i in range(dimensions):
      if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
        sp[i * 2 + 1] = 1
    x_0 = torch.nn.functional.pad(x_0, sp, "replicate")

    x = self.convs(torch.cat([x_e, x_0],
                             dim=1))  # input channels: (cat_chns + up_chns)
    return x


def build_down_ops(dimensions, in_channels, features, act, norm, dropout,
                   num_levels):
  """Returns a list of down-level ops."""
  down_ops = []
  conv_0 = TwoConv(dimensions, in_channels, features[0], act, norm, dropout)
  down_ops.append(conv_0)
  for level in range(1, num_levels):
    down_op = Down(dimensions, features[level - 1], features[level], act, norm,
                   dropout)
    down_ops.append(down_op)
  down_ops = nn.ModuleList(down_ops)
  return down_ops


class BasicUNet(nn.Module):
  """Basic Unet."""

  def __init__(
      self,
      dimensions=3,
      in_channels=1,
      out_channels=2,
      features=(32, 32, 64, 128, 256),
      act=("LeakyReLU", {
          "negative_slope": 0.1,
          "inplace": True
      }),
      norm=("instance", {
          "affine": True
      }),
      dropout=0.0,
      upsample="deconv",
      custom_down_ops=None,
  ):
    """A UNet implementation with 1D/2D/3D supports.

    Based on:
        Falk et al. "U-Net – Deep Learning for Cell Counting, Detection,
        and
        Morphometry". Nature Methods 16, 67–70 (2019), DOI:
        http://dx.doi.org/10.1038/s41592-018-0261-2
    Args:
        dimensions: number of spatial dimensions. Defaults to 3 for spatial
          3D inputs.
        in_channels: number of input channels. Defaults to 1.
        out_channels: number of output channels. Defaults to 2.
        features: integers as numbers of features. The values correspond to the
          encoder feature sizes.
        act: activation type and arguments. Defaults to LeakyReLU.
        norm: feature normalization type and arguments. Defaults to instance
          norm.
        dropout: dropout ratio. Defaults to no dropout.
        upsample: upsampling mode, available options are ``"deconv"``,
          ``"pixelshuffle"``, ``"nontrainable"``.
        custom_down_ops: Customized down-ops for the encoder.
    """
    super().__init__()

    self.num_levels = len(features)
    if self.num_levels < 2:
      raise ValueError("The Unet should atleast have a depth of two.")
    fea = list(features)
    print(f"BasicUNet features: {fea}.")

    if custom_down_ops:
      self.down_ops = custom_down_ops
    else:
      self.down_ops = build_down_ops(
          dimensions=dimensions,
          in_channels=in_channels,
          features=fea,
          act=act,
          norm=norm,
          dropout=dropout,
          num_levels=self.num_levels)

    self.up_ops = []
    upcat_0 = UpCat(
        dimensions,
        fea[1],
        fea[0],
        fea[0],
        act,
        norm,
        dropout,
        upsample,
        halves=False)
    self.up_ops.append(upcat_0)
    for level in range(1, self.num_levels - 1):
      upcat_op = UpCat(dimensions, fea[level + 1], fea[level], fea[level], act,
                       norm, dropout, upsample)
      self.up_ops.append(upcat_op)
    self.up_ops = nn.ModuleList(self.up_ops)

    self.final_conv = Conv["conv", dimensions](
        fea[0], out_channels, kernel_size=1)

  def forward(self, x):
    """Forward function for BasicUnet.

    Args:
        x: input should have spatially N dimensions ``(Batch, in_channels,
          dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`. It is
          recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling
          inputs have even edge lengths.

    Returns:
        A torch Tensor of "raw" predictions in shape
        ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
    """
    # Down path.
    down_features = []
    for level in range(self.num_levels):
      x = self.down_ops[level](x)
      down_features.append(x)

    # Up Path.
    x = down_features[-1]
    for level in range(self.num_levels-2, -1, -1):
      x = self.up_ops[level](x, down_features[level])

    # Final conv.
    logits = self.final_conv(x)
    return logits
