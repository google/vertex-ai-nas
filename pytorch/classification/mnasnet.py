# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""Implementation of MnasNet model in PyTorch."""

import collections
import functools

from pytorch.classification import mnasnet_constants
import pyglove as pg
import torch
from torch import nn


class BlockSpec(object):
  """A container class that specifies the block configuration for MnasNet."""

  def __init__(self,
               num_repeats,
               block_fn,
               expand_ratio,
               kernel_size,
               se_ratio,
               output_filters):
    self.num_repeats = num_repeats
    self.block_fn = block_fn
    self.expand_ratio = expand_ratio
    self.kernel_size = kernel_size
    self.se_ratio = se_ratio
    self.output_filters = output_filters


def build_block_specs(block_specs=None):
  """Builds the list of BlockSpec objects for MnasNet."""
  if not block_specs:
    block_specs = mnasnet_constants.MNASNET_A1_BLOCK_SPECS
  if len(block_specs) != mnasnet_constants.MNASNET_NUM_BLOCKS:
    raise ValueError('The block_specs of MnasNet must be a length {} list.'
                     .format(mnasnet_constants.MNASNET_NUM_BLOCKS))
  return [BlockSpec(*b) for b in block_specs]


class Swish(nn.Module):
  """Swish activation function."""

  def forward(self, x):
    return torch.sigmoid(x) * x


class BatchNormActivation(nn.Module):
  """Combined Batch Normalization and Activation layers."""

  def __init__(self,
               channels,
               momentum=0.9997,
               epsilon=1e-4,
               use_sync_bn=False,
               activation='relu'):
    super(BatchNormActivation, self).__init__()
    if not use_sync_bn:
      self.bn = nn.BatchNorm2d(channels, eps=epsilon, momentum=momentum)
    else:
      raise ValueError('SyncBatchNorm not implemented.')

    if activation == 'relu':
      self.act = nn.ReLU()
    elif activation == 'swish':
      self.act = Swish()
    else:
      raise ValueError('Activation {} not implemented.'.format(activation))

  def forward(self, x):
    return self.act(self.bn(x))


class SE(nn.Module):
  """Squeeze-and-excitation layer."""

  def __init__(self, channels, se_ratio):
    super(SE, self).__init__()
    num_reduced_channels = max(1, int(channels * se_ratio))
    self.se = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Conv2d(
            in_channels=channels,
            out_channels=num_reduced_channels,
            kernel_size=(1, 1)),
        Swish(),
        nn.Conv2d(
            in_channels=num_reduced_channels,
            out_channels=channels,
            kernel_size=(1, 1)),
        nn.Sigmoid(),
    )

  def forward(self, x):
    return self.se(x) * x


class DropBlock(nn.Module):

  def __init__(self):
    super(DropBlock, self).__init__()
    pass

  def forward(self, x):
    return x


class DropConnect(nn.Module):

  def __init__(self, drop_connect_rate):
    super(DropConnect, self).__init__()
    pass

  def forward(self, x):
    return x


class MBConvBlock(nn.Module):
  """Mobile Inverted Residual Bottleneck Block."""

  def __init__(self,
               in_channels,
               out_channels,
               expand_ratio,
               strides,
               kernel_size=3,
               se_ratio=None,
               batch_norm_activation=BatchNormActivation,
               dropblock=DropBlock,
               drop_connect_rate=None):
    super(MBConvBlock, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.strides = strides

    if expand_ratio != 1.0:
      self.expand = nn.Sequential(
          nn.Conv2d(
              in_channels=in_channels,
              out_channels=in_channels * expand_ratio,
              kernel_size=(1, 1),
              bias=False),
          batch_norm_activation(in_channels * expand_ratio),
          dropblock(),
      )
    else:
      self.expand = None

    self.depthwise = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels * expand_ratio,
            out_channels=in_channels * expand_ratio,
            kernel_size=kernel_size,
            stride=strides,
            padding=kernel_size // 2,
            groups=in_channels * expand_ratio,
            bias=False),
        batch_norm_activation(in_channels * expand_ratio),
        dropblock(),
    )

    if se_ratio is not None and se_ratio > 0 and se_ratio <= 1:
      self.se = SE(in_channels * expand_ratio, se_ratio)
    else:
      self.se = None

    self.project = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels * expand_ratio,
            out_channels=out_channels,
            kernel_size=(1, 1),
            bias=False),
        batch_norm_activation(out_channels),
        dropblock(),
    )

    if in_channels == out_channels and strides == 1:
      if drop_connect_rate:
        self.drop_connect = DropConnect(drop_connect_rate)
      else:
        self.drop_connect = None

  def forward(self, x):
    shortcut = x
    if self.expand is not None:
      x = self.expand(x)
    x = self.depthwise(x)
    if self.se is not None:
      x = self.se(x)
    x = self.project(x)
    if self.in_channels == self.out_channels and self.strides == 1:
      if self.drop_connect is not None:
        x = self.drop_connect(x)
      x = x + shortcut
    return x


class ClassificationHead(nn.Module):
  """Classification head."""

  def __init__(self,
               num_classes,
               channels,
               endpoints_num_filters=0,
               aggregation='top',
               dropout_rate=0.0,
               batch_norm_activation=BatchNormActivation):
    super(ClassificationHead, self).__init__()
    ops = []
    ops.append(collections.OrderedDict())
    if endpoints_num_filters > 0:
      self.conv0 = nn.Sequential(
          nn.Conv2d(
              in_channels=channels,
              out_channels=endpoints_num_filters,
              kernel_size=(1, 1),
              bias=False),
          batch_norm_activation(endpoints_num_filters),
      )
      channels = endpoints_num_filters
    else:
      self.conv0 = None
    self.pool = nn.AdaptiveAvgPool2d((1, 1))
    if dropout_rate > 0.0:
      self.dropout = nn.Dropout2d(dropout_rate)
    else:
      self.dropout = None
    self.fc = nn.Linear(
        in_features=channels,
        out_features=num_classes)

  def forward(self, x):
    if self.conv0:
      x = self.conv0(x)
    x = self.pool(x)
    x = x.view(x.shape[0], -1)
    if self.dropout:
      x = self.dropout(x)
    x = self.fc(x)
    return x


class MnasNet(nn.Module):
  """Class to build MnasNet family models."""

  def __init__(self,
               block_specs=build_block_specs(),
               batch_norm_activation=BatchNormActivation):
    super(MnasNet, self).__init__()
    self._block_specs = block_specs

    self.stem = nn.Sequential(
        nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False),
        batch_norm_activation(32),
    )

    self.blocks = nn.ModuleList()
    for i, block_spec in enumerate(block_specs):
      block = collections.OrderedDict()
      for j in range(block_spec.num_repeats):
        strides = 1 if j > 0 else mnasnet_constants.MNASNET_STRIDES[i]
        if i == 0 and j == 0:
          in_channels = 32
        elif j == 0:
          in_channels = block_specs[i - 1].output_filters
        else:
          in_channels = block_spec.output_filters

        if block_spec.block_fn == 'conv':
          block['{}'.format(j)] = nn.Sequential(
              nn.Conv2d(
                  in_channels=in_channels,
                  out_channels=block_spec.output_filters,
                  kernel_size=block_spec.kernel_size,
                  stride=strides,
                  padding=block_spec.kernel_size // 2,
                  bias=False),
              batch_norm_activation(block_spec.output_filters),
          )
        else:
          block['{}'.format(j)] = MBConvBlock(
              in_channels=in_channels,
              out_channels=block_spec.output_filters,
              expand_ratio=block_spec.expand_ratio,
              strides=strides,
              kernel_size=block_spec.kernel_size,
              se_ratio=block_spec.se_ratio,
              batch_norm_activation=batch_norm_activation)
      block = nn.Sequential(block)
      self.blocks.append(block)

  def forward(self, x):
    x = self.stem(x)

    endpoints = []
    for block in self.blocks:
      x = block(x)
      endpoints.append(x)

    return endpoints[-1]


def build_mnasnet_model(params):
  """Build MnasNet model."""
  batch_norm_params = params.batch_norm_activation
  batch_norm_activation = functools.partial(
      BatchNormActivation,
      momentum=batch_norm_params.batch_norm_momentum,
      epsilon=batch_norm_params.batch_norm_epsilon,
      use_sync_bn=batch_norm_params.use_sync_bn,
      activation=batch_norm_params.activation)
  block_specs = pg.from_json_str(params.tunable_mnasnet.block_specs)
  backbone = MnasNet(block_specs, batch_norm_activation)
  head_params = params.classification_head
  head = ClassificationHead(
      params.architecture.num_classes,
      block_specs[-1].output_filters,
      head_params.endpoints_num_filters,
      head_params.aggregation,
      head_params.dropout_rate,
      batch_norm_activation)
  return nn.Sequential(backbone, head)
