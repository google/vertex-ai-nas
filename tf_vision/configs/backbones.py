# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Backbone configurations."""
import dataclasses
from typing import Optional

from official.modeling import hyperparams
from official.vision.beta.configs import backbones


@dataclasses.dataclass
class TunableSpineNet(hyperparams.Config):
  """Tunable SpineNet config."""
  # model_id is the model depth for Spinenet. 49 means Spinenet49.
  model_id: str = '49'
  min_level: int = 3
  max_level: int = 7
  block_specs_json: Optional[str] = None
  endpoints_num_filters: int = 256
  resample_alpha: float = 0.5
  block_repeats: int = 1
  filter_size_scale: float = 1.0
  init_stochastic_depth_rate: float = 0.0


@dataclasses.dataclass
class TunableSpineNetMBConv(hyperparams.Config):
  """Tunable SpineNetMBConv config."""
  # model_id is the model depth for SpinenetMBConv. 49 means Spinenet49.
  model_id: str = '49'
  min_level: int = 3
  max_level: int = 7
  block_specs_json: Optional[str] = None
  se_ratio: float = 0.2
  endpoints_num_filters: int = 256
  block_repeats: int = 1
  filter_size_scale: float = 1.0
  init_stochastic_depth_rate: float = 0.0


@dataclasses.dataclass
class TunableMnasNet(hyperparams.Config):
  """Tunable MnasNet config."""
  block_specs_json: Optional[str] = None
  endpoints_num_filters: int = 1280
  features_only: bool = False


@dataclasses.dataclass
class MnasNet(hyperparams.Config):
  """MnasNet config."""
  endpoints_num_filters: int = 1280


@dataclasses.dataclass
class Backbone(hyperparams.OneOfConfig):
  """Configuration for backbones.

  Attributes:
    type: 'str', type of backbone be used, one of the fields below.
    tunable_spinenet: tunable spinenet backbone config.
    tunable_spinenet_mbconv: tunable spinenet mbconv backbone config.
    tunable_mnasnet: tunable MnasNet backbone config.
    resnet: ResNet backbone config.
    efficientnet: EfficientNet backbone config.
    mnasnet: MnasNet backbone config.
  """
  type: Optional[str] = None
  tunable_spinenet: TunableSpineNet = dataclasses.field(
      default_factory=TunableSpineNet
  )
  tunable_spinenet_mbconv: TunableSpineNetMBConv = dataclasses.field(
      default_factory=TunableSpineNetMBConv
  )
  tunable_mnasnet: TunableMnasNet = dataclasses.field(
      default_factory=TunableMnasNet
  )
  resnet: backbones.ResNet = dataclasses.field(default_factory=backbones.ResNet)
  efficientnet: backbones.EfficientNet = dataclasses.field(
      default_factory=backbones.EfficientNet
  )
  mnasnet: MnasNet = dataclasses.field(default_factory=MnasNet)
