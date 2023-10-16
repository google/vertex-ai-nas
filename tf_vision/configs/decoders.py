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
"""Decoder configurations."""
import dataclasses
from typing import Optional

from official.modeling import hyperparams
from official.vision.beta.configs import decoders


@dataclasses.dataclass
class Identity(hyperparams.Config):
  """Identity config."""


@dataclasses.dataclass
class TunableNASFPN(hyperparams.Config):
  """Tunable NASFPN config."""
  num_filters: int = 256
  num_repeats: int = 5
  use_separable_conv: bool = False
  block_specs_json: Optional[str] = None


@dataclasses.dataclass
class Decoder(hyperparams.OneOfConfig):
  """Configuration for decoders.

  Attributes:
    type: 'str', type of decoder be used, one of the fields below.
    identity: identity decoder config.
    tunable_nasfpn: tunable NASFPN decoder.
    fpn: FPN decoder.
    aspp: ASPP decoder.
  """
  type: Optional[str] = None
  identity: Identity = dataclasses.field(default_factory=Identity)
  tunable_nasfpn: TunableNASFPN = dataclasses.field(
      default_factory=TunableNASFPN
  )
  fpn: decoders.FPN = dataclasses.field(default_factory=decoders.FPN)
  aspp: decoders.ASPP = dataclasses.field(default_factory=decoders.ASPP)
