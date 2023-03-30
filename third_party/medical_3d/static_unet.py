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
"""Static Unet model."""

from third_party.medical_3d import basic_unet


# https://github.com/Project-MONAI/MONAI/blob/2db93d8d050feb2e245de0b9189b813693012b96/monai/networks/nets/basic_unet.py
def unet(params):
  """Returns a Unet model."""
  return basic_unet.BasicUNet(
      dimensions=3,
      in_channels=params['in_channels'],
      out_channels=params['out_channels'],
      features=params['features'],
  )
