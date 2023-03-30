# Lint: python3
# Copyright 2020 Google Research. All Rights Reserved.
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
"""Image augmentation search-space.

NOTE: These classes decouple implementation of search-space from its
definition. This way it is only dependent on PyGlove and not on
TensorFlow etc.
"""

import pyglove as pg


@pg.members([
    ("num_ops", pg.typing.Int(min_value=1), "Number of augmentation ops."),
    ("magnitude", pg.typing.Int(min_value=0, max_value=10),
     "Magnitude of augmentation ops.")
])
class RandAugmentDetectionSpecBuilder(pg.Object):
  """Define the spec for RandAugment detection search-space."""
  pass


@pg.members([
    ("num_ops", pg.typing.Int(min_value=1), "Number of augmentation ops."),
    ("magnitude", pg.typing.Int(min_value=0, max_value=10),
     "Magnitude of augmentation ops.")
])
class RandAugmentSegmentationSpecBuilder(pg.Object):
  """Define the spec for RandAugment segmentation search-space."""
  pass
