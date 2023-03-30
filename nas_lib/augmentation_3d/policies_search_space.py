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
"""3D augmentation search-space.

NOTE: These classes decouple implementation of search-space from its
definition. This way it is only dependent on PyGlove and not on
TensorFlow etc.
"""

import pyglove as pg


# NOTE: This class is just an example to create your own search-space.
# The user can add their own search-space classes here.
@pg.members([
    ('bbox_rotation', pg.typing.Float(min_value=0.),
     'The maximum rotation amount.'),
    ('world_rotation', pg.typing.Float(min_value=0.),
     'The maximum rotation amount.'),
    ('world_scale_min', pg.typing.Float(min_value=0.),
     'Min value for scaling.'),
    ('world_scale_max', pg.typing.Float(min_value=0.),
     'Max value for scaling.'),
    ('frustum_dropout_theta', pg.typing.Float(min_value=0., default=0.03),
     'Theta angle width for dropping points.'),
    ('frustum_dropout_phi', pg.typing.Float(min_value=0., default=0.0),
     'Phi angle width for dropping points.'),
])
class BasicSearchSpaceSpecBuilder(pg.Object):
  """Define the spec for a basic augmentation search-space."""
  pass
