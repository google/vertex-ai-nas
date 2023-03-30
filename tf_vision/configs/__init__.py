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

"""Configs package definition."""

from tf_vision.configs import image_classification
from tf_vision.configs import pointpillars as tunable_pointpillars
from tf_vision.configs import retinanet
from tf_vision.configs import semantic_segmentation

from tf_vision.pointpillars.configs import pointpillars as baseline_pointpillars
