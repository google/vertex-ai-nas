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

"""Factory to create TF Vision experiment config."""

from official.core import config_definitions as cfg
from official.core import exp_factory


def config_generator(model):
  """Generates and returns a default experiment config."""
  if model == 'classification':
    default_config = exp_factory.get_exp_config('image_classification_nas')
  elif model == 'retinanet':
    default_config = exp_factory.get_exp_config('retinanet_nas')
  elif model == 'segmentation':
    default_config = exp_factory.get_exp_config('semantic_segmentation_nas')
  elif model == 'pointpillars':
    default_config = exp_factory.get_exp_config('pointpillars_nas')
  elif model == 'pointpillars_baseline':
    default_config = exp_factory.get_exp_config('pointpillars_baseline')
  else:
    raise ValueError('Model %s is not supported.' % model)

  return default_config
