# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Factory to provide model configs."""

from tf1.detection.configs import classification_config
from tf1.detection.configs import maskrcnn_config
from tf1.detection.configs import retinanet_config
from tf1.detection.configs import segmentation_config
from tf1.detection.configs import shapemask_config
from tf1.detection.projects.vild.configs import vild_config
from tf1.hyperparameters import params_dict


def config_generator(model):
  """Model function generator."""
  if model == 'classification':
    default_config = classification_config.CLASSIFICATION_CFG
    restrictions = classification_config.CLASSIFICATION_RESTRICTIONS
  elif model == 'retinanet':
    default_config = retinanet_config.RETINANET_CFG
    restrictions = retinanet_config.RETINANET_RESTRICTIONS
  elif model == 'mask_rcnn' or model == 'cascade_mask_rcnn':
    default_config = maskrcnn_config.MASKRCNN_CFG
    restrictions = maskrcnn_config.MASKRCNN_RESTRICTIONS
  elif model == 'segmentation':
    default_config = segmentation_config.SEGMENTATION_CFG
    restrictions = segmentation_config.SEGMENTATION_RESTRICTIONS
  elif model == 'shapemask':
    default_config = shapemask_config.SHAPEMASK_CFG
    restrictions = shapemask_config.SHAPEMASK_RESTRICTIONS
  elif model == 'vild':
    default_config = vild_config.VILD_CFG
    restrictions = vild_config.VILD_RESTRICTIONS
  else:
    raise ValueError('Model %s is not supported.' % model)

  return params_dict.ParamsDict(default_config, restrictions)
