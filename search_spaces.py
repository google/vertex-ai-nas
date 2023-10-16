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
"""Definition of NAS search spaces.

It is used by both `cloud_search_main` and `nas client` (used to generate
`search_space_spec` for Cloud API) to ensure consistency between the two
places.
"""

from tf_vision.search_spaces import tunable_pointpillars_search_space


import pyglove as pg

from nas_lib.augmentation_2d import policies_search_space as augmentation_search_space
from nas_architecture import tunable_autoaugment_search_space
from nas_architecture import tunable_efficientnetv2_search_space
from nas_architecture import tunable_mnasnet_search_space
from nas_architecture import tunable_nasfpn_search_space
from nas_architecture import tunable_spinenet_search_space




@pg.members([
    ("image_size", pg.typing.Int(default=512), "Input image size."),
    ("endpoints_num_filters", pg.typing.Int(default=256),
     "The number of filters of the endpoint features that SpineNet produces."),
    ("filter_size_scale", pg.typing.Float(default=1.0),
     "The filter size scaling factor for SpineNet backbone."),
    ("resample_alpha", pg.typing.Float(default=0.5),
     "The resample alpha parameters for SpineNet backbone."),
    ("block_repeats", pg.typing.Int(default=1),
     "The number of repeats for a block in SpineNet backbone."),
    ("head_num_convs", pg.typing.Int(default=3),
     "The number of convolution layers to be added in RetinaNet head."),
    ("head_num_filters", pg.typing.Int(default=3),
     "The number of filters of the convolution layers in RetinaNet head."),
])
class SpineNetScalingSpecBuilder(pg.Object):
  """Define the CompoundScaling spec for SpineNet."""
  pass


def get_search_space(search_space):
  """Returns search space definition."""
  if search_space == "nasfpn":
    return nasfpn_search_space()
  elif search_space == "spinenet":
    return spinenet_search_space()
  elif search_space == "spinenet_v2":
    return spinenet_v2_search_space()
  elif search_space == "spinenet_mbconv":
    return spinenet_mbconv_search_space()
  elif search_space == "mnasnet":
    return mnasnet_search_space()
  elif search_space == "efficientnet_v2":
    return efficientnet_v2_search_space()
  elif search_space == "pointpillars":
    return pointpillars_search_space()

  elif search_space == "randaugment_detection":
    return randaugment_detection_search_space()
  elif search_space == "randaugment_segmentation":
    return randaugment_segmentation_search_space()
  elif search_space == "autoaugment_detection":
    return autoaugment_detection_search_space()
  elif search_space == "autoaugment_segmentation":
    return autoaugment_segmentation_search_space()

  elif search_space == "spinenet_scaling":
    return spinenet_scaling_search_space()
  else:
    raise ValueError("Unexpected search_space value: {}".format(search_space))


def nasfpn_search_space():
  """Returns NAS-FPN search space."""
  return tunable_nasfpn_search_space.nasfpn_search_space(
      min_level=3,
      max_level=7,
      level_candidates=[4, 5, 6, 7],
      num_intermediate_blocks=2)


def spinenet_search_space():
  """Returns SpineNet search space."""
  return tunable_spinenet_search_space.spinenet_search_space(
      min_level=3,
      max_level=7,
      intermediate_blocks_alloc={
          2: 1,
          3: 3,
          4: 5,
          5: 2
      },
      intermediate_level_offsets=[-1, 0, 1, 2],
      block_fn_candidates=["residual", "bottleneck"],
      num_blocks_search_window=4)


def spinenet_v2_search_space():
  """Returns SpineNet-V2 search space."""
  return tunable_spinenet_search_space.spinenet_search_space(
      min_level=3,
      max_level=7,
      intermediate_blocks_alloc={
          2: 1,
          3: 3,
          4: 5,
          5: 1
      },
      intermediate_level_offsets=[-1, 0, 1],
      block_fn_candidates=["bottleneck"],
      num_blocks_search_window=4)


def spinenet_mbconv_search_space():
  # Blocks allocation based on EfficientNet-B0.
  return tunable_spinenet_search_space.spinenet_search_space(
      min_level=3,
      max_level=7,
      intermediate_blocks_alloc={
          2: 1,
          3: 1,
          4: 5,
          5: 4
      },
      intermediate_level_offsets=[-1, 0, 1, 2],
      block_fn_candidates=["mbconv"],
      num_blocks_search_window=4)


def mnasnet_search_space():
  """Returns MNasNet search space."""
  return tunable_mnasnet_search_space.mnasnet_search_space(
      reference="mobilenet_v2"
  )


def efficientnet_v2_search_space():
  return tunable_efficientnetv2_search_space.efficientnetv2_search_space()


def pointpillars_search_space():
  """Returns Lidar search space."""
  return tunable_pointpillars_search_space.pointpillars_search_space()



def randaugment_detection_search_space():
  return augmentation_search_space.RandAugmentDetectionSpecBuilder(
      num_ops=pg.one_of(list(range(1, 3))),
      magnitude=pg.one_of(list(range(1, 11))))


def randaugment_segmentation_search_space():
  return augmentation_search_space.RandAugmentSegmentationSpecBuilder(
      num_ops=pg.one_of(list(range(1, 3))),
      magnitude=pg.one_of(list(range(1, 11))))


def autoaugment_detection_search_space():
  total_num_ops = tunable_autoaugment_search_space.DETECTION_OPS_COUNT
  return tunable_autoaugment_search_space.autoaugment_search_space(
      total_num_ops=total_num_ops, num_ops_per_sub_policy=2, num_sub_policies=5)


def autoaugment_segmentation_search_space():
  total_num_ops = tunable_autoaugment_search_space.SEGMENTATION_OPS_COUNT
  return tunable_autoaugment_search_space.autoaugment_search_space(
      total_num_ops=total_num_ops, num_ops_per_sub_policy=2, num_sub_policies=5)



def spinenet_scaling_search_space():
  """Returns scaling search space for SpineNet."""
  return SpineNetScalingSpecBuilder(
      image_size=pg.one_of([512, 640]),
      endpoints_num_filters=pg.one_of([192, 256]),
      filter_size_scale=pg.one_of([0.8, 1.0, 1.2]),
      resample_alpha=pg.one_of([0.75, 1.0, 1.25]),
      block_repeats=pg.one_of([2, 3, 4]),
      head_num_convs=pg.one_of([4, 5]),
      head_num_filters=pg.one_of([192, 256]))
