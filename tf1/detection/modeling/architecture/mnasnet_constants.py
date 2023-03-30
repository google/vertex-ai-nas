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
"""MnasNet model constants."""

# NOTE: These constants are used for the tunable-mnasnet-search-space
# as well. The goal here is to keep these in a separate file to avoid
# dependency on any implementation specific details such as the use
# of TensorFlow.

MNASNET_NUM_BLOCKS = 7
MNASNET_STRIDES = [1, 2, 2, 2, 1, 2, 1]  # Same as MobileNet-V2.

# The fixed MnasNet-A1 architecture discovered by NAS.
# Each element represents a specification of a building block:
# (num_repeats, block_fn, expand_ratio, kernel_size, se_ratio, output_filters)
MNASNET_A1_BLOCK_SPECS = [
    (1, 'mbconv', 1, 3, 0.0, 16),
    (2, 'mbconv', 6, 3, 0.0, 24),
    (3, 'mbconv', 3, 5, 0.25, 40),
    (4, 'mbconv', 6, 3, 0.0, 80),
    (2, 'mbconv', 6, 3, 0.25, 112),
    (3, 'mbconv', 6, 5, 0.25, 160),
    (1, 'mbconv', 6, 3, 0.0, 320),
]
