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
"""Base config template that defines train, eval and backbones."""

from pytorch.classification import params_dict

# pylint: disable=line-too-long
_base_config_dict = {
    'runtime': {
        'accelerator': 'gpu',
        'data_type': 'float32',
        'strategy': 'ddp',
        'num_processes': None,
        'dataloader_num_workers': 0,
    },
    'type': 'classification',
    'architecture': {
        'backbone': 'tunable_mnasnet',
        # Note that `num_classes` is the total number of classes including one
        # background class whose index is 0.
        'num_classes': 1001,
    },
    'batch_norm_activation': {
        'batch_norm_momentum': 0.9,
        'batch_norm_epsilon': 1e-5,
        'batch_norm_trainable': True,
        'use_sync_bn': False,
        'activation': 'relu',
    },
    'tunable_mnasnet': {
        'block_specs': None,
    },
    'classification_head': {
        'endpoints_num_filters': 0,
        'aggregation': 'top',  # `top` or `all`.
        'dropout_rate': 0.0,
    },
    'train': {
        'data_path': None,
        'data_file_type': 'tar',
        'data_size': 1281167,
        'batch_size': 32,
        'num_epochs': 90,
        'random_erase': 0.0,
        'shard_shuffle_size': 8,
        'sample_shuffle_size': 256,
        'optimizer': {
            'type': 'SGD',
            'learning_rate': 0.1,
            'momentum': 0.9,
            'weight_decay': 0.0001,
        },
        'lr_scheduler': {
            'type': 'cosine',
            'min_lr': 0,
            'warmup_epochs': 0,
            'warmup_decay': 0.01,
        },
    },
    'eval': {
        'data_path': None,
        'data_file_type': 'tar',
        'data_size': 5000,
        'batch_size': 32,
    },
    'predict': {
        'predict_batch_size': 8,
    },
    'checkpoint': {
        'load_path': None,
        'save_file_prefix': 'checkpoint',
    },
}

BASE_CONFIG = params_dict.ParamsDict(_base_config_dict)
# pylint: enable=line-too-long
