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

"""Utilities to create configs for training and evaluation."""
import os
import pprint
from typing import Any, Optional

from absl import logging
from tf_vision.configs import factory
import pyglove as pg
import tensorflow as tf
import yaml

from official.core import config_definitions as cfg
from official.modeling import hyperparams


def create_pointpillars_params(
    params,
    args,
    serialized_tunable_object,
    tpu_address = None):
  """Create configs for pointpillars."""
  if args.use_tpu:
    runtime = {
        "tpu": tpu_address,
    }
  else:
    gpus = tf.config.list_physical_devices("GPU")
    logging.info("There are %d GPUs.", len(gpus))
    runtime = {
        "distribution_strategy": "mirrored",
        "num_gpus": len(gpus),
        "all_reduce_alg": "hierarchical_copy",
    }
  params.override({
      "runtime": runtime,
      "task": {
          "model": {
              "block_specs_json": serialized_tunable_object,
          },
          "train_data": {
              "input_path": args.training_data_path,
          },
          "validation_data": {
              "input_path": args.validation_data_path,
          },
      },
  }, is_strict=False)
  params.validate()
  params.lock()
  pp = pprint.PrettyPrinter()
  params_str = pp.pformat(params.as_dict())
  logging.info("Model Parameters: %s", params_str)
  return params


def create_params(args,
                  search_space,
                  serialized_tunable_object,
                  tpu_address = None):
  """Returns `ParamsDict` based on `args` and `serialized_tunable_object`."""

  # Load default parameters based on model type.
  params = factory.config_generator(args.model)

  # Override default parameters using input config file.
  if args.config_file:
    params = hyperparams.override_params_dict(
        params, args.config_file, is_strict=True)

  # Override parameters using inline config string.
  params = hyperparams.override_params_dict(
      params, args.params_override, is_strict=True)

  if args.model.startswith("pointpillars"):
    params = create_pointpillars_params(
        params, args, serialized_tunable_object, tpu_address)
    return params

  # Override model backbones and decoders.
  if search_space == "nasfpn":
    params.override(
        {
            "task": {
                "model": {
                    "decoder": {
                        "type": "tunable_nasfpn",
                        "tunable_nasfpn": {
                            "block_specs_json": serialized_tunable_object,
                        }
                    }
                }
            },
        },
        is_strict=True)
  elif search_space == "mnasnet":
    params.override(
        {
            "task": {
                "model": {
                    "backbone": {
                        "type": "tunable_mnasnet",
                        "tunable_mnasnet": {
                            "block_specs_json": serialized_tunable_object,
                        }
                    }
                }
            },
        },
        is_strict=True)
  elif search_space in ["spinenet", "spinenet_v2"]:
    params.override(
        {
            "task": {
                "model": {
                    "backbone": {
                        "type": "tunable_spinenet",
                        "tunable_spinenet": {
                            "block_specs_json": serialized_tunable_object,
                        }
                    },
                    "decoder": {
                        "type": "identity"
                    },
                }
            },
        },
        is_strict=True)
  elif search_space == "spinenet_mbconv":
    params.override({
        "task": {
            "model": {
                "backbone": {
                    "type": "tunable_spinenet_mbconv",
                    "tunable_spinenet_mbconv": {
                        "block_specs_json": serialized_tunable_object,
                    }
                },
                "decoder": {
                    "type": "identity"
                },
            }
        },
    }, is_strict=True)
  elif search_space in ["randaugment_detection", "autoaugment_detection"]:
    params.override(
        {
            "task": {
                "train_data": {
                    "parser": {
                        "aug_policy": serialized_tunable_object,
                    },
                },
            },
        },
        is_strict=True)
  elif search_space in ["randaugment_segmentation", "autoaugment_segmentation"]:
    params.override(
        {
            "task": {
                "train_data": {
                    "aug_policy": serialized_tunable_object,
                },
            },
        },
        is_strict=True)
  elif search_space == "spinenet_scaling":
    scaling_config = pg.from_json_str(serialized_tunable_object)
    scaling_model_spec_params = {
        "task": {
            "model": {
                "backbone": {
                    "type": "tunable_spinenet",
                    "tunable_spinenet": {
                        "endpoints_num_filters":
                            scaling_config.endpoints_num_filters,
                        "resample_alpha":
                            scaling_config.resample_alpha,
                        "block_repeats":
                            scaling_config.block_repeats,
                        "filter_size_scale":
                            scaling_config.filter_size_scale,
                    }
                },
                "decoder": {
                    "type": "identity"
                },
                "head": {
                    "num_convs": scaling_config.head_num_convs,
                    "num_filters": scaling_config.head_num_filters,
                },
                "input_size": [
                    scaling_config.image_size, scaling_config.image_size, 3
                ],
            }
        },
    }

    # Write scaling parameters as yaml to job_dir, so that it can be reused
    # later (with --params_override flag).
    scaling_params_file = os.path.join(args.job_dir, "scaling_params.yaml")
    with tf.io.gfile.GFile(scaling_params_file, "w") as f:
      yaml.dump(scaling_model_spec_params, f)

    params.override(scaling_model_spec_params, is_strict=True)
  else:
    raise ValueError(f"Unexpected search_space {search_space}.")

  # Use TPU by default.
  params.override(
      {
          "runtime": {
              "tpu": tpu_address,
              "distribution_strategy": "tpu",
              "mixed_precision_dtype": "bfloat16",
          },
          "task": {
              "model": {
                  "norm_activation": {
                      # Use sync batchnorm for TPU as default.
                      "use_sync_bn": True,
                  }
              },
              "train_data": {
                  "dtype": "bfloat16",
                  "input_path": args.training_data_path,
                  "decoder": {
                      "type": "simple_decoder",
                      "simple_decoder": {
                          # To ensure that source_id is integer format for TPU.
                          "regenerate_source_id": True
                      }
                  },
              },
              "validation_data": {
                  "dtype": "bfloat16",
                  "input_path": args.validation_data_path,
                  "decoder": {
                      "type": "simple_decoder",
                      "simple_decoder": {
                          # To ensure that source_id is integer format for TPU.
                          "regenerate_source_id": False
                      }
                  },
              },
          },
      },
      is_strict=True)

  if not args.use_tpu:
    gpus = tf.config.list_physical_devices("GPU")
    logging.info("There are %d GPUs.", len(gpus))
    params.override(
        {
            "runtime": {
                "distribution_strategy": "mirrored",
                "mixed_precision_dtype": "float16",
                "loss_scale": "dynamic",
                "num_gpus": len(gpus),
                # TODO: Figure out why NCCL would cause hanging
                # issue. In the mean time use hierarchical_copy as a workaround.
                "all_reduce_alg": "hierarchical_copy",
            },
            "task": {
                "model": {
                    "norm_activation": {
                        "use_sync_bn": False,
                    }
                },
                "train_data": {
                    "dtype": "float16"
                },
                "validation_data": {
                    "dtype": "float16"
                },
            }
        },
        is_strict=True)

  params.validate()
  params.lock()
  pp = pprint.PrettyPrinter()
  params_str = pp.pformat(params.as_dict())
  logging.info("Model Parameters: %s", params_str)
  return params
