# NAS codes release history

## Latest

### Major change
- Merged tensorflow and pytorch code repos into one.
- Removed ai-platform specific code.
- Restructured folders, moved tf/pytorch specific files to their own folders.
- Added apache-2 licenses to some of the files missing headers.

### Tutorials update
- Replaced tf.io.gfile dependency in tutorials with gcsfuse.
- Added information about workarounds for accessing GCS data during local runs.
- Added an explanation of third-party I/O `gcs_utils` wrapper libraries.

## Release on 01/08/2023

### Major change
- Vertex NAS now requires regional artifact registry instead of
  google-container-registry to prevent excess docker pull charges.
  The environment setup page is now updated to provide more instructions.
  Apart from environment set up, the 'build' and 'search_in_local'
  command now require passing a region corresponding to the artifact
  registry.

### Dependency change
- Removed fsspec dependency for GCS interaction inside dockers.
  Using google-cloud-storage library instead.

### Functionalities update
- Allow resumption of previous model selection job.

### Issues Fixed
- Fixed possible random crash of Proxy-task search controller due to
  "missing file" error during read even when the file is present.

## Release on 12/07/2022

### Functionalities update
- Added tunable PointPillars model based on TF-vision and search space.

### Tutorials update
- Added notebooks for tunable PointPillars.

## Release on 11/17/2022

### Dependency change
- Nas-client now requires 'google-cloud-storage'. Run
  "pip install --upgrade google-cloud-storage" in a virtual environment.
- Pyglove code is no longer shipped. Instead use "pip install pyglove==0.1.0"
  inside your dockerfiles and before running nas-client.
  More information available at 'Set up your environment' page
  in Vertex NAS documentation.

### Issues Fixed
- Fixed TF-vision trainer to skip training and evaluation for checkpointed
  steps after restarted.
- Remove experimental configs from tf_vision folder.
- Fixed protobuf library version for pre-built trainers.

### Functionalities update
- Added proxy-task design tools to help find the optimal proxy task
  for Vertex NAS. See 'Proxy task design' page in Vertex NAS documentation.
- Multiple latency workers now operate as a worker pool where a free
  latency worker can pick any trial.
- Added a function to get trial id in `cloud_nas_utils.py`.
- Added ability to mount local directory for custom training data when running
  local search with docker.
- Added ability to run multi-machine mirrored distributed training using
  nas-client.
- Added ability to access GCS buckets from locally running dockers.
- Upgraded pytorch mnasnet example for efficient distributed training.
- Added pytorch mnasnet search and finetune configs.
- Added scripts for sharding imagenet data.

### Tutorials update
- Added GCS-Fuse usage to tutorial1.
- Added clarification for log retention to tutorial1.
- Added information for accessing GCS from locally running docker to tutorial1.
- Updated pytorch mnasnet notebooks.
- Updated the instructions to not report an invalid reward.
- Added workaround for Cloud Shell tensorboard command error in tutorial1.

## Release on 08/02/2022

### Issues Fixed
- Updated dockerfiles to prevent NVIDIA key issues.
- Job info for a cancelled NAS job can be queried now.
- Pre-built model latency compute now uses the model input size instead of
  a fixed image size.
- The TPU related config-files are tagged with postfix "_tpu.yaml".
- Vertex nas-client parser code moved to "vertex_nas_client_parser.py".
- Some of the Vertex nas-client functions moved to "vertex_client_utils.py".
- Fixed threshold on invalid search space values in search space tutorial.

### Functionalities update
- Make the job timeout customizable, and increase default from 7 days to 14 days.
- Make nas-client be able to use A2 machines and A100 GPUs.
- Remove restriction of master machine type choices.
- Allow nas-client to run local binary without building and running docker.
- Added search resumption capability to Vertex NAS.

### Tutorials update
- Updated latency docker flags instruction in the section "Add device specific latency constraint" for tutorial 4.
- Append `--load_fast=false` to the tensorboard command in tutorial1 to avoid 401 error.
- Updated variable length model search space in tutorial2.
- Added gcsfuse issue to troubleshooting doc.

## Release on 04/05/2022

### Vertex AI NAS private GA launch
- Added [product documentation](https://cloud.google.com/vertex-ai/docs/neural-architecture-search).
- Supported [8 regions](https://cloud.google.com/vertex-ai/docs/neural-architecture-search/pricing).

### Functionalities update
- Enabled latency calculation in `vertex_nas_cli.py` for cloud retrain job.
- Replaced the flag `--latency_calculator_config` with `--latency_docker_flags` in `vertex_nas_cli.py` for passing flags to latency calculator container.
- Added a new command `run_latency_calculator_local` in `vertex_nas_cli.py` for running latency calculator container locally.
- Added [support](https://source.cloud.google.com/cloud-nas-260507/nas-codes-release/+/master:tutorial/vertex_tutorial4.md) for running multiple local latency calculator docker on multiple hardware devices.

### Issues fix
- Removed unused codes in `vertex_client_utils.py`.

### Tutorials update
- Added MIT license for tutorials.
- Fixed stage2 failure for vertex_tutorial4.
- Removed stage2 from vertex_nas_classification.ipynb.
