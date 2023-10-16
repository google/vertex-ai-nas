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
# ==============================================================================
"""Parser for nas-client."""

import argparse


def str_2_bool(v):
  """Auxiliary function to support boolean command-line arguments."""
  if not isinstance(v, str):
    raise ValueError("{} is not string type".format(v))
  return v.lower() == "true"


def add_project_id_parser(parser):
  """Configure the project_id parser."""
  parser.add_argument(
      "--project_id",
      type=str,
      required=False,
      default="",
      help="The GCP project ID.")


def add_service_account_parser(parser):
  """Configure the service_account parser."""
  parser.add_argument(
      "--service_account",
      type=str,
      required=False,
      default="",
      help="The email address to a service account.")


def add_region_parser(parser):
  """Configure the region and service-endpoint parser."""
  parser.add_argument(
      "--service_endpoint",
      type=str,
      default="PROD",
      help="Vertex AI service endpoint.")
  parser.add_argument(
      "--region",
      type=str,
      default="us-central1",
      choices=[
          "asia-east1", "asia-southeast1", "europe-west1", "europe-west4",
          "us-central1", "us-east1", "us-east4", "us-west1"
      ],
      help="The cloud region (https://cloud.google.com/about/locations) in "
      "which the training jobs will be run.")


def add_latency_docker_id_parser(parser):
  parser.add_argument(
      "--latency_calculator_docker_id",
      type=str,
      required=False,
      default="",
      help="The ID of the latency/memory calculation docker.")


def add_docker_id_parser(parser):
  parser.add_argument(
      "--trainer_docker_id",
      type=str,
      required=False,
      default="",
      help="The ID of the training docker.")
  add_latency_docker_id_parser(parser)


def add_docker_file_parser(parser):
  """Configure the parser for building containers."""
  parser.add_argument(
      "--trainer_docker_file",
      type=str,
      required=False,
      default="tf_vision/nas_multi_trial.Dockerfile",
      help="The Docker file to build the training docker.")
  parser.add_argument(
      "--latency_calculator_docker_file",
      type=str,
      required=False,
      default="tf_vision/latency_computation_using_saved_model.Dockerfile",
      help="The Docker file to build the latency/memory calculation docker.")

  parser.add_argument(
      "--proxy_task_model_selection_docker_file",
      type=str,
      required=False,
      default="proxy_task/proxy_task_model_selection.Dockerfile",
      help="The Docker file to build the proxy task model selection docker.")
  parser.add_argument(
      "--proxy_task_search_controller_docker_file",
      type=str,
      required=False,
      default="proxy_task/proxy_task_search_controller.Dockerfile",
      help="The Docker file to build the proxy task model selection docker.")
  parser.add_argument(
      "--proxy_task_variance_measurement_docker_file",
      type=str,
      required=False,
      default="proxy_task/proxy_task_variance_measurement.Dockerfile",
      help="The Docker file to build the proxy task variance docker.")


def add_cloud_search_parser(parser):
  """Configure the parser for running NAS on cloud."""
  parser.add_argument(
      "--job_name",
      type=str,
      required=True,
      help="The display name for Cloud NAS job.")
  parser.add_argument(
      "--max_nas_trial",
      required=False,
      type=int,
      default=5,
      help="The maximum number of trials (architectures) to search.")
  parser.add_argument(
      "--max_parallel_nas_trial",
      required=False,
      type=int,
      default=5,
      help="The maximum number of parallel trials (architectures) to search.")
  parser.add_argument(
      "--max_failed_nas_trial",
      type=int,
      default=5,
      help="The number of failed trials after which the NAS job will be failed."
  )
  parser.add_argument(
      "--train_max_parallel_trial",
      type=int,
      default=5,
      help="The number of parallel train trials to be launched.")
  parser.add_argument(
      "--train_frequency",
      type=int,
      default=1000,
      help="The regular interval at which the train trials are launched. "
      "For example, if the value is set to 1000, then after every 1000 "
      "search-trials 'train_max_parallel_trial' number of train "
      "trials will be launched.")


def add_search_docker_flags_parser(parser):
  parser.add_argument(
      "--search_docker_flags",
      type=str,
      nargs="*",
      default=[],
      help="Flags passed to the docker for search. The flags can "
      "be specified as: "
      " '--search_docker_flags flag1=val1 flag2=val2 flag3=val3'. "
      "You can pass variable number of flags.")


def add_train_docker_flags_parser(parser):
  parser.add_argument(
      "--train_docker_flags",
      type=str,
      nargs="*",
      default=None,
      help="Flags passed to the docker for train. The flags can "
      "be specified as: "
      " '--train_docker_flags flag1=val1 flag2=val2 flag3=val3'. "
      "You can pass variable number of flags.")


def add_gcs_parser(parser):
  parser.add_argument(
      "--root_output_dir",
      required=True,
      default="",
      help="Cloud job root output directory. The job output dir is "
      "`root_output_dir/job_id` for search/train jobs.")


def add_target_reward_metric_parser(parser):
  parser.add_argument(
      "--nas_target_reward_metric",
      required=False,
      default="",
      help="The metric name of NAS reward reported by trainer. It will be set "
      "as the environment `CLOUD_ML_HP_METRIC_TAG` in the training docker.")


def add_search_job_cloud_resource_parser(parser):
  """Configure cloud-resource for the search-job."""
  # Machine configuration options for the search-trials.
  parser.add_argument(
      "--accelerator_type",
      type=str,
      default="NVIDIA_TESLA_V100",
      choices=[
          "", "NVIDIA_TESLA_A100", "NVIDIA_TESLA_K80", "NVIDIA_TESLA_P4",
          "NVIDIA_TESLA_P100", "NVIDIA_TESLA_T4", "NVIDIA_TESLA_V100"
      ],
      help="The accelerator used for search. "
      "NOTE: The project must be allocated sufficient quota before you can use "
      "the desired type amd number of accelerators."
      "Empty string means only CPU machine is used. "
      "By default TPU will use TPU type (v2). ")
  parser.add_argument(
      "--num_gpus",
      type=int,
      default=2,
      help="Only used if 'accelerator_type' is set to GPU."
      "NOTE: This setting is per-machine.")
  parser.add_argument(
      "--master_machine_type",
      type=str,
      default="n1-highmem-16",
      help="Master machine CPU configuration on cloud for search-trials. "
      "Please visit "
      "https://cloud.google.com/vertex-ai/docs/training/configure-compute#machine-types"
      " to see what GPU configurations are supported by each"
      " master-machine-type.")
  parser.add_argument(
      "--num_mirror_machines",
      type=int,
      default=1,
      help="Total number of mirror machines with the same configuration "
      "as above. When > 1, a multi-machine mirrored distributed training "
      "strategy is assumed.  Value of 1 assumes regular training.")


def add_train_job_cloud_resource_parser(parser):
  """Configure cloud-resource for the train-job."""
  # Machine configuration options for the train-trials.
  parser.add_argument(
      "--train_accelerator_type",
      type=str,
      default="NVIDIA_TESLA_V100",
      choices=[
          "", "NVIDIA_TESLA_A100", "NVIDIA_TESLA_K80", "NVIDIA_TESLA_P4",
          "NVIDIA_TESLA_P100", "NVIDIA_TESLA_T4", "NVIDIA_TESLA_V100"
      ],
      help="The accelerator used for train. "
      "NOTE: The project must be allocated sufficient quota before you can use "
      "the desired type amd number of accelerators."
      "Empty string means only CPU machine is used. "
      "By default TPU will use TPU type (v2). ")
  parser.add_argument(
      "--train_num_gpus",
      type=int,
      default=2,
      help="Only used if 'train_accelerator_type' is set to GPU."
      "NOTE: This setting is per-machine.")
  parser.add_argument(
      "--train_master_machine_type",
      type=str,
      default="n1-highmem-16",
      help="Master machine CPU configuration on cloud for train-trials. "
      "Please visit "
      "https://cloud.google.com/vertex-ai/docs/training/configure-compute#machine-types"
      " to see what GPU configurations are supported by each"
      " master-machine-type.")
  parser.add_argument(
      "--train_num_mirror_machines",
      type=int,
      default=1,
      help="Total number of mirror machines with the same configuration "
      "as above. When > 1, a multi-machine mirrored distributed training "
      "strategy is assumed. Value of 1 assumes regular training.")


def add_latency_parser(parser):
  """Configure latency related flags."""
  parser.add_argument(
      "--latency_docker_flags",
      type=str,
      nargs="*",
      default=[],
      help="Flags passed to the docker for latency calculation. The flags can "
      "be specified as: "
      " '--latency_docker_flags flag1=val1 flag2=val2 flag3=val3'. "
      "You can pass variable number of flags.")
  parser.add_argument(
      "--target_device_type",
      type=str,
      default="NVIDIA_TESLA_V100",
      choices=[
          "CPU", "NVIDIA_TESLA_K80", "NVIDIA_TESLA_P4", "NVIDIA_TESLA_P100",
          "NVIDIA_TESLA_T4", "NVIDIA_TESLA_V100"
      ],
      help="The type of target device used for latency/memory calculation. A "
      "separate job with the specified targeting device will be launched to "
      "evaluate the models generated during search.")
  parser.add_argument(
      "--use_prebuilt_latency_calculator",
      type=str_2_bool,
      default=False,
      help="Set to True to use pre-built latency-calculator.")
  parser.add_argument(
      "--prebuilt_latency_image_width",
      type=int,
      required=False,
      default=512,
      help="Width of testing image. Only used if "
      "use_prebuilt_latency_calculator is set to true.")
  parser.add_argument(
      "--prebuilt_latency_image_height",
      type=int,
      required=False,
      default=512,
      help="Height of testing image. Only used if "
      "use_prebuilt_latency_calculator is set to true.")
  parser.add_argument(
      "--prebuilt_latency_input_node",
      type=str,
      required=False,
      default="Placeholder:0",
      help="The Tensorflow SavedModel input_node. Only used if "
      "use_prebuilt_latency_calculator is set to true.")
  parser.add_argument(
      "--prebuilt_latency_output_nodes",
      type=str,
      required=False,
      default="DetectionBoxes:0,DetectionClasses:0,DetectionScores:0",
      help="The Tensorflow SavedModel output_mode. Only used if "
      "use_prebuilt_latency_calculator is set to true and prebuilt model is not"
      " `retinanet` nor `classification`.")
  parser.add_argument(
      "--prebuilt_num_repetitions_for_latency_computation",
      type=int,
      required=False,
      default=100,
      help="The number of repetitions to run for latency calculation. Only used"
      " if use_prebuilt_latency_calculator is set to true.")
  parser.add_argument(
      "--prebuilt_latency_tensorrt_conversion_on_gpu",
      type=str_2_bool,
      required=False,
      default=True,
      help="Whether to use TensorRT conversion for Nvidia GPU. Only used if "
      "use_prebuilt_latency_calculator is set to true.")
  parser.add_argument(
      "--num_cloud_latency_workers",
      type=int,
      default=1,
      required=False,
      help="The total number of parallel latency calculator workers on Google "
      "Cloud. latency_worker_id and num_latency_workers are set automatically "
      "based on this flag value.")


def add_search_space_parser(parser):
  """Configure search_space related flags."""
  parser.add_argument(
      "--use_prebuilt_trainer",
      type=str_2_bool,
      default=False,
      help="True to use NAS prebuilt trainers which can be built using "
      "`Dockerfile` in the root dir of NAS codes.")

  parser.add_argument(
      "--prebuilt_search_space",
      type=str,
      default="",
      help="The NAS prebuilt search spaces like nasfpn, spinenet. They are "
      "defined in search_spaces.py in the NAS codes.")
  parser.add_argument(
      "--search_space_module",
      type=str,
      default="",
      help="The python module that defines search space, e.g., "
      "`tutorial.tutorial3_mnist_search.get_search_space` "
      "refers to the function `get_search_space()` in file '/tutorial/tutorial3_mnist_search.py'."
  )


def build_container_parser(base):
  """Configure the parser to build containers."""
  parser = base.add_parser(
      "build",
      description="Builds NAS containers and pushes to GCP artifact registry.")
  add_region_parser(parser)
  add_project_id_parser(parser)
  add_docker_id_parser(parser)
  add_docker_file_parser(parser)

  parser.add_argument(
      "--use_cache",
      type=str_2_bool,
      required=False,
      default=True,
      help="Whether to use local cache for docker build.")
  parser.add_argument(
      "--proxy_task_model_selection_docker_id",
      type=str,
      required=False,
      default="",
      help="The ID of the proxy-task model selection docker id.")
  parser.add_argument(
      "--proxy_task_search_controller_docker_id",
      type=str,
      required=False,
      default="",
      help="The ID of the proxy-task search controller docker id.")
  parser.add_argument(
      "--proxy_task_variance_measurement_docker_id",
      type=str,
      required=False,
      default="",
      help="The ID of the proxy task variance measurement docker id.")


def search_in_local_parser(base):
  """Configure the local run parser."""
  parser = base.add_parser(
      "search_in_local", description="Run NAS job locally.")
  add_region_parser(parser)
  add_project_id_parser(parser)
  add_docker_id_parser(parser)
  add_search_space_parser(parser)
  add_search_docker_flags_parser(parser)
  parser.add_argument(
      "--local_data_dir",
      required=False,
      default="",
      help="Local data input directory.")
  parser.add_argument(
      "--local_output_dir",
      required=False,
      default="",
      help="Local job output directory.")
  parser.add_argument(
      "--run_local_binary",
      required=False,
      default=False,
      help="True when want to run binary without building and running docker.")
  parser.add_argument(
      "--local_binary", required=False, default="", help="Local binary path.")
  parser.add_argument(
      "--local_binary_flags",
      type=str,
      nargs="*",
      default=[],
      help="Flags passed to the local binary for search. The flags can "
      "be specified as: "
      " '--local_binary_flags flag1=val1 flag2=val2 flag3=val3'. "
      "You can pass variable number of flags.")


def search_parser(base):
  """Configure the search parser."""
  parser = base.add_parser("search", description="Run NAS job on Cloud.")
  add_project_id_parser(parser)
  add_region_parser(parser)
  add_docker_id_parser(parser)
  add_search_space_parser(parser)
  add_target_reward_metric_parser(parser)
  add_search_docker_flags_parser(parser)
  add_cloud_search_parser(parser)
  add_gcs_parser(parser)
  add_search_job_cloud_resource_parser(parser)
  add_train_job_cloud_resource_parser(parser)
  add_train_docker_flags_parser(parser)
  add_latency_parser(parser)


def search_resume_parser(base):
  """Configure the search resume parser."""
  parser = base.add_parser(
      "search_resume", description="Resume NAS job on Cloud.")
  add_project_id_parser(parser)
  add_region_parser(parser)
  add_cloud_search_parser(parser)
  add_gcs_parser(parser)
  parser.add_argument(
      "--previous_nas_job_id",
      type=str,
      required=True,
      help="The previous NAS job ID to resume search from. "
      "NOTE: The NAS job ID is numeric only.")
  parser.add_argument(
      "--previous_latency_job_id",
      type=str,
      required=False,
      default="",
      help="The previous cloud latency job ID corresponding to the "
      "previous search job. "
      "NOTE: The job ID is numeric only.")


def train_parser(base):
  """Configure the train (stage-2) only job."""
  parser = base.add_parser(
      "train",
      description="Train searched model architecture (trial) in Cloud.")
  add_project_id_parser(parser)
  add_region_parser(parser)
  add_docker_id_parser(parser)
  add_search_space_parser(parser)
  add_target_reward_metric_parser(parser)
  add_gcs_parser(parser)
  add_train_job_cloud_resource_parser(parser)
  add_train_docker_flags_parser(parser)
  add_latency_parser(parser)
  parser.add_argument(
      "--search_job_id",
      type=int,
      required=True,
      help="The numeric job-id for the previous search job.")
  parser.add_argument(
      "--search_job_region",
      type=str,
      required=True,
      help="The region for the previous search job.")
  parser.add_argument(
      "--train_nas_trial_numbers",
      type=str,
      help="A list of trial numbers of `search_job_id` for stage2 "
      "training, separated by comma.")
  parser.add_argument(
      "--train_job_suffix",
      type=str,
      required=True,
      default="",
      help="The suffix of this train job. E.g., for search job-id "
      "`1234` and `train_job_suffix=trial_1`, the launched "
      "train job name will have prefix `search_1234_trial_1`.")


def list_trials_parser(base):
  """Configure the list_trials parser."""
  parser = base.add_parser(
      "list_trials", description="List NAS trials for specific job.")
  add_project_id_parser(parser)
  add_region_parser(parser)
  parser.add_argument(
      "--job_id",
      type=int,
      required=True,
      help="The nas job id generated by Vertex AI, in the format of unique "
      "numeric number.")
  parser.add_argument(
      "--trials_output_file",
      type=str,
      required=False,
      default="",
      help="The output file path to write trial details. If not specified, "
      "trial details will be written to stdout.")
  parser.add_argument(
      "--max_trials",
      type=int,
      required=False,
      default="20",
      help="The max number of trials in the output. If set to 0, will output "
      "all trials.")


def run_latency_calculator_local_parser(base):
  """Configure the run_latency_calculator_local command parser."""
  parser = base.add_parser(
      "run_latency_calculator_local",
      description="Run local latency calculation job.")
  add_project_id_parser(parser)
  add_region_parser(parser)
  add_latency_docker_id_parser(parser)
  add_latency_parser(parser)
  parser.add_argument(
      "--search_job_id",
      type=int,
      required=False,
      help="The nas search job id generated by Vertex AI, in the format of "
      "unique numeric number.")

  parser.add_argument(
      "--controller_job_id",
      type=int,
      required=False,
      help="The controller-job (proxy-task model selection job "
      "or proxy-task search controller job) id generated by "
      "Vertex AI, in the format of unique numeric number.")

  parser.add_argument(
      "--local_output_dir",
      required=False,
      default="",
      help="Local job output directory. The latency container will run for a "
      "long time, and this path can be used to get outputs from the "
      "latency container. Please refer to _LOCAL_RUN_OUTPUT_DIR variable "
      "for the corresponding path inside the container.")
  parser.add_argument(
      "--prebuilt_trainer_model",
      type=str,
      default="",
      required=False,
      choices=["retinanet", "segmentation", "classification", "mask_rcnn"],
      help="Which prebuilt trainer model to use. Only used if "
      "use_prebuilt_latency_calculator is set to true.")
  parser.add_argument(
      "--latency_worker_id",
      type=int,
      default=0,
      required=False,
      help="Which latency calculation worker to start. Should be an integer in "
      "[0, num_latency_workers - 1]. If num_latency_workers > 1, each worker "
      "will only handle a subset of the parallel training trials based on "
      "their trial-ids.")
  parser.add_argument(
      "--num_latency_workers",
      type=int,
      default=1,
      required=False,
      help="The total number of parallel latency calculator workers. If "
      "num_latency_workers > 1, it is used to select a subset of the parallel "
      "training trials based on their trial-ids.")


def select_proxy_task_models_parser(base):
  """Parser for proxy-task search command."""
  parser = base.add_parser(
      "select_proxy_task_models",
      description="Selects a sample of models from search space to be "
      "later used for proxy task correlation score computation.")
  parser.add_argument(
      "--proxy_task_model_selection_docker_id",
      type=str,
      required=True,
      default="",
      help="The ID of the proxy-task model selection docker id.")
  parser.add_argument(
      "--job_name",
      type=str,
      required=True,
      help="The display name for Cloud NAS job.")
  parser.add_argument(
      "--max_parallel_nas_trial",
      required=True,
      type=int,
      help="The maximum number of parallel trials (architectures) to run.")
  parser.add_argument(
      "--accuracy_metric_id",
      type=str,
      required=True,
      help="The accuracy metric-id based on which the models will be selected.")
  parser.add_argument(
      "--latency_metric_id",
      type=str,
      help="The latency metric-id based on which the models will be "
      "selected. This is not required if the training job does not "
      "compute latency.")
  parser.add_argument(
      "--previous_model_selection_dir",
      type=str,
      required=False,
      default="",
      help="Set this to a previously run model selection job directory "
      "to resume a previous model-selection job if required. "
      "You will need to delete the last "
      "partially run iteration job information from the "
      "'MODEL_SELECTION_STATE.json' file. If set, the "
      "'root_output_dir' will not be used.")
  add_project_id_parser(parser)
  add_service_account_parser(parser)
  add_region_parser(parser)
  add_docker_id_parser(parser)
  add_search_space_parser(parser)
  add_target_reward_metric_parser(parser)
  add_search_docker_flags_parser(parser)
  add_gcs_parser(parser)
  add_search_job_cloud_resource_parser(parser)
  add_latency_parser(parser)


def search_proxy_task_parser(base):
  """Parser for proxy-task search command."""
  parser = base.add_parser(
      "search_proxy_task",
      description="Launches a search job over "
      "a proxy-task search space.")
  parser.add_argument(
      "--proxy_task_search_controller_docker_id",
      type=str,
      required=True,
      default="",
      help="The ID of the proxy-task search-controller docker id.")
  parser.add_argument(
      "--job_name",
      type=str,
      required=True,
      help="The display name for Cloud NAS job.")
  parser.add_argument(
      "--proxy_task_config_generator_module",
      type=str,
      required=True,
      help="The proxy-task configuration generator module. When called, the "
      "module should return a list of proxy-task configurations to try.")
  parser.add_argument(
      "--proxy_task_model_selection_job_id",
      type=int,
      required=True,
      help="The numeric job-id for a previous "
      "proxy-task model selection job. This job provides the reference "
      "models for proxy-task evaluation. These models will be reused to "
      "launch multiple proxy-task-jobs, each corresponding to a new "
      "proxy-task-configuration that your generator provides.")
  parser.add_argument(
      "--proxy_task_model_selection_job_region",
      type=str,
      required=True,
      help="The region for the proxy-task model selection job.")
  parser.add_argument(
      "--desired_accuracy_correlation",
      type=float,
      default=0.65,
      help="If not 'None', the proxy-task will be stopped if its correlation "
      "crosses this limit. If 'None', then this check is not done. "
      "NOTE: Any of the checks for 'desired_accuracy_correlation', "
      "'desired_accuracy', and 'training_time_hrs_limit' "
      "can independently stop the proxy-task "
      "when triggered. For a more customized behavior, please modify "
      "proxy_task_search_controller_lib.has_met_stopping_condition function.")
  parser.add_argument(
      "--desired_accuracy",
      type=float,
      help="If not 'None', the proxy-task will be stopped if its accuracy "
      "crosses this limit. If 'None', then this check is not done. "
      "NOTE: Any of the checks for 'desired_accuracy_correlation', "
      "'desired_accuracy', and 'training_time_hrs_limit' "
      "can independently stop the proxy-task "
      "when triggered. For a more customized behavior, please modify "
      "proxy_task_search_controller_lib.has_met_stopping_condition function.")
  parser.add_argument(
      "--training_time_hrs_limit",
      type=float,
      default=3.0,
      required=True,
      help="The proxy-task will be stopped if its training-time "
      "crosses this limit. "
      "NOTE: Any of the checks for 'desired_accuracy_correlation', "
      "'desired_accuracy', and 'training_time_hrs_limit' "
      "can independently stop the proxy-task "
      "when triggered. For a more customized behavior, please modify "
      "proxy_task_search_controller_lib.has_met_stopping_condition function.")
  parser.add_argument(
      "--desired_latency_correlation",
      type=float,
      default=0.65,
      help="If not 'None', a proxy-task job is considered a failure if its "
      "latency correlation is less than this threshold.")
  parser.add_argument(
      "--early_stop_proxy_task_if_not_best",
      type=str_2_bool,
      default=True,
      help="If 'True', a proxy-task will be stopped early if its current-cost "
      "is found larger than the previous best proxy-task.")
  parser.add_argument(
      "--previous_proxy_task_search_dir",
      type=str,
      required=False,
      default="",
      help="Set this to a previously run proxy-task search directory "
      "to resume a previous search or compare additional proxy-tasks. If set, "
      "the 'root_output_dir' will not be used.")
  add_project_id_parser(parser)
  add_service_account_parser(parser)
  add_region_parser(parser)
  add_docker_id_parser(parser)
  add_search_space_parser(parser)
  add_target_reward_metric_parser(parser)
  add_gcs_parser(parser)
  add_search_job_cloud_resource_parser(parser)
  add_search_docker_flags_parser(parser)
  add_latency_parser(parser)


def measure_proxy_task_variance_parser(base):
  """Parser for measuring proxy task variance command."""
  parser = base.add_parser(
      "measure_proxy_task_variance",
      description="Measure variance and smoothness for the proxy task.")
  parser.add_argument(
      "--proxy_task_variance_measurement_docker_id",
      type=str,
      required=True,
      default="",
      help="The ID of the proxy task variance measurement docker id.")
  parser.add_argument(
      "--job_name",
      type=str,
      required=True,
      help="The display name for Cloud NAS job.")
  add_project_id_parser(parser)
  add_service_account_parser(parser)
  add_region_parser(parser)
  add_docker_id_parser(parser)
  add_search_space_parser(parser)
  add_target_reward_metric_parser(parser)
  add_search_docker_flags_parser(parser)
  add_gcs_parser(parser)
  add_search_job_cloud_resource_parser(parser)


def create_nas_cli_parser():
  """Returns parser for nas-client."""
  parser = argparse.ArgumentParser(description="Cloud AI NAS platform script.")

  subparser = parser.add_subparsers(dest="command")
  subparser.required = True

  build_container_parser(subparser)
  search_in_local_parser(subparser)
  search_parser(subparser)
  search_resume_parser(subparser)
  train_parser(subparser)
  list_trials_parser(subparser)
  run_latency_calculator_local_parser(subparser)

  search_proxy_task_parser(subparser)
  select_proxy_task_models_parser(subparser)
  measure_proxy_task_variance_parser(subparser)

  return parser
