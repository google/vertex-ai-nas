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

"""Main function to compute latency on cloud via docker using a saved-model.

Given a saved-model this file computes the average latency in seconds. This
file will be run via a docker on cloud. The latency-computation can be done
either on GPU or CPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import multiprocessing
import os
import timeit
from typing import Any, Dict, Text
from absl import logging

import cloud_nas_utils
import model_metrics_evaluator
import tf_utils
import numpy
import nvgpu
from PIL import Image
import tensorflow as tf
from tensorflow.experimental import tensorrt as trt


_GPU_TYPE = "GPU"
_RGB_IMAGE_TYPE = "RGB"
_JPEG_IMAGE_TYPE = "JPEG"
_IMAGE_PIXEL_TYPE = "uint8"
_NUM_IMAGE_CHANNELS = 3
_MAX_PIXEL_VAL = 255
_SEC_TO_MILLISEC = 1000


def export_tensorrt_model(
    saved_model_dir,
    tensorrt_model_dir,
    max_workspace_size_bytes = 2 << 20,
    precision_mode = "FP16"):
  """Exports TensorRT model."""
  params = trt.ConversionParams(
      max_workspace_size_bytes=max_workspace_size_bytes,
      precision_mode=precision_mode)
  converter = trt.Converter(
      input_saved_model_dir=saved_model_dir, conversion_params=params)
  converter.convert()
  converter.save(tensorrt_model_dir)


def run_tensorrt_conversion_in_separate_process(saved_model_path,
                                                tensorrt_model_path,
                                                precision_mode):
  """Use a separate process to run tensorrt conversion."""

  kwargs = {
      "saved_model_dir": saved_model_path,
      "tensorrt_model_dir": tensorrt_model_path,
      "max_workspace_size_bytes": 2 << 30,
      "precision_mode": precision_mode,
  }
  export_tensorrt_model_process = multiprocessing.Process(
      target=export_tensorrt_model, kwargs=kwargs)
  export_tensorrt_model_process.start()
  export_tensorrt_model_process.join()


def compute_model_metrics(saved_model_path, input_node,
                          image_width, image_height,
                          output_nodes,
                          pointpillars_input_spec,
                          device_type,
                          num_repetitions_warm_up, num_repetitions,
                          return_val_dict):
  """Computes average latency of a saved-model.

  Args:
    saved_model_path: Path to the saved-model directory.
    input_node: The name of the placeholder for the input image.
    image_width: The width of the image to be processed.
    image_height: The height of the image to be processed.
    output_nodes: Comma separated output-nodes to be evaluated.
    pointpillars_input_spec: Input spec for pointpillars model.
    device_type: One of the two device types - 'CPU' or 'GPU'.
    num_repetitions_warm_up:  Number of prediction repetitions to be done for
      warm-up before computing latency.
    num_repetitions: Number of repetitions to average for latency computation.
    return_val_dict: A shared-process dictionary which holds the returned output
      model-metrics. This function is used as a multiprocessing.Process
      `target`, so it cannot directly return a value back to the parent
      caller-process. Hence, this shared-variable is used instead to hold
      the returned value.
  """
  # For TF2 SavedModel, we do not need input and output nodes to run inference.
  del input_node
  del output_nodes

  # Check for correct device usage.
  if device_type == _GPU_TYPE:
    physical_devices = tf.config.experimental.list_physical_devices(_GPU_TYPE)
    assert physical_devices, "No GPUs found."
    # Set memory growth to True, so that the runtime initialization will not
    # allocate all memory on the device at the beginning at once.
    for device in physical_devices:
      tf.config.experimental.set_memory_growth(device, True)

  # Create input for 2D model.
  if image_height:
    random_array = numpy.random.rand(image_height, image_width,
                                     _NUM_IMAGE_CHANNELS) * _MAX_PIXEL_VAL
    pil_image = Image.fromarray(
        random_array.astype(_IMAGE_PIXEL_TYPE)).convert(_RGB_IMAGE_TYPE)
    buf = io.BytesIO()
    pil_image.save(buf, format=_JPEG_IMAGE_TYPE)
    encoded_image = buf.getvalue()

  # Create input for 3D model.
  elif pointpillars_input_spec:
    num_pillars, num_points, num_features, bev_height, bev_width = [
        int(x) for x in pointpillars_input_spec.split(",")
    ]
    pillars = tf.random.uniform(
        shape=[1, num_pillars, num_points, num_features],
        minval=0.0, maxval=1.0,
        dtype=tf.float32, name="pillars")
    indices = tf.random.uniform(
        shape=[1, num_pillars, 2],
        minval=0, maxval=min(bev_height, bev_width),
        dtype=tf.int32, name="indices")

  # Get memory used by tensorflow, the memory won't be counted into
  # model inference usage.
  default_tf_memory = nvgpu.gpu_info(
  )[0]["mem_used"] if device_type == _GPU_TYPE else 0
  print("Defult Tensorflow Memory: {}".format(default_tf_memory))

  # Run tf session for inference.
  device = "GPU:0" if device_type == _GPU_TYPE else "CPU:0"
  with tf.device(device):
    logging.info("Loading saved-model.")
    model = tf.saved_model.load(saved_model_path)
    predict_fn = model.signatures["serving_default"]
    logging.info("Running predictions.")

    if image_height:
      def _run_prediction():
        predict_fn(tf.constant([encoded_image]))
    elif pointpillars_input_spec:
      def _run_prediction():
        predict_fn(pillars=pillars, indices=indices)

    # Do some predictions for warm-up so that any set-up cost is not
    # included in the latency computation.
    if num_repetitions_warm_up > 0:
      avg_latency_in_millisecs_warm_up = timeit.timeit(
          _run_prediction, number=num_repetitions_warm_up) / float(
              num_repetitions_warm_up) * _SEC_TO_MILLISEC
      logging.info("Average prediction time during warm-up in millisecs is %f",
                   avg_latency_in_millisecs_warm_up)
    # Now run the actual latency computation.
    num_repetitions = max(num_repetitions, 1)
    avg_latency_in_millisecs = timeit.timeit(
        _run_prediction,
        number=num_repetitions) / float(num_repetitions) * _SEC_TO_MILLISEC
    logging.info("Average prediction time in millisecs is %f",
                 avg_latency_in_millisecs)

  activation_model_memory = nvgpu.gpu_info(
  )[0]["mem_used"] if device_type == _GPU_TYPE else 0
  model_memory = activation_model_memory - default_tf_memory
  model_metrics = {
      "latency_in_milliseconds": avg_latency_in_millisecs,
      "device_type": device_type,
      "model_memory": model_memory
  }
  logging.info("Model metrics are %s", model_metrics)
  return_val_dict["model_metrics"] = model_metrics


def compute_model_metrics_in_separate_process(
    saved_model_path, input_node, image_width, image_height,
    output_nodes,
    pointpillars_input_spec,
    device_type, num_repetitions_warm_up,
    num_repetitions):
  """Use a separate process to run latency estimation.

  We use a separate thread to clear the memory used by Tensorflow. Otherwise,
  Tensorflow will not clear memory and we cannot estimate memory consumption
  for different models.

  Args:
    saved_model_path: Path to the saved-model directory.
    input_node: The name of the placeholder for the input image.
    image_width: The width of the image to be processed.
    image_height: The height of the image to be processed.
    output_nodes: Comma separated output-nodes to be evaluated.
    pointpillars_input_spec: Input spec for pointpillars model.
    device_type: One of the two device types - 'CPU' or 'GPU'.
    num_repetitions_warm_up:  Number of prediction repetitions to be done for
      warm-up before computing latency.
    num_repetitions: Number of repetitions to average for latency computation.

  Returns:
    Model metrics such as latency and memory.
  """
  manager = multiprocessing.Manager()
  return_val_dict = manager.dict()
  kwargs = {
      "saved_model_path": saved_model_path,
      "input_node": input_node,
      "image_width": image_width,
      "image_height": image_height,
      "output_nodes": output_nodes,
      "pointpillars_input_spec": pointpillars_input_spec,
      "device_type": device_type,
      "num_repetitions_warm_up": num_repetitions_warm_up,
      "num_repetitions": num_repetitions,
      "return_val_dict": return_val_dict
  }
  write_model_metrics_process = multiprocessing.Process(
      target=compute_model_metrics, kwargs=kwargs)
  write_model_metrics_process.start()
  write_model_metrics_process.join()
  return return_val_dict["model_metrics"]


class CloudGpuMetricsEvaluator(model_metrics_evaluator.ModelMetricsEvaluator):
  """Computes latency and memory utilization for a NAS model on cloud GPU."""

  def __init__(self, service_endpoint, project_id, nas_job_id,
               use_tensorrt_conversion_on_gpu, input_node,
               image_width, image_height, output_nodes,
               pointpillars_input_spec,
               precision_mode,
               device_type, num_repetitions_warm_up,
               num_repetitions, latency_worker_id = 0,
               num_latency_workers = 1):
    super(CloudGpuMetricsEvaluator, self).__init__(
        service_endpoint=service_endpoint,
        project_id=project_id,
        nas_job_id=nas_job_id,
        latency_worker_id=latency_worker_id,
        num_latency_workers=num_latency_workers)
    self.use_tensorrt_conversion_on_gpu = use_tensorrt_conversion_on_gpu
    self.input_node = input_node
    self.image_width = image_width
    self.image_height = image_height
    self.output_nodes = output_nodes
    self.device_type = device_type
    self.num_repetitions_warm_up = num_repetitions_warm_up
    self.pointpillars_input_spec = pointpillars_input_spec
    self.precision_mode = precision_mode
    self.num_repetitions = num_repetitions

  def evaluate_saved_model(self, trial_id, saved_model_path):
    """Computes metrics for the saved-model."""
    model_path_to_use = saved_model_path
    if self.device_type == "GPU" and self.use_tensorrt_conversion_on_gpu:
      logging.info(
          "Initiating tensorrt-model conversion for saved-model: %s.",
          saved_model_path)
      tensorrt_model_path = os.path.join(self.job_output_dir, str(trial_id),
                                         "tensorrt_model")
      run_tensorrt_conversion_in_separate_process(
          saved_model_path=saved_model_path,
          tensorrt_model_path=tensorrt_model_path,
          precision_mode=self.precision_mode
      )
      logging.info("Exported tensorrt-model at %s", tensorrt_model_path)
      model_path_to_use = tensorrt_model_path

    logging.info("Calculate latency for %s", model_path_to_use)
    return compute_model_metrics_in_separate_process(
        saved_model_path=model_path_to_use,
        input_node=self.input_node,
        image_width=self.image_width,
        image_height=self.image_height,
        output_nodes=self.output_nodes,
        pointpillars_input_spec=self.pointpillars_input_spec,
        device_type=self.device_type,
        num_repetitions_warm_up=self.num_repetitions_warm_up,
        num_repetitions=self.num_repetitions)


def str_2_bool(v):
  """Auxiliary function to support boolean command-line arguments."""
  if not isinstance(v, str):
    raise ValueError("{} is not string type".format(v))
  return v.lower() == "true"


def create_arg_parser():
  """Creates arg parser."""
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      "--saved_model_path", type=str, help="Path to the saved-model directory.")
  parser.add_argument(
      "--input_node",
      type=str,
      help="The name of the input-node for the encoded-image string.")
  parser.add_argument(
      "--image_width", type=int, help="The width of the image to be processed.")
  parser.add_argument(
      "--image_height",
      type=int,
      help="The height of the image to be processed.")
  parser.add_argument(
      "--output_nodes",
      type=str,
      help="A list of the names of the output-nodes to be evaluated.")
  parser.add_argument(
      "--precision_mode",
      type=str,
      default="FP16",
      choices=["FP16", "FP32"],
      help="Precision mode.")
  parser.add_argument(
      "--pointpillars_input_spec",
      type=str,
      default="",
      help="5 integers separated by comma: num_pillars, num_points_per_pillar,"
           "num_features_per_point, image_height, image_width")
  parser.add_argument(
      "--device_type",
      type=str,
      default="GPU",
      choices=["GPU", "CPU"],
      help="Device type on which latency calculation will be run.")
  parser.add_argument(
      "--use_tensorrt_conversion_on_gpu",
      type=str_2_bool,
      default=False,
      help="True to apply tensorrt conversion to saved-model. "
      "This option applies only for a GPU device.")
  parser.add_argument(
      "--num_repetitions_warm_up",
      type=int,
      default=20,
      help="Number of repetitions for warm-up before computing latency.")
  parser.add_argument(
      "--num_repetitions_for_latency_computation",
      type=int,
      default=20,
      help="Number of repetitions for latency computation.")
  parser.add_argument(
      "--project_id",
      type=str,
      default="",
      help="The project ID to check for NAS job.")
  parser.add_argument(
      "--nas_job_id", type=str, default="", help="The NAS job id.")
  parser.add_argument(
      "--service_endpoint",
      type=str,
      default="https://ml.googleapis.com/v1",
      help="The end point of the service. Default is https://ml.googleapis.com/v1."
  )
  parser.add_argument(
      "--latency_worker_id",
      type=int,
      default=0,
      help="The id of this latency worker.")
  parser.add_argument(
      "--num_latency_workers",
      type=int,
      default=1,
      help="The total number of the latency workers.")
  return parser


def main(FLAGS):  

  logging.info("Starting latency calculator job.")
  if FLAGS.device_type != "GPU" and FLAGS.use_tensorrt_conversion_on_gpu:
    raise ValueError(
        "use_tensorrt_conversion_on_gpu is only supported for GPU device_type.")
  model_latency_evaluater = CloudGpuMetricsEvaluator(
      service_endpoint=FLAGS.service_endpoint,
      project_id=FLAGS.project_id,
      nas_job_id=FLAGS.nas_job_id,
      use_tensorrt_conversion_on_gpu=FLAGS.use_tensorrt_conversion_on_gpu,
      input_node=FLAGS.input_node,
      image_width=FLAGS.image_width,
      image_height=FLAGS.image_height,
      output_nodes=FLAGS.output_nodes,
      pointpillars_input_spec=FLAGS.pointpillars_input_spec,
      precision_mode=FLAGS.precision_mode,
      device_type=FLAGS.device_type,
      num_repetitions_warm_up=FLAGS.num_repetitions_warm_up,
      num_repetitions=FLAGS.num_repetitions_for_latency_computation,
      latency_worker_id=FLAGS.latency_worker_id,
      num_latency_workers=FLAGS.num_latency_workers)
  model_latency_evaluater.run_continuous_evaluation_loop()


if __name__ == "__main__":
  tf_utils.suppress_tf_logging()
  cloud_nas_utils.setup_logging()
  flags = create_arg_parser().parse_args()
  main(flags)
