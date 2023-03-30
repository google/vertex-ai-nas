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
"""Classification input and model functions for serving/inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from tf1.detection.dataloader import mode_keys
from tf1.detection.modeling import factory
from tf1.detection.serving import inputs
from tf1.hyperparameters import params_dict


def serving_input_fn(batch_size,
                     desired_image_size,
                     stride,
                     input_type,
                     input_name='input'):
  """Input function for SavedModels and TF serving.

  Args:
    batch_size: The batch size.
    desired_image_size: The tuple/list of two integers, specifying the desired
      image size.
    stride: an integer, the stride of the backbone network. The processed image
      will be (internally) padded such that each side is the multiple of this
      number.
    input_type: a string of 'image_tensor', 'image_bytes' or 'tf_example',
      specifying which type of input will be used in serving.
    input_name: a string to specify the name of the input signature.

  Returns:
    a `tf.estimator.export.ServingInputReceiver` for a SavedModel.
  """
  placeholder, features = inputs.build_serving_input(input_type, batch_size,
                                                     desired_image_size, stride)
  return tf_estimator.export.ServingInputReceiver(
      features=features, receiver_tensors={
          input_name: placeholder,
      })


def build_predictions(features, params, output_image_info):
  """Builds the predictions for serving.

  Args:
    features: features to be passed to the serving model graph
    params: hyperparameters to be passed to the serving model graph
    output_image_info: bool, whether output the image_info node.

  Returns:
    predictions: model outputs for serving.
  """
  model_fn = factory.model_generator(params)
  outputs = model_fn.build_outputs(
      features['images'], labels=None, mode=mode_keys.PREDICT)

  logits = outputs['logits']
  predictions = {
      'class_id': tf.argmax(logits, axis=1, name='ClassId'),
      'probabilities': tf.nn.softmax(logits, name='Probabilities')
  }

  if output_image_info:
    predictions['image_info'] = tf.identity(
        features['image_info'], name='ImageInfo')
  return predictions


def serving_model_fn_builder(export_tpu_model, output_image_info):
  """Serving model_fn builder.

  Args:
    export_tpu_model: bool, whether to export a TPU or CPU/GPU model.
    output_image_info: bool, whether output the image_info node.

  Returns:
    A function that returns (TPU)EstimatorSpec for PREDICT mode.
  """

  def _serving_model_fn(features, labels, mode, params):
    """Builds the serving model_fn."""
    del labels  # unused.
    if mode != tf_estimator.ModeKeys.PREDICT:
      raise ValueError('To build the serving model_fn, set '
                       'mode = `tf.estimator.ModeKeys.PREDICT`')

    model_params = params_dict.ParamsDict(params)
    predictions = build_predictions(
        features=features,
        params=model_params,
        output_image_info=output_image_info)

    if export_tpu_model:
      return tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions)
    return tf_estimator.EstimatorSpec(mode=mode, predictions=predictions)

  return _serving_model_fn
