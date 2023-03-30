# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Search space examples."""

import pyglove as pg

_KERNEL_SIZE = [3, 5, 7, 9]


def basic_one_of_search_space():
  return pg.Dict(kernel_size=pg.one_of(_KERNEL_SIZE))


def basic_sublist_of_search_space():
  return pg.sublist_of(
      2, _KERNEL_SIZE, choices_sorted=False, choices_distinct=True)


def mnist_list_of_dictionary_search_space():
  """Returns a list and dictionary based search space."""

  def _layer(base_filter_size):
    """Returns a layer specification."""
    return pg.Dict(
        kernel_size=pg.one_of([3, 5]),
        filter_size=pg.one_of(
            [int(base_filter_size * 0.75), base_filter_size,
             int(base_filter_size * 1.25)]))

  return pg.List([_layer(base_filter_size=4), _layer(base_filter_size=8)])


@pg.members([('kernel_size', pg.typing.Int()),
             ('filter_size', pg.typing.Int())])
class LayerSpec(pg.Object):
  """Specifies a model-layer."""


@pg.members([('layers', pg.typing.List(pg.typing.Object(LayerSpec),
                                       min_size=1))])
class ModelSpec(pg.Object):
  """Specifies a model."""


def mnist_class_search_space():
  """Returns a class-based mnist search space."""

  def _layer(base_filter_size):
    """Returns a layer specification."""
    return LayerSpec(
        kernel_size=pg.one_of([3, 5]),
        filter_size=pg.one_of(
            [int(base_filter_size * 0.75), base_filter_size,
             int(base_filter_size * 1.25)]))

  return ModelSpec(
      layers=[_layer(base_filter_size=4),
              _layer(base_filter_size=8)])


@pg.members([('current_layer_idx', pg.typing.Int()),
             ('previous_layers', pg.typing.List(pg.typing.Int()))])
class LayerConnectionSpec(pg.Object):
  """Specifies a choice of two previous layers as an input-combination for the current layer."""


@pg.members([('layer_specs',
              pg.typing.List(pg.typing.Object(LayerConnectionSpec)))])
class ModelConnectionSpec(pg.Object):
  """Specifies connection choices for all the layers."""


def architecture_connection_search_space():
  """Returns an architecture-connection search space."""
  def _build_layer_spec(current_layer_idx):
    # Pick any 2 distinct layer-indices from the previous layers.
    previous_layers = pg.sublist_of(
        2,
        list(range(current_layer_idx)),
        choices_distinct=True,
        choices_sorted=True)
    return LayerConnectionSpec(
        current_layer_idx=current_layer_idx, previous_layers=previous_layers)

  # Build layer-specs for layers 2-to-9.
  return ModelConnectionSpec(
      layer_specs=[_build_layer_spec(i) for i in range(2, 10)])


@pg.members([('kernel_size', pg.typing.Int()),
             ('filter_size', pg.typing.Int()),
             ('skip_layer', pg.typing.Bool())])
class SkippableLayerSpec(pg.Object):
  """Specifies a model-layer which can be skipped."""


@pg.members([('layer_specs',
              pg.typing.List(pg.typing.Object(SkippableLayerSpec)))])
class VariableLengthModelSpec(pg.Object):
  """Specifies a model with variable number of layers, each with different spec."""


def variable_length_model_search_space():
  """Returns a variable length model search space."""
  # To create a variable length model search space for 2, 3, or 4 layers model:
  # Create first 2 layers which can not be skipped followed
  # by next 2 layers which may or may not be skipped.
  layer_specs = [
      SkippableLayerSpec(
          kernel_size=pg.oneof([3, 5]),
          filter_size=pg.oneof([4, 6, 8]),
          skip_layer=False)
  ] * 2 + [
      SkippableLayerSpec(
          kernel_size=pg.oneof([3, 5]),
          filter_size=pg.oneof([4, 6, 8]),
          skip_layer=pg.oneof([True, False]))
  ] * 2
  return VariableLengthModelSpec(layer_specs=layer_specs)


def get_search_space(search_space_choice):
  """Returns a search space given a name."""
  if search_space_choice == 'basic_one_of_search_space':
    return basic_one_of_search_space()
  elif search_space_choice == 'basic_sublist_of_search_space':
    return basic_sublist_of_search_space()
  elif search_space_choice == 'mnist_list_of_dictionary_search_space':
    return mnist_list_of_dictionary_search_space()
  elif search_space_choice == 'mnist_class_search_space':
    return mnist_class_search_space()
  elif search_space_choice == 'architecture_connection_search_space':
    return architecture_connection_search_space()
  elif search_space_choice == 'variable_length_model_search_space':
    return variable_length_model_search_space()
  else:
    raise ValueError('Unknown search space: %s' % search_space_choice)
