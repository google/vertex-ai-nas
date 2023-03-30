# Copyright 2019 Google Research. All Rights Reserved.
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
"""NAS-FPN.

Golnaz Ghiasi, Tsung-Yi Lin, Ruoming Pang, Quoc V. Le.
NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection.
https://arxiv.org/abs/1904.07392. CVPR 2019.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
from typing import List, Optional, Text, Tuple
import pydot
import pyglove as pg
import tensorflow.compat.v1 as tf
from nas_lib.lib.layers import nn_ops


@pg.members([
    ('batch_norm_relu', pg.typing.Any(default=nn_ops.BatchNormRelu()),
     'An operation that includes a batch normalization layer '
     'followed by a relu layer(optional).'),
])
class FpnFeatureResampler(pg.Object):
  """FPN feature resampler."""

  def __call__(self, feature, source_level, target_level,
               target_feature_dims, is_training, name=None):
    """Resample input feature from source level to target level."""
    feature_dims = feature.get_shape().as_list()[3]
    with tf.variable_scope('resample_{}'.format(name)):
      if feature_dims != target_feature_dims:
        feature = tf.layers.conv2d(
            feature, filters=target_feature_dims,
            kernel_size=(1, 1), padding='same')
        feature = self.batch_norm_relu(
            feature,
            is_training=is_training,
            relu=False,
            name='bn')
      if source_level < target_level:
        stride = int(2 ** (target_level - source_level))
        feature = tf.layers.max_pooling2d(
            inputs=feature,
            pool_size=stride,
            strides=[stride, stride],
            padding='SAME')
      elif source_level > target_level:
        scale = int(2 ** (source_level - target_level))
        feature = self._nearest_upsampling(feature, scale=scale)
    return feature

  def _nearest_upsampling(self, feature, scale):
    """Nearest neighbor upsampling implementation.

    Args:
      feature: A tensor with a shape of [batch, height_in, width_in, channels].
      scale: An integer multiple to scale resolution of input feature.
    Returns:
      A tensor with a shape of [batch, height_in * scale, width_in * scale,
        channels]. Same dtype as input feature.
    """
    with tf.name_scope('nearest_upsampling'):
      batch_size, _, _, channels = feature.get_shape().as_list()
      height = tf.shape(feature)[1]
      width = tf.shape(feature)[2]
      batch_size = -1 if batch_size is None else batch_size
      # Instead of broadcasting with a 6-d tensor, we're using stacking here
      # for TfLite compatibity.
      output = tf.stack([feature] * scale, axis=3)
      output = tf.stack([output] * scale, axis=2)
      return tf.reshape(output,
                        [batch_size, height * scale, width * scale, channels])


class FpnCell(pg.Object):
  """Interface for a FPN cell."""

  @abc.abstractmethod
  def __call__(self, features, min_level, max_level, feature_dims,
               feature_resampler, is_training):
    """Build a FPN cell.

    This method should take input FPN features from min_level to max_level,
    and output FPN features from min_level and max_level.

    Args:
      features: A dictionary of integer (level) to tensors as input features.
      min_level: Minimum level in input and output features.
      max_level: Maximum level in input and output features.
      feature_dims: Output FPN feature dimensions.
      feature_resampler: FPN feature resampler.
      is_training: If True, graph is created under training mode.

    Returns:
      A dict of int to Tensor (of size max_level - min_level + 1) representing
      output FPN features of min_level to max_level.
    """


@pg.members([
    ('node_type', pg.typing.Enum('output', ['intermediate', 'output']),
     'Node type.'),
    ('level', pg.typing.Int(min_value=1), 'Target output level'),
    ('input_offsets', pg.typing.List(pg.typing.Int(min_value=0)),
     'Offsets of two input features among accumulated input feature sets.')
])
class FpnNode(pg.Object):
  """Feature combination node in FPN layer."""

  @abc.abstractmethod
  def __call__(self, inputs, input_levels, is_training=False):
    """Build FPN node.

    Args:
      inputs: Input features.
      input_levels: Input feature levels, which is an 1:1 mapping with `inputs`.
      is_training: If True, graph is constructed under training mode.

    Returns:
      A tensor of combined feature.
    """
    pass

  @abc.abstractmethod
  def pydot_node(self, node_id):
    """Create a PyDot Node."""


# NOTE: we explicitly call pg.members even there is no extra
# fields in order to register concrete nodes for deserialization purpose.


@pg.members([])
class SumNode(FpnNode):
  """FPN node to sum input features."""

  def __call__(self, inputs, input_levels, is_training=False):
    return tf.math.add_n(inputs)

  def pydot_node(self, node_id):
    """Create a PyDot Node."""
    return pydot.Node(
        node_id,
        shape='box',
        style='rounded, filled',
        fillcolor='#7FFFD3',
        label='"sum: %d"' % self.level)


@pg.members([])
class GlobalAttentionNode(FpnNode):
  """FPN node to combine input features with global attention."""

  def __call__(self, inputs, input_levels, is_training=False):
    if len(inputs) != 2:
      raise NotImplementedError(
          'GlobalAttentionNode only support inputs of two tensors. '
          'Encountered: {}'.format(inputs))
    assert len(inputs) == len(input_levels)

    def global_attention(feature0, feature1):
      with tf.variable_scope('global_attention'):
        m = tf.reduce_max(feature0, axis=[1, 2], keepdims=True)
        m = tf.sigmoid(m)
        return feature0 + feature1 * m

    if input_levels[0] > input_levels[1]:
      return global_attention(inputs[0], inputs[1])
    return global_attention(inputs[1], inputs[0])

  def pydot_node(self, node_id):
    """Create a PyDot Node."""
    return pydot.Node(
        node_id,
        shape='box',
        style='rounded, filled',
        fillcolor='yellow',
        label='"attention: %d"' % self.level)


@pg.members([
    ('nodes', pg.typing.List(pg.typing.Object(FpnNode)), 'FPN nodes'),
    ('use_separable_conv', pg.typing.Bool(default=False),
     'If True, use separable convolution for convolution layer in FPN cell.'
     'Otherwise standard convolution will be used.'),
    ('dropblock', pg.typing.Any(default=nn_ops.DropBlock()),
     'Dropblock layer to use after each node.'),
    ('batch_norm_relu', pg.typing.Any(default=nn_ops.BatchNormRelu()),
     'An operation that includes a batch normalization layer '
     'followed by a relu layer(optional).')
])
class FpnCellV1(FpnCell):
  """FPN cell v1 which is described by the NAS-FPN paper."""

  def __call__(self, features, min_level, max_level, feature_dims,
               feature_resampler, is_training):
    """Build a FPN cell."""
    # Number of output connections from each feature.
    num_output_connections = [0] * len(features)
    num_output_levels = max_level - min_level + 1
    feature_levels = list(range(min_level, max_level + 1))

    def resample_feature(node, input_index):
      """Resample feature based on input_index using settings from node."""
      assert input_index < len(node.input_offsets)
      feature_index = node.input_offsets[input_index]
      if feature_index >= len(features):
        raise ValueError(
            'input_offset ({}) is larger than num features({})'.format(
                feature_index, len(features)))
      resampled_feat = feature_resampler(
          feature=features[feature_index],
          source_level=feature_levels[feature_index],
          target_level=node.level,
          target_feature_dims=feature_dims,
          is_training=is_training,
          name='0_{}_{}'.format(feature_index, len(features)))

      num_output_connections[feature_index] += 1
      return resampled_feat, feature_levels[feature_index]

    for i, node in enumerate(self.nodes):
      with tf.variable_scope('sub_policy{}'.format(i)):
        tf.logging.info('sub_policy {} : {}'.format(i, node))

        # Create new feature from sub policy.
        new_level = node.level
        feature0, feature0_level = resample_feature(node, 0)
        feature1, feature1_level = resample_feature(node, 1)
        new_feature = node(
            [feature0, feature1], [feature0_level, feature1_level], is_training)

        # Add intermediate nodes that do not have any connections to output.
        if node.node_type == 'output':
          for j, (feature, feature_level, num_output) in enumerate(
              zip(features, feature_levels, num_output_connections)):
            if num_output == 0 and feature_level == new_level:
              num_output_connections[j] += 1
              new_feature += feature

        with tf.variable_scope('op_after_combine{}'.format(len(features))):
          # ReLU -> Conv -> BN after binary op.
          new_feature = tf.nn.relu(new_feature)
          if self.use_separable_conv:
            conv_op = functools.partial(
                tf.layers.separable_conv2d, depth_multiplier=1)
          else:
            conv_op = tf.layers.conv2d
          new_feature = conv_op(
              new_feature,
              filters=feature_dims,
              kernel_size=(3, 3),
              use_bias=False,
              padding='same',
              name='conv')

          new_feature = self.batch_norm_relu(
              new_feature, is_training=is_training, relu=False, name='bn')

          new_feature = self.dropblock(new_feature, is_training=is_training)
        features.append(new_feature)
        feature_levels.append(new_level)
        num_output_connections.append(0)

    output_features = {}
    for i in range(len(features) - num_output_levels, len(features)):
      level = feature_levels[i]
      output_features[level] = features[i]

    tf.logging.info('Output feature pyramid: {}'.format(output_features))
    return output_features

  def draw(self,
           graph,
           input_nodes,
           cell_id = 'cell0',
           compact = False):
    """Draw NAS-FPN cell on pydot graph."""

    if input_nodes is None:
      min_level = min([node.level for node in self.nodes])
      max_level = max([node.level for node in self.nodes])
      subgraph, input_nodes = _input_subgraph(min_level, max_level)
      graph.add_subgraph(subgraph)

    if compact:
      num_output_nodes = len(input_nodes)
      level_and_nodes = [(level, node, None) for level, node in input_nodes]

      subgraph = pydot.Cluster(
          cell_id, clusterrank='local', rank='max',
          color='gray', style='rounded', label=cell_id)

      for i, node in enumerate(self.nodes):
        graph_node = node.pydot_node('%s_node%d' % (cell_id, i))
        level_and_nodes.append((node.level, graph_node, node))

      # Add all nodes to graph.
      for _, graph_node, node in level_and_nodes:
        if node is not None:
          subgraph.add_node(graph_node)

      min_level = min([level for level, _, _ in level_and_nodes])
      max_level = max([level for level, _, _ in level_and_nodes])

      # Build per-level nodes topology.
      for current_level in range(min_level, max_level + 1):
        from_index = None
        for i, (level, _, _) in enumerate(level_and_nodes):
          if level == current_level:
            if (from_index is not None
                and from_index not in level_and_nodes[i][2].input_offsets):
              subgraph.add_edge(pydot.Edge(
                  '%s' % level_and_nodes[from_index][1].get_name(),
                  '%s' % level_and_nodes[i][1].get_name(),
                  style='invis'))
            from_index = i

      # Build node level connection.
      for level, graph_node, node in level_and_nodes:
        if node is not None:
          for input_index in node.input_offsets:
            subgraph.add_edge(pydot.Edge(
                '%s' % level_and_nodes[input_index][1].get_name(),
                '%s' % graph_node.get_name(),
                constraint=level_and_nodes[input_index][0] == level,
                tooltip='%s -> %s' % (
                    level_and_nodes[input_index][1].get_name(),
                    graph_node.get_name())))
      graph.add_subgraph(subgraph)
      return [(level, graph_node)  # pytype: disable=bad-return-type
              for level, graph_node, _ in level_and_nodes][-num_output_nodes:]
    else:
      num_output_nodes = len(input_nodes)
      subgraph = pydot.Cluster(
          cell_id, rank='max', style='rounded', label=cell_id)

      for i, node in enumerate(self.nodes):
        dot_node = node.pydot_node('%s_node%d' % (cell_id, i))
        subgraph.add_node(dot_node)
        for input_index in node.input_offsets:
          subgraph.add_edge(pydot.Edge(
              '%s' % input_nodes[input_index][1].get_name(),
              '%s' % dot_node.get_name(),
              tooltip='%s -> %s' % (input_nodes[input_index][1].get_name(),
                                    dot_node.get_name())))
        input_nodes.append((node.level, dot_node))
      graph.add_subgraph(subgraph)
      return input_nodes[-num_output_nodes:]  # pytype: disable=bad-return-type


@pg.members([
    ('min_level', pg.typing.Int(min_value=1, default=3),
     'Minimum level in NAS-FPN output feature maps.'),
    ('max_level', pg.typing.Int(min_value=1, default=7),
     'Maximum level in NAS-FPN output feature maps.'),
    ('feature_resampler', pg.typing.Object(FpnFeatureResampler),
     'FPN feature resampler used to resample features across levels.'),
    ('cell', pg.typing.Object(FpnCell), 'FPN cell.'),
    ('feature_dims', pg.typing.Int(min_value=1, default=256),
     'Number of filters in FPN layers'),
    ('num_repeats', pg.typing.Int(min_value=1, default=7),
     'Number of repeats for feature pyramid network.'),
])
class Fpn(pg.Object):
  """Feature pyramid network."""

  def __call__(self, multilevel_features, is_training=False):
    """Returns the FPN features for a given multilevel features.

    Args:
      multilevel_features: a `dict` containing `int` keys for continuous feature
        levels, e.g., [2, 3, 4, 5]. The values are corresponding features with
        shape [batch_size, height_l, width_l, num_filters].
      is_training: `bool` if True, the model is in training mode.

    Returns:
      a `dict` containing `int` keys for continuous feature levels
      [min_level, min_level + 1, ..., max_level]. The values are corresponding
      FPN features with shape [batch_size, height_l, width_l, feature_dims].
    """
    features = []
    for level in range(self.min_level, self.max_level + 1):
      if level in multilevel_features.keys():
        # TODO: The original impl. does't downsample
        # the backbone feature.
        features.append(self.feature_resampler(
            feature=multilevel_features[level],
            source_level=level,
            target_level=level,
            target_feature_dims=self.feature_dims,
            is_training=is_training,
            name='l%d' % level))
      else:
        # Adds a coarser level by downsampling the last feature map.
        features.append(self.feature_resampler(
            feature=features[-1],
            source_level=level - 1,
            target_level=level,
            target_feature_dims=self.feature_dims,
            is_training=is_training,
            name='p%d' % level))
    with tf.variable_scope('fpn_cells'):
      for i in range(self.num_repeats):
        with tf.variable_scope('cell_{}'.format(i)):
          tf.logging.info('building cell {}'.format(i))
          feature_dict = self.cell(
              features=features,
              min_level=self.min_level,
              max_level=self.max_level,
              feature_dims=self.feature_dims,
              feature_resampler=self.feature_resampler,
              is_training=is_training)
          features = [feature_dict[level] for level in range(
              self.min_level, self.max_level + 1)]
    return feature_dict

  def draw(self,
           graph,
           input_nodes = None,
           compact = True):
    """Draw NAS-FPN on a pydot graph.

    Args:
      graph: A pydot Graph object.
      input_nodes: An optional list of (level, pydot_node) as input nodes.
        If None, input_nodes will be automatically generated.
      compact: If True, draw graph in compact mode.

    Returns:
      A list of (level, pydot_node) as output nodes.
    """
    if input_nodes is None:
      subgraph, input_nodes = _input_subgraph(self.min_level, self.max_level)
      graph.add_subgraph(subgraph)

    for i in range(self.num_repeats):
      input_nodes = self.cell.draw(
          graph, input_nodes, 'cell%d' % i, compact=compact)
    return input_nodes


def _input_subgraph(min_level, max_level):
  """Create pydot subgraph for input nodes."""
  subgraph = pydot.Cluster(
      'inputs', style='rounded', color='gray',
      rank='max', label='input nodes')
  input_nodes = []
  for i in range(min_level, max_level + 1):
    input_node = pydot.Node('input_%d' % i,
                            shape='box',
                            style='rounded, filled',
                            fillcolor='#FDF5E6',
                            label='input: %d' % i)
    subgraph.add_node(input_node)
    input_nodes.append((i, input_node))
  return subgraph, input_nodes
