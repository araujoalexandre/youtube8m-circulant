# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf

from utils_layer import CirculantLayer, CirculantLayerWithFactor
from utils_layer import context_gating

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer("moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")
flags.DEFINE_bool("MoE_gating", True, 
                  """Activate Context Gating after MoE.""")
flags.DEFINE_float("moe_l2", 1e-8,
                   "L2 penalty for MoeModel.")

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   is_training=None,
                   add_batch_norm=None,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   gating=None,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
    gating = gating or FLAGS.MoE_gating
    l2_penalty = FLAGS.moe_l2

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    
    if gating:
      with tf.variable_scope('gating_video_level'):
        final_probabilities = context_gating(final_probabilities, 
                                add_batch_norm, is_training)  

    return {"predictions": final_probabilities}

class Circulant_MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
    model_input_dim = model_input.get_shape().as_list()

    with tf.variable_scope('circulant_gates') as scope:

      gate_activations_size = vocab_size * (num_mixtures + 1)
      initializer = tf.random_normal_initializer(
        stddev=1 / math.sqrt(gate_activations_size))
      circ_gate_activations = CirculantLayer(
        model_input_dim, gate_activations_size, initializer)

      gate_activations = circ_gate_activations.matmul(model_input)

    with tf.variable_scope('circulant_expert') as scope:

      expert_activations_size = vocab_size * num_mixtures
      initializer = tf.random_normal_initializer(
              stddev=1 / math.sqrt(expert_activations_size))
      circ_expert_activations = CirculantLayer(
        model_input_dim, expert_activations_size, initializer)

      expert_activations = circ_expert_activations.matmul(model_input)

      initializer = tf.constant_initializer(1/math.sqrt(expert_activations_size))
      biases = tf.get_variable(
                name='biases',
                shape=(expert_activations_size),
                dtype=tf.float32,
                initializer=initializer,
                trainable=True)
      expert_activations = tf.nn.bias_add(expert_activations, biases)


    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}


class CirculantWithFactor_MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   k_factor=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
    k_factor = k_factor or FLAGS.k_factor

    model_input_dim = model_input.get_shape().as_list()

    with tf.variable_scope('circulant_gates') as scope:

      gate_activations_size = vocab_size * (num_mixtures + 1)
      initializer = tf.random_normal_initializer(
        stddev=1 / math.sqrt(gate_activations_size))
      circ_gate_activations = CirculantLayerWithFactor(
        model_input_dim, gate_activations_size, 
        k_factor=k_factor, initializer=initializer)

      gate_activations = circ_gate_activations.matmul(model_input)

    with tf.variable_scope('circulant_expert') as scope:

      expert_activations_size = vocab_size * num_mixtures
      initializer = tf.random_normal_initializer(
              stddev=1 / math.sqrt(expert_activations_size))
      circ_expert_activations = CirculantLayerWithFactor(
        model_input_dim, expert_activations_size, 
        k_factor=k_factor, initializer=initializer)

      expert_activations = circ_expert_activations.matmul(model_input)

      initializer = tf.constant_initializer(1/math.sqrt(expert_activations_size))
      biases = tf.get_variable(
                name='biases',
                shape=(expert_activations_size),
                dtype=tf.float32,
                initializer=initializer,
                trainable=True)
      expert_activations = tf.nn.bias_add(expert_activations, biases)


    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}
