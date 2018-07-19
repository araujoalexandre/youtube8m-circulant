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

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils
from utils_layer import CirculantLayer, NetVLAD, DBof
from utils_layer import context_gating

import tensorflow.contrib.slim as slim
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")

flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool("dbof_gating", True, 
                  """Activate Context Gating layer after DBoF clustering.""")

flags.DEFINE_integer("netvlad_cluster_size", 128,
                     "Number of units in the Netvlad cluster layer.")
flags.DEFINE_integer("netvlad_hidden_size", 1024,
                     "Number of units in the Netvlad hidden layer.")
flags.DEFINE_bool("netvlad_gating", True,
                    "Activate Context Gating layer after Netvlad clustering.")
flags.DEFINE_bool("netvlad_add_batch_norm", True,
                  "Adds batch normalization to the Netvlad model.")

flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")

flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")

flags.DEFINE_bool("input_add_batch_norm", True,
                  "Adds batch normalization to the input model.")

flags.DEFINE_integer("full_hidden_size", 1024,
                     "Number of units in the fc layer in Double DbofD/NetVLAD model.")
flags.DEFINE_bool("full_add_batch_norm", True,
                  "Adds batch normalization to the Double DbofD/NetVLAD model.")
flags.DEFINE_bool("full_gating", True,
                  "Activate Context Gating layer after Full fc layer.")

flags.DEFINE_bool('moe_add_batch_norm', True, 
                  "Adds batch normalization to the MoE Layer.")

k_factor

class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators

    output = slim.fully_connected(
        avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}

class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   gating=None,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size
    gating = gating or FLAGS.dbof_gating

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.get_variable("cluster_weights",
      [feature_size, cluster_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
        [cluster_size],
        initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.get_variable("hidden1_weights",
      [cluster_size, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    if gating:
      with tf.variable_scope('gating_frame_level'):
        activation = context_gating(activation, add_batch_norm, is_training)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=add_batch_norm,
        **unused_params)

class LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)

class NetVLADModel(models.BaseModel):
  """Creates a NetVLAD based model.
  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).
  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """


  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   gating=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.netvlad_cluster_size
    hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
    gating = FLAGS.netvlad_gating 

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])

    video_NetVLAD = NetVLAD(1024, max_frames, cluster_size, add_batch_norm, is_training)
    audio_NetVLAD = NetVLAD(128, max_frames, cluster_size // 2, add_batch_norm, is_training)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_VLAD"):
        vlad_video = video_NetVLAD.forward(reshaped_input[:, 0:1024]) 

    with tf.variable_scope("audio_VLAD"):
        vlad_audio = audio_NetVLAD.forward(reshaped_input[:, 1024:])

    vlad = tf.concat([vlad_video, vlad_audio], 1)

    vlad_dim = vlad.get_shape().as_list()[1] 
    hidden1_weights = tf.get_variable("hidden1_weights",
      [vlad_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
       
    activation = tf.matmul(vlad, hidden1_weights)

    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")

    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
   
    activation = tf.nn.relu6(activation)
   
    if gating:
      with tf.variable_scope('gating_frame_level'):
        activation = context_gating(activation, add_batch_norm, is_training)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=add_batch_norm,
        **unused_params)

class DoubleDbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   gating=None,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size
    gating = gating or FLAGS.dbof_gating

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    video_DBoF = DBof(1024, max_frames, cluster_size, 
      FLAGS.dbof_pooling_method, add_batch_norm, is_training)
    audio_DBoF = DBof(128, max_frames, cluster_size // 2, 
      FLAGS.dbof_pooling_method, add_batch_norm, is_training)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_DBoF"):
        dbof_video = video_DBoF.forward(reshaped_input[:, 0:1024]) 

    with tf.variable_scope("audio_DBoF"):
        dbof_audio = audio_DBoF.forward(reshaped_input[:, 1024:])

    activation = tf.concat([dbof_video, dbof_audio], 1)

    dim = activation.get_shape().as_list()[1]
    hidden1_weights = tf.get_variable("hidden1_weights",
      [dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(dim)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    if gating:
      with tf.variable_scope('gating_frame_level'):
        activation = context_gating(activation, add_batch_norm, is_training)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=add_batch_norm,
        **unused_params)


class DoubleDbofDoubleNetVLADModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   sample_random_frames=None,
                   input_add_batch_norm=None,
                   dbof_add_batch_norm=None,
                   dbof_cluster_size=None,
                   dbof_hidden_size=None,
                   dbof_gating=None,
                   netvlad_add_batch_norm=None,
                   netvlad_cluster_size=None,
                   netvlad_hidden_size=None,
                   netvlad_gating=None,
                   full_hidden_size=None,
                   full_add_batch_norm=None,
                   full_gating=None,
                   moe_add_batch_norm=None,
                   is_training=True,
                   **unused_params):

    iterations = iterations or FLAGS.iterations
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    input_add_batch_norm = input_add_batch_norm or FLAGS.input_add_batch_norm
    
    dbof_add_batch_norm = dbof_add_batch_norm or FLAGS.dbof_add_batch_norm
    dbof_cluster_size = dbof_cluster_size or FLAGS.dbof_cluster_size
    dbof_hidden_size = dbof_hidden_size or FLAGS.dbof_hidden_size
    dbof_gating = dbof_gating or FLAGS.dbof_gating

    netvlad_add_batch_norm = netvlad_add_batch_norm or FLAGS.netvlad_add_batch_norm
    netvlad_cluster_size = netvlad_cluster_size or FLAGS.netvlad_cluster_size
    netvlad_hidden_size = netvlad_hidden_size or FLAGS.netvlad_hidden_size
    netvlad_gating = netvlad_gating or FLAGS.netvlad_gating

    full_add_batch_norm = full_add_batch_norm or FLAGS.full_add_batch_norm
    full_hidden_size = full_hidden_size or FLAGS.full_hidden_size
    full_gating = full_gating or FLAGS.full_gating

    moe_add_batch_norm = moe_add_batch_norm or FLAGS.moe_add_batch_norm

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if input_add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope('DBof'):

      video_DBoF = DBof(1024, max_frames, dbof_cluster_size, 
        FLAGS.dbof_pooling_method, dbof_add_batch_norm, is_training)
      audio_DBoF = DBof(128, max_frames, dbof_cluster_size // 2, 
        FLAGS.dbof_pooling_method, dbof_add_batch_norm, is_training)

      with tf.variable_scope("video_DBoF"):
          dbof_video = video_DBoF.forward(reshaped_input[:, 0:1024]) 

      with tf.variable_scope("audio_DBoF"):
          dbof_audio = audio_DBoF.forward(reshaped_input[:, 1024:])

      dbof_activation = tf.concat([dbof_video, dbof_audio], 1)

      dim = dbof_activation.get_shape().as_list()[1]
      dbof_hidden_weights = tf.get_variable("dbof_hidden_weights",
        [dim, dbof_hidden_size],
        initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(dim)))
      tf.summary.histogram("dbof_hidden_weights", dbof_hidden_weights)
      
      dbof_activation = tf.matmul(dbof_activation, dbof_hidden_weights)
      
      if dbof_add_batch_norm:
        dbof_activation = slim.batch_norm(
            dbof_activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="hidden1_bn")
      else:
        dbof_hidden_biases = tf.get_variable("dbof_hidden_biases",
          [dbof_hidden_size],
          initializer = tf.random_normal_initializer(stddev=0.01))
        tf.summary.histogram("dbof_hidden_biases", dbof_hidden_biases)
        dbof_activation += dbof_hidden_biases
      
      dbof_activation = tf.nn.relu6(dbof_activation)
      tf.summary.histogram("dbof_hidden_output", dbof_activation)

    with tf.variable_scope('NetVLAD'):

      video_NetVLAD = NetVLAD(1024, max_frames, netvlad_cluster_size, netvlad_add_batch_norm, is_training)
      audio_NetVLAD = NetVLAD(128, max_frames, netvlad_cluster_size // 2, netvlad_add_batch_norm, is_training)

      with tf.variable_scope("video_VLAD"):
          vlad_video = video_NetVLAD.forward(reshaped_input[:, 0:1024]) 

      with tf.variable_scope("audio_VLAD"):
          vlad_audio = audio_NetVLAD.forward(reshaped_input[:, 1024:])

      vlad = tf.concat([vlad_video, vlad_audio], 1)

      vlad_dim = vlad.get_shape().as_list()[1] 
      vlad_hidden_weights = tf.get_variable("vlad_hidden_weights",
        [vlad_dim, netvlad_hidden_size],
        initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(netvlad_hidden_size)))
         
      vlad_activation = tf.matmul(vlad, vlad_hidden_weights)

      if netvlad_add_batch_norm:
        vlad_activation = slim.batch_norm(
            vlad_activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="netvlad_hidden_bn")
      else:
        netvlad_hidden_biases = tf.get_variable("netvlad_hidden_biases",
          [netvlad_hidden_size],
          initializer = tf.random_normal_initializer(stddev=0.01))
        tf.summary.histogram("netvlad_hidden_biases", netvlad_hidden_biases)
        vlad_activation += netvlad_hidden_biases
     
      vlad_activation = tf.nn.relu6(vlad_activation)
      tf.summary.histogram("vlad_hidden_output", vlad_activation)

    full_activation = tf.concat([dbof_activation, vlad_activation], 1)

    with tf.variable_scope('merge_dbof_netvlad'):

      full_activation_dim = full_activation.get_shape().as_list()[1] 
      full_hidden_weights = tf.get_variable("full_hidden_weights",
        [full_activation_dim, full_hidden_size],
        initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(full_hidden_size)))
         
      full_activation = tf.matmul(full_activation, full_hidden_weights)

      if full_add_batch_norm:
        full_activation = slim.batch_norm(
            full_activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="netvlad_hidden_bn")
      else:
        full_hidden_biases = tf.get_variable("full_hidden_biases",
          [netvlad_hidden_size],
          initializer = tf.random_normal_initializer(stddev=0.01))
        tf.summary.histogram("full_hidden_biases", full_hidden_biases)
        full_activation += full_hidden_biases
     
      full_activation = tf.nn.relu6(full_activation)
      tf.summary.histogram("vlad_hidden_output", full_activation)

    if full_gating:
      with tf.variable_scope('gating_frame_level'):
        full_activation = context_gating(full_activation, full_add_batch_norm, is_training)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=full_activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=moe_add_batch_norm,
        **unused_params)



class Circulant_DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")
    
    initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size))
    input_dim = reshaped_input.get_shape().as_list()
    circ_layer_cluster = CirculantLayer(input_dim, cluster_size, initializer)

    for i, weights in enumerate(circ_layer_cluster.weights):
      tf.summary.histogram("circulant_cluster_weights{}".format(i), weights)
    
    activation = circ_layer_cluster.matmul(reshaped_input)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
        [cluster_size],
        initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)
    
    initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size))
    input_dim = activation.get_shape().as_list()
    circ_layer_hidden = CirculantLayer(input_dim, hidden1_size, initializer)

    for i, weights in enumerate(circ_layer_hidden.weights):
      tf.summary.histogram("circulant_hidden1_weights{}".format(i), weights)
    
    activation = circ_layer_hidden.matmul(activation)

    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=add_batch_norm,
        **unused_params)

class CirculantWithFactor_DoubleDbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   gating=None,
                   k_factor=None,
                   **unused_params):
    
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size
    gating = gating or FLAGS.dbof_gating
    k_factor = k_factor or FLAGS.k_factor

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")


    with tf.variable_scope("video_DBoF"):
      video_DBoF = DBof(1024, max_frames, cluster_size, 
                      FLAGS.dbof_pooling_method, add_batch_norm, is_training)
      dbof_video = video_DBoF.forward(reshaped_input[:, 0:1024]) 

    with tf.variable_scope("audio_DBoF"):
      audio_DBoF = DBof(128, max_frames, cluster_size // 2, 
                        FLAGS.dbof_pooling_method, add_batch_norm, is_training)
      dbof_audio = audio_DBoF.forward(reshaped_input[:, 1024:])

    activation = tf.concat([dbof_video, dbof_audio], 1)

    with tf.variable_scope('circulant_fc_layer'):
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size))
      input_dim = activation.get_shape().as_list()
      circ_layer_hidden = CirculantLayer(input_dim, hidden1_size, 
                  k_factor=k_factor, initializer=initializer)
      activation = circ_layer_hidden.matmul(activation)

    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    if gating:
      with tf.variable_scope('gating_frame_level'):
        activation = context_gating(activation, add_batch_norm, is_training)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=add_batch_norm,
        **unused_params)
