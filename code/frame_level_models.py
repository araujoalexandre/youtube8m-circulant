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
from utils_layer import CirculantLayer, CirculantLayerWithFactor, DBofCirculant
from utils_layer import NetVLAD, DBof, NetFV
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

flags.DEFINE_integer('k_factor', 1, 
                  "k_factor for circulant layer.")

flags.DEFINE_bool("video_add_batch_norm", True,
                  "Adds batch normalization to the video emmbedding in double Video/Audio model.")
flags.DEFINE_bool("audio_add_batch_norm", True,
                  "Adds batch normalization to the audio emmbedding in double Video/Audio model.")
flags.DEFINE_integer("video_hidden_size", 1024,
                     "Number of units in the fc layer in video emmbedding in double Video/Audio model.")
flags.DEFINE_integer("audio_hidden_size", 1024,
                     "Number of units in the fc layer in audio emmbedding in double Video/Audio model.")


flags.DEFINE_bool('fv_add_batch_norm', True, 
                  "Adds batch normalization to the FisherVector Model.")
flags.DEFINE_integer("fv_cluster_size", 64,
                     "Number of units to the FisherVector Model.")
flags.DEFINE_integer("fv_hidden_size", 1024,
                     "Number of units in the FisherVector hidden layer.")
flags.DEFINE_bool("fv_couple_weights", True,
                     "Coupling cluster weights or not") 
flags.DEFINE_float("fv_coupling_factor", 0.01,
                     "Coupling factor")
flags.DEFINE_bool("fv_gating", True,
                  "Activate Context Gating layer after Fisher Vector Layer")

flags.DEFINE_integer('n_bagging', 1, 
                     "Number of bagging to do in EnsembleEarlyConcat Model")
flags.DEFINE_bool('embedding_add_batch_norm', True, 
                  "Adds batch normalization to the embedding layer of EnsembleEarlyConcat Model.")
flags.DEFINE_bool('fc_add_batch_norm', True, 
                  "Adds batch normalization to the fc layer in EnsembleEarlyConcat Model.")
flags.DEFINE_integer("fc_hidden_size", 1024,
                     "Number of units in the EnsembleEarlyConcat hidden layer.")
flags.DEFINE_bool("fc_circulant", True,
                  "Activate Ciculant FC Layer after embbeding in EnsembleEarlyConcat")
flags.DEFINE_bool("fc_gating", True,
                  "Activate Context Gating layer after FC from Ensemble Early Concat")


flags.DEFINE_integer("gru_cells", 1024, "Number of GRU cells.")
flags.DEFINE_integer("gru_layers", 2, "Number of GRU layers.")
flags.DEFINE_bool("lstm_random_sequence", False,
                     "Random sequence input for lstm.")
flags.DEFINE_bool("gru_random_sequence", False,
                     "Random sequence input for gru.")

flags.DEFINE_bool('fc_dbof_circulant', False, "Circulant FC after dbof")
flags.DEFINE_bool('fc_netvlad_circulant', False, "Circulant FC after NetVald")
flags.DEFINE_bool('fc_fisher_circulant', False, "Circulant FC after fisher")



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

# class LstmModel(models.BaseModel):

#   def create_model(self, model_input, vocab_size, num_frames, **unused_params):
#     """Creates a model which uses a stack of LSTMs to represent the video.

#     Args:
#       model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
#                    input features.
#       vocab_size: The number of classes in the dataset.
#       num_frames: A vector of length 'batch' which indicates the number of
#            frames for each video (before padding).

#     Returns:
#       A dictionary with a tensor containing the probability predictions of the
#       model in the 'predictions' key. The dimensions of the tensor are
#       'batch_size' x 'num_classes'.
#     """
#     lstm_size = FLAGS.lstm_cells
#     number_of_layers = FLAGS.lstm_layers

#     stacked_lstm = tf.contrib.rnn.MultiRNNCell(
#             [
#                 tf.contrib.rnn.BasicLSTMCell(
#                     lstm_size, forget_bias=1.0)
#                 for _ in range(number_of_layers)
#                 ])

#     loss = 0.0

#     outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
#                                        sequence_length=num_frames,
#                                        dtype=tf.float32)

#     aggregated_model = getattr(video_level_models,
#                                FLAGS.video_level_classifier_model)

#     return aggregated_model().create_model(
#         model_input=state[-1].h,
#         vocab_size=vocab_size,
#         # **unused_params)

class LstmModel(models.BaseModel):

  def create_model(self, 
                   model_input, 
                   vocab_size, 
                   num_frames,
                   moe_add_batch_norm=None,
                   is_training=True, 
                   **unused_params):
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
    random_frames = FLAGS.lstm_random_sequence
    iterations = FLAGS.iterations
    moe_add_batch_norm = moe_add_batch_norm or FLAGS.moe_add_batch_norm

    if random_frames:
      num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                             iterations)

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=True)
                for _ in range(number_of_layers)
                ], state_is_tuple=True)

    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=moe_add_batch_norm,
        **unused_params)


class GruModel(models.BaseModel):

  def create_model(self, 
                   model_input, 
                   vocab_size, 
                   num_frames,
                   moe_add_batch_norm=None,
                   is_training=True, 
                   **unused_params):
    """Creates a model which uses a stack of GRUs to represent the video.
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
    gru_size = FLAGS.gru_cells
    number_of_layers = FLAGS.gru_layers
    random_frames = FLAGS.gru_random_sequence
    iterations = FLAGS.iterations

    moe_add_batch_norm = moe_add_batch_norm or FLAGS.moe_add_batch_norm
    
    if random_frames:
      num_frames_2 = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      model_input = utils.SampleRandomFrames(model_input, num_frames_2,
                                             iterations)
    
    stacked_GRU = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GRUCell(gru_size)
                for _ in range(number_of_layers)
                ], state_is_tuple=True)

    with tf.variable_scope("RNN"):
      outputs, state = tf.nn.dynamic_rnn(stacked_GRU, model_input,
                                         sequence_length=num_frames,
                                         dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=state[-1],
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=FLAGS.moe_add_batch_norm,
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

class NetFVModel(models.BaseModel):
  """Creates a NetFV based model.
     It emulates a Gaussian Mixture Fisher Vector pooling operations
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
                   add_batch_norm=None,
                   cluster_size=None,
                   hidden_size=None,
                   couple_weights=None,
                   coupling_factor=None,
                   moe_add_batch_norm=None,
                   is_training=True,
                   **unused_params):

    iterations = iterations or FLAGS.iterations
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    
    add_batch_norm = add_batch_norm or FLAGS.fv_add_batch_norm
    cluster_size = cluster_size or FLAGS.fv_cluster_size
    hidden1_size = hidden_size or FLAGS.fv_hidden_size
    couple_weights = couple_weights or FLAGS.fv_couple_weights
    coupling_factor = coupling_factor or FLAGS.fv_coupling_factor
    gating = FLAGS.fv_gating

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

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    with tf.variable_scope("video_FV"):
      video_NetFV = NetFV(1024, max_frames, cluster_size, add_batch_norm, 
          couple_weights, coupling_factor, is_training)
      fv_video = video_NetFV.forward(reshaped_input[:,0:1024]) 

    with tf.variable_scope("audio_FV"):
      audio_NetFV = NetFV(128, max_frames, cluster_size // 2, add_batch_norm, 
        couple_weights, coupling_factor, is_training)
      fv_audio = audio_NetFV.forward(reshaped_input[:,1024:])

    fv = tf.concat([fv_video, fv_audio], 1)

    fv_dim = fv.get_shape().as_list()[1] 
    hidden1_weights = tf.get_variable("hidden1_weights",
      [fv_dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    
    activation = tf.matmul(fv, hidden1_weights)

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
        add_batch_norm=moe_add_batch_norm,
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

class DoubleVideoDoubleAudioModel(models.BaseModel):
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
                   video_add_batch_norm=None,
                   audio_add_batch_norm=None,
                   video_hidden_size=None,
                   audio_hidden_size=None,                   
                   dbof_cluster_size=None,
                   netvlad_cluster_size=None,
                   full_hidden_size=None,
                   full_add_batch_norm=None,
                   full_gating=None,
                   moe_add_batch_norm=None,
                   is_training=True,
                   **unused_params):

    iterations = iterations or FLAGS.iterations
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    input_add_batch_norm = input_add_batch_norm or FLAGS.input_add_batch_norm
    
    video_add_batch_norm = video_add_batch_norm or FLAGS.video_add_batch_norm
    audio_add_batch_norm = audio_add_batch_norm or FLAGS.audio_add_batch_norm
    video_hidden_size = video_hidden_size or FLAGS.video_hidden_size
    audio_hidden_size = audio_hidden_size or FLAGS.audio_hidden_size

    dbof_cluster_size = dbof_cluster_size or FLAGS.dbof_cluster_size
    netvlad_cluster_size = netvlad_cluster_size or FLAGS.netvlad_cluster_size


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

    with tf.variable_scope('video'):


      with tf.variable_scope("DBoF"):
        video_DBoF = DBof(1024, max_frames, dbof_cluster_size, 
          FLAGS.dbof_pooling_method, video_add_batch_norm, is_training)
        dbof_video = video_DBoF.forward(reshaped_input[:, 0:1024]) 

      with tf.variable_scope("NetVLAD"):
        video_NetVLAD = NetVLAD(1024, max_frames, netvlad_cluster_size, video_add_batch_norm, is_training)
        vlad_video = video_NetVLAD.forward(reshaped_input[:, 0:1024]) 

      video_activation = tf.concat([dbof_video, vlad_video], 1)

      dim = video_activation.get_shape().as_list()[1]
      video_hidden_weights = tf.get_variable("dbof_hidden_weights",
        [dim, video_hidden_size],
        initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(video_hidden_size)))
      tf.summary.histogram("video_hidden_weights", video_hidden_weights)
      
      video_activation = tf.matmul(video_activation, video_hidden_weights)
      
      if video_add_batch_norm:
        video_activation = slim.batch_norm(
            video_activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="hidden1_bn")
      else:
        video_hidden_biases = tf.get_variable("dbof_hidden_biases",
          [video_hidden_size],
          initializer = tf.random_normal_initializer(stddev=0.01))
        tf.summary.histogram("dbof_hidden_biases", video_hidden_biases)
        video_activation += video_hidden_biases
      
      video_activation = tf.nn.relu6(video_activation)
      tf.summary.histogram("dbof_hidden_output", video_activation)

    with tf.variable_scope('audio'):
      
      with tf.variable_scope("DBoF"):
        audio_DBoF = DBof(128, max_frames, dbof_cluster_size // 2, 
          FLAGS.dbof_pooling_method, audio_add_batch_norm, is_training)
        dbof_audio = audio_DBoF.forward(reshaped_input[:, 1024:])

      with tf.variable_scope("NetVLAD"):
        audio_NetVLAD = NetVLAD(128, max_frames, netvlad_cluster_size // 2, audio_add_batch_norm, is_training)
        vlad_audio = audio_NetVLAD.forward(reshaped_input[:, 1024:])

      audio_activation = tf.concat([dbof_audio, vlad_audio], 1)

      audio_dim = audio_activation.get_shape().as_list()[1] 
      audio_hidden_weights = tf.get_variable("audio_hidden_weights",
        [audio_dim, audio_hidden_size],
        initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(audio_hidden_size)))
         
      audio_activation = tf.matmul(audio_activation, audio_hidden_weights)

      if audio_add_batch_norm:
        audio_activation = slim.batch_norm(
            audio_activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="audio_hidden_bn")
      else:
        audio_hidden_biases = tf.get_variable("audio_hidden_biases",
          [video_hidden_size],
          initializer = tf.random_normal_initializer(stddev=0.01))
        tf.summary.histogram("audio_hidden_biases", video_hidden_biases)
        audio_activation += audio_hidden_biases
     
      audio_activation = tf.nn.relu6(audio_activation)
      tf.summary.histogram("audio_hidden_output", audio_activation)


    full_activation = tf.concat([video_activation, audio_activation], 1)

    with tf.variable_scope('merge_video_audio'):

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
          [full_hidden_size],
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

class Ensemble_DoubleDbofModel_NetVLADModel(models.BaseModel):

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   **unused_params):
    

    with tf.variable_scope('dbof'):
      dbof = DoubleDbofModel().create_model(model_input, vocab_size, num_frames)
      dbof_predictions = dbof['predictions']
    
    with tf.variable_scope('netvlad'):
      netvlad = NetVLADModel().create_model(model_input, vocab_size, num_frames)
      netvlad_predictions = netvlad['predictions']

    final = (dbof_predictions + netvlad_predictions) / 2

    return {'predictions': final}



class EnsembleEarlyConcat(models.BaseModel):
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
                   n_bagging=None,
                   embedding_add_batch_norm=None,
                   dbof_cluster_size=None,
                   netvlad_cluster_size=None,
                   fv_cluster_size=None,
                   fv_couple_weights=None,
                   fv_coupling_factor=None,
                   fc_hidden_size=None,
                   fc_add_batch_norm=None,
                   fc_circulant=None,
                   fc_gating=None,
                   k_factor=None,
                   moe_add_batch_norm=None,
                   is_training=True,
                   **unused_params):

    iterations = iterations or FLAGS.iterations
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    input_add_batch_norm = input_add_batch_norm or FLAGS.input_add_batch_norm

    n_bagging = n_bagging or FLAGS.n_bagging
    
    embedding_add_batch_norm = embedding_add_batch_norm or FLAGS.embedding_add_batch_norm

    dbof_cluster_size = dbof_cluster_size or FLAGS.dbof_cluster_size
    netvlad_cluster_size = netvlad_cluster_size or FLAGS.netvlad_cluster_size
    fv_cluster_size = fv_cluster_size or FLAGS.fv_cluster_size
    fv_couple_weights = fv_couple_weights or FLAGS.fv_couple_weights
    fv_coupling_factor = fv_coupling_factor or FLAGS.fv_coupling_factor

    fc_add_batch_norm = fc_add_batch_norm or FLAGS.fc_add_batch_norm
    fc_hidden_size = fc_hidden_size or FLAGS.fc_hidden_size
    fc_circulant = fc_circulant or FLAGS.fc_circulant
    k_factor = k_factor or FLAGS.k_factor
    fc_gating = fc_gating or FLAGS.fc_gating

    moe_add_batch_norm = moe_add_batch_norm or FLAGS.moe_add_batch_norm


    def get_input(id_):
      num_frames_cast = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      if random_frames:
        sample_model_input = utils.SampleRandomFrames(model_input, num_frames_cast, iterations)
      else:
        sample_model_input = utils.SampleRandomSequence(model_input, num_frames_cast, iterations)
      max_frames = sample_model_input.get_shape().as_list()[1]
      feature_size = sample_model_input.get_shape().as_list()[2]
      reshaped_input = tf.reshape(sample_model_input, [-1, feature_size])
      tf.summary.histogram("input_hist_{}".format(id_), reshaped_input)
      if input_add_batch_norm:
        reshaped_input = slim.batch_norm(
            reshaped_input,
            center=True,
            scale=True,
            is_training=is_training,
            scope="input_bn_{}".format(id_))
      return reshaped_input, max_frames

    sample_model_inputs_video = []
    sample_model_inputs_audio = []
    for i in range(n_bagging):
      sample_model_input, max_frames = get_input(i)
      sample_model_inputs_video.append(sample_model_input[:, 0:1024])
      sample_model_inputs_audio.append(sample_model_input[:, 1024:])

    def make_embedding(model_inputs, size, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, name):
      with tf.variable_scope(name):

        with tf.variable_scope("DBoF_{}".format(name), reuse=tf.AUTO_REUSE):
          dbof_cls = DBof(size, max_frames, dbof_cluster_size, 
            FLAGS.dbof_pooling_method, embedding_add_batch_norm, is_training)
          list_dbof = []
          if len(model_inputs) > 1:
            for model_input in model_inputs:
              dbof = dbof_cls.forward(model_input)
              list_dbof.append(dbof)
            dbof = tf.concat(list_dbof, 1)
          else:
            dbof = dbof_cls.forward(model_inputs[0])

        with tf.variable_scope("NetVLAD{}".format(name), reuse=tf.AUTO_REUSE):
          netvlad_cls = NetVLAD(size, max_frames, netvlad_cluster_size, 
           embedding_add_batch_norm, is_training)
          list_vlad = []
          if len(model_inputs) > 1:
            for model_input in model_inputs:
              netvlad = netvlad_cls.forward(model_input)
              list_vlad.append(netvlad)
            netvlad = tf.concat(list_vlad, 1)
          else:
            netvlad = netvlad_cls.forward(model_inputs[0])

        with tf.variable_scope("Fisher_vector{}".format(name), reuse=tf.AUTO_REUSE):
          netfv_cls = NetFV(size, max_frames, fv_cluster_size, 
            embedding_add_batch_norm, fv_couple_weights, fv_coupling_factor, 
            is_training)
          list_fv = []
          if len(model_inputs) > 1:
            for model_input in model_inputs:
              fv = netfv_cls.forward(model_input)
              list_fv.append(fv)
            fv = tf.concat(list_fv, 1)
          else:
            fv = netfv_cls.forward(model_inputs[0])

      return dbof, netvlad, fv

    dbof_video, netvlad_video, fv_video = make_embedding(sample_model_inputs_video, 1024, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, 'video')
    dbof_audio, netvlad_audio, fv_audio = make_embedding(sample_model_inputs_audio, 128, 
      dbof_cluster_size // 2, netvlad_cluster_size // 2, fv_cluster_size // 2, 'audio')

    activation = tf.concat([dbof_video, netvlad_video, fv_video, 
                            dbof_audio, netvlad_audio, fv_audio], 1)

    with tf.variable_scope('merge_video_audio'):

      if not fc_circulant:
        activation_dim = activation.get_shape().as_list()[1] 
        fc_hidden_weights = tf.get_variable("fc_hidden_weights",
          [activation_dim, fc_hidden_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size)))
        activation = tf.matmul(activation, fc_hidden_weights)
      else:
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size))
        input_dim = activation.get_shape().as_list()
        circ_layer_hidden = CirculantLayerWithFactor(input_dim, fc_hidden_size, 
                    k_factor=k_factor, initializer=initializer)
        activation = circ_layer_hidden.matmul(activation)

      if fc_add_batch_norm:
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="fc_hidden_bn")
      else:
        fc_hidden_biases = tf.get_variable("fc_hidden_biases",
          [fc_hidden_size],
          initializer = tf.random_normal_initializer(stddev=0.01))
        tf.summary.histogram("fc_hidden_biases", fc_hidden_biases)
        activation += fc_hidden_biases
     
      activation = tf.nn.relu6(activation)
      tf.summary.histogram("full_hidden_output", activation)

    if fc_gating:
      with tf.variable_scope('gating_frame_level'):
        activation = context_gating(activation, fc_add_batch_norm, is_training)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=moe_add_batch_norm,
        **unused_params)

class EnsembleEarlyConcatAverage(models.BaseModel):
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
                   n_bagging=None,
                   embedding_add_batch_norm=None,
                   dbof_cluster_size=None,
                   netvlad_cluster_size=None,
                   fv_cluster_size=None,
                   fv_couple_weights=None,
                   fv_coupling_factor=None,
                   fc_hidden_size=None,
                   fc_add_batch_norm=None,
                   fc_circulant=None,
                   fc_gating=None,
                   k_factor=None,
                   moe_add_batch_norm=None,
                   is_training=True,
                   **unused_params):

    iterations = iterations or FLAGS.iterations
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    input_add_batch_norm = input_add_batch_norm or FLAGS.input_add_batch_norm

    n_bagging = n_bagging or FLAGS.n_bagging
    
    embedding_add_batch_norm = embedding_add_batch_norm or FLAGS.embedding_add_batch_norm

    dbof_cluster_size = dbof_cluster_size or FLAGS.dbof_cluster_size
    netvlad_cluster_size = netvlad_cluster_size or FLAGS.netvlad_cluster_size
    fv_cluster_size = fv_cluster_size or FLAGS.fv_cluster_size
    fv_couple_weights = fv_couple_weights or FLAGS.fv_couple_weights
    fv_coupling_factor = fv_coupling_factor or FLAGS.fv_coupling_factor

    fc_add_batch_norm = fc_add_batch_norm or FLAGS.fc_add_batch_norm
    fc_hidden_size = fc_hidden_size or FLAGS.fc_hidden_size
    fc_circulant = fc_circulant or FLAGS.fc_circulant
    k_factor = k_factor or FLAGS.k_factor
    fc_gating = fc_gating or FLAGS.fc_gating

    moe_add_batch_norm = moe_add_batch_norm or FLAGS.moe_add_batch_norm


    def get_input(id_):
      num_frames_cast = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      if random_frames:
        sample_model_input = utils.SampleRandomFrames(model_input, num_frames_cast, iterations)
      else:
        sample_model_input = utils.SampleRandomSequence(model_input, num_frames_cast, iterations)
      max_frames = sample_model_input.get_shape().as_list()[1]
      feature_size = sample_model_input.get_shape().as_list()[2]
      reshaped_input = tf.reshape(sample_model_input, [-1, feature_size])
      tf.summary.histogram("input_hist_{}".format(id_), reshaped_input)
      if input_add_batch_norm:
        reshaped_input = slim.batch_norm(
            reshaped_input,
            center=True,
            scale=True,
            is_training=is_training,
            scope="input_bn_{}".format(id_))
      return reshaped_input, max_frames

    sample_model_inputs_video = []
    sample_model_inputs_audio = []
    for i in range(n_bagging):
      sample_model_input, max_frames = get_input(i)
      sample_model_inputs_video.append(sample_model_input[:, 0:1024])
      sample_model_inputs_audio.append(sample_model_input[:, 1024:])

    def make_embedding(model_inputs, size, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, name):
      with tf.variable_scope(name):

        with tf.variable_scope("DBoF_{}".format(name), reuse=tf.AUTO_REUSE):
          dbof_cls = DBof(size, max_frames, dbof_cluster_size, 
            FLAGS.dbof_pooling_method, embedding_add_batch_norm, is_training)
          list_dbof = []
          if len(model_inputs) > 1:
            for model_input in model_inputs:
              dbof = dbof_cls.forward(model_input)
              list_dbof.append(dbof)
            dbof = tf.add_n(list_dbof) / len(list_dbof)
          else:
            dbof = dbof_cls.forward(model_inputs[0])

        with tf.variable_scope("NetVLAD{}".format(name), reuse=tf.AUTO_REUSE):
          netvlad_cls = NetVLAD(size, max_frames, netvlad_cluster_size, 
           embedding_add_batch_norm, is_training)
          list_vlad = []
          if len(model_inputs) > 1:
            for model_input in model_inputs:
              netvlad = netvlad_cls.forward(model_input)
              list_vlad.append(netvlad)
            netvlad = tf.add_n(list_vlad) / len(list_vlad)
          else:
            netvlad = netvlad_cls.forward(model_inputs[0])

        with tf.variable_scope("Fisher_vector{}".format(name), reuse=tf.AUTO_REUSE):
          netfv_cls = NetFV(size, max_frames, fv_cluster_size, 
            embedding_add_batch_norm, fv_couple_weights, fv_coupling_factor, 
            is_training)
          list_fv = []
          if len(model_inputs) > 1:
            for model_input in model_inputs:
              fv = netfv_cls.forward(model_input)
              list_fv.append(fv)
            fv = tf.add_n(list_fv) / len(list_fv)
          else:
            fv = netfv_cls.forward(model_inputs[0])

      return dbof, netvlad, fv

    dbof_video, netvlad_video, fv_video = make_embedding(sample_model_inputs_video, 1024, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, 'video')
    dbof_audio, netvlad_audio, fv_audio = make_embedding(sample_model_inputs_audio, 128, 
      dbof_cluster_size // 2, netvlad_cluster_size // 2, fv_cluster_size // 2, 'audio')

    activation = tf.concat([dbof_video, netvlad_video, fv_video, 
                            dbof_audio, netvlad_audio, fv_audio], 1)

    with tf.variable_scope('merge_video_audio'):

      if not fc_circulant:
        activation_dim = activation.get_shape().as_list()[1] 
        fc_hidden_weights = tf.get_variable("fc_hidden_weights",
          [activation_dim, fc_hidden_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size)))
        activation = tf.matmul(activation, fc_hidden_weights)
      else:
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size))
        input_dim = activation.get_shape().as_list()
        circ_layer_hidden = CirculantLayerWithFactor(input_dim, fc_hidden_size, 
                    k_factor=k_factor, initializer=initializer)
        activation = circ_layer_hidden.matmul(activation)

      if fc_add_batch_norm:
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="fc_hidden_bn")
      else:
        fc_hidden_biases = tf.get_variable("fc_hidden_biases",
          [fc_hidden_size],
          initializer = tf.random_normal_initializer(stddev=0.01))
        tf.summary.histogram("fc_hidden_biases", fc_hidden_biases)
        activation += fc_hidden_biases
     
      activation = tf.nn.relu6(activation)
      tf.summary.histogram("full_hidden_output", activation)

    if fc_gating:
      with tf.variable_scope('gating_frame_level'):
        activation = context_gating(activation, fc_add_batch_norm, is_training)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=moe_add_batch_norm,
        **unused_params)

class EnsembleEarlyConcatMaxPooling(models.BaseModel):
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
                   n_bagging=None,
                   embedding_add_batch_norm=None,
                   dbof_cluster_size=None,
                   netvlad_cluster_size=None,
                   fv_cluster_size=None,
                   fv_couple_weights=None,
                   fv_coupling_factor=None,
                   fc_hidden_size=None,
                   fc_add_batch_norm=None,
                   fc_circulant=None,
                   fc_gating=None,
                   k_factor=None,
                   moe_add_batch_norm=None,
                   is_training=True,
                   **unused_params):

    iterations = iterations or FLAGS.iterations
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    input_add_batch_norm = input_add_batch_norm or FLAGS.input_add_batch_norm

    n_bagging = n_bagging or FLAGS.n_bagging
    
    embedding_add_batch_norm = embedding_add_batch_norm or FLAGS.embedding_add_batch_norm

    dbof_cluster_size = dbof_cluster_size or FLAGS.dbof_cluster_size
    netvlad_cluster_size = netvlad_cluster_size or FLAGS.netvlad_cluster_size
    fv_cluster_size = fv_cluster_size or FLAGS.fv_cluster_size
    fv_couple_weights = fv_couple_weights or FLAGS.fv_couple_weights
    fv_coupling_factor = fv_coupling_factor or FLAGS.fv_coupling_factor

    fc_add_batch_norm = fc_add_batch_norm or FLAGS.fc_add_batch_norm
    fc_hidden_size = fc_hidden_size or FLAGS.fc_hidden_size
    fc_circulant = fc_circulant or FLAGS.fc_circulant
    k_factor = k_factor or FLAGS.k_factor
    fc_gating = fc_gating or FLAGS.fc_gating

    moe_add_batch_norm = moe_add_batch_norm or FLAGS.moe_add_batch_norm


    def get_input(id_):
      num_frames_cast = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      if random_frames:
        sample_model_input = utils.SampleRandomFrames(model_input, num_frames_cast, iterations)
      else:
        sample_model_input = utils.SampleRandomSequence(model_input, num_frames_cast, iterations)
      max_frames = sample_model_input.get_shape().as_list()[1]
      feature_size = sample_model_input.get_shape().as_list()[2]
      reshaped_input = tf.reshape(sample_model_input, [-1, feature_size])
      tf.summary.histogram("input_hist_{}".format(id_), reshaped_input)
      if input_add_batch_norm:
        reshaped_input = slim.batch_norm(
            reshaped_input,
            center=True,
            scale=True,
            is_training=is_training,
            scope="input_bn_{}".format(id_))
      return reshaped_input, max_frames

    sample_model_inputs_video = []
    sample_model_inputs_audio = []
    for i in range(n_bagging):
      sample_model_input, max_frames = get_input(i)
      sample_model_inputs_video.append(sample_model_input[:, 0:1024])
      sample_model_inputs_audio.append(sample_model_input[:, 1024:])

    def make_embedding(model_inputs, size, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, name):
      with tf.variable_scope(name):

        with tf.variable_scope("DBoF_{}".format(name), reuse=tf.AUTO_REUSE):
          dbof_cls = DBof(size, max_frames, dbof_cluster_size, 
            FLAGS.dbof_pooling_method, embedding_add_batch_norm, is_training)
          list_dbof = []
          if len(model_inputs) > 1:
            for model_input in model_inputs:
              dbof = dbof_cls.forward(model_input)
              list_dbof.append(dbof)
            # dbof = tf.add_n(list_dbof) / len(list_dbof)
            dbof = tf.reduce_max(list_dbof, 0)
          else:
            dbof = dbof_cls.forward(model_inputs[0])

        with tf.variable_scope("NetVLAD{}".format(name), reuse=tf.AUTO_REUSE):
          netvlad_cls = NetVLAD(size, max_frames, netvlad_cluster_size, 
           embedding_add_batch_norm, is_training)
          list_vlad = []
          if len(model_inputs) > 1:
            for model_input in model_inputs:
              netvlad = netvlad_cls.forward(model_input)
              list_vlad.append(netvlad)
            # netvlad = tf.add_n(list_vlad) / len(list_vlad)
            netvlad = tf.reduce_max(list_vlad, 0)
          else:
            netvlad = netvlad_cls.forward(model_inputs[0])

        with tf.variable_scope("Fisher_vector{}".format(name), reuse=tf.AUTO_REUSE):
          netfv_cls = NetFV(size, max_frames, fv_cluster_size, 
            embedding_add_batch_norm, fv_couple_weights, fv_coupling_factor, 
            is_training)
          list_fv = []
          if len(model_inputs) > 1:
            for model_input in model_inputs:
              fv = netfv_cls.forward(model_input)
              list_fv.append(fv)
            # fv = tf.add_n(list_fv) / len(list_fv)
            fv = tf.reduce_max(list_fv, 0)
          else:
            fv = netfv_cls.forward(model_inputs[0])

      return dbof, netvlad, fv

    dbof_video, netvlad_video, fv_video = make_embedding(sample_model_inputs_video, 1024, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, 'video')
    dbof_audio, netvlad_audio, fv_audio = make_embedding(sample_model_inputs_audio, 128, 
      dbof_cluster_size // 2, netvlad_cluster_size // 2, fv_cluster_size // 2, 'audio')

    activation = tf.concat([dbof_video, netvlad_video, fv_video, 
                            dbof_audio, netvlad_audio, fv_audio], 1)

    with tf.variable_scope('merge_video_audio'):

      if not fc_circulant:
        activation_dim = activation.get_shape().as_list()[1] 
        fc_hidden_weights = tf.get_variable("fc_hidden_weights",
          [activation_dim, fc_hidden_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size)))
        activation = tf.matmul(activation, fc_hidden_weights)
      else:
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size))
        input_dim = activation.get_shape().as_list()
        circ_layer_hidden = CirculantLayerWithFactor(input_dim, fc_hidden_size, 
                    k_factor=k_factor, initializer=initializer)
        activation = circ_layer_hidden.matmul(activation)

      if fc_add_batch_norm:
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="fc_hidden_bn")
      else:
        fc_hidden_biases = tf.get_variable("fc_hidden_biases",
          [fc_hidden_size],
          initializer = tf.random_normal_initializer(stddev=0.01))
        tf.summary.histogram("fc_hidden_biases", fc_hidden_biases)
        activation += fc_hidden_biases
     
      activation = tf.nn.relu6(activation)
      tf.summary.histogram("full_hidden_output", activation)

    if fc_gating:
      with tf.variable_scope('gating_frame_level'):
        activation = context_gating(activation, fc_add_batch_norm, is_training)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=moe_add_batch_norm,
        **unused_params)

class EnsembleEarlyConcatAverageWithFC(models.BaseModel):
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
                   n_bagging=None,
                   embedding_add_batch_norm=None,
                   dbof_cluster_size=None,
                   netvlad_cluster_size=None,
                   fv_cluster_size=None,
                   fv_couple_weights=None,
                   fv_coupling_factor=None,
                   fc_hidden_size=None,
                   fc_add_batch_norm=None,
                   fc_circulant=None,
                   fc_gating=None,
                   k_factor=None,
                   moe_add_batch_norm=None,
                   is_training=True,
                   **unused_params):

    iterations = iterations or FLAGS.iterations
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    input_add_batch_norm = input_add_batch_norm or FLAGS.input_add_batch_norm

    n_bagging = n_bagging or FLAGS.n_bagging
    
    embedding_add_batch_norm = embedding_add_batch_norm or FLAGS.embedding_add_batch_norm

    dbof_cluster_size = dbof_cluster_size or FLAGS.dbof_cluster_size
    netvlad_cluster_size = netvlad_cluster_size or FLAGS.netvlad_cluster_size
    fv_cluster_size = fv_cluster_size or FLAGS.fv_cluster_size
    fv_couple_weights = fv_couple_weights or FLAGS.fv_couple_weights
    fv_coupling_factor = fv_coupling_factor or FLAGS.fv_coupling_factor

    fc_add_batch_norm = fc_add_batch_norm or FLAGS.fc_add_batch_norm
    fc_hidden_size = fc_hidden_size or FLAGS.fc_hidden_size
    fc_circulant = fc_circulant or FLAGS.fc_circulant
    k_factor = k_factor or FLAGS.k_factor
    fc_gating = fc_gating or FLAGS.fc_gating

    moe_add_batch_norm = moe_add_batch_norm or FLAGS.moe_add_batch_norm


    def get_input(id_):
      num_frames_cast = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      if random_frames:
        sample_model_input = utils.SampleRandomFrames(model_input, num_frames_cast, iterations)
      else:
        sample_model_input = utils.SampleRandomSequence(model_input, num_frames_cast, iterations)
      max_frames = sample_model_input.get_shape().as_list()[1]
      feature_size = sample_model_input.get_shape().as_list()[2]
      reshaped_input = tf.reshape(sample_model_input, [-1, feature_size])
      tf.summary.histogram("input_hist_{}".format(id_), reshaped_input)
      if input_add_batch_norm:
        reshaped_input = slim.batch_norm(
            reshaped_input,
            center=True,
            scale=True,
            is_training=is_training,
            scope="input_bn_{}".format(id_))
      return reshaped_input, max_frames

    sample_model_inputs_video = []
    sample_model_inputs_audio = []
    for i in range(n_bagging):
      sample_model_input, max_frames = get_input(i)
      sample_model_inputs_video.append(sample_model_input[:, 0:1024])
      sample_model_inputs_audio.append(sample_model_input[:, 1024:])


    def make_fc(input_, name):
      if not fc_circulant:
        dim = input_.get_shape().as_list()[1] 
        fc_hidden_weights = tf.get_variable("{}_fc_hidden_weights".format(name),
          [dim, fc_hidden_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size)))
        return tf.matmul(input_, fc_hidden_weights)
      else:
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size))
        input_dim = input_.get_shape().as_list()
        circ_layer_hidden = CirculantLayerWithFactor(input_dim, fc_hidden_size, 
                    k_factor=k_factor, initializer=initializer)
        return circ_layer_hidden.matmul(input_)


    def make_embedding(model_inputs, size, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, name):
      with tf.variable_scope(name):

        with tf.variable_scope("DBoF_{}".format(name), reuse=tf.AUTO_REUSE):
          dbof_cls = DBof(size, max_frames, dbof_cluster_size, 
            FLAGS.dbof_pooling_method, embedding_add_batch_norm, is_training)
          list_dbof = []
          if len(model_inputs) > 1:
            for model_input in model_inputs:
              dbof = dbof_cls.forward(model_input)
              list_dbof.append(dbof)
            dbof = tf.add_n(list_dbof) / len(list_dbof)
          else:
            dbof = dbof_cls.forward(model_inputs[0])
          dbof = make_fc(dbof, 'dbof')

        with tf.control_dependencies([dbof]):
          with tf.variable_scope("NetVLAD_{}".format(name), reuse=tf.AUTO_REUSE):
            netvlad_cls = NetVLAD(size, max_frames, netvlad_cluster_size, 
             embedding_add_batch_norm, is_training)
            list_vlad = []
            if len(model_inputs) > 1:
              for model_input in model_inputs:
                netvlad = netvlad_cls.forward(model_input)
                list_vlad.append(netvlad)
              netvlad = tf.add_n(list_vlad) / len(list_vlad)
            else:
              netvlad = netvlad_cls.forward(model_inputs[0])
            netvlad = make_fc(netvlad, 'netvlad')

        with tf.control_dependencies([netvlad]):
          with tf.variable_scope("Fisher_vector_{}".format(name), reuse=tf.AUTO_REUSE):
            netfv_cls = NetFV(size, max_frames, fv_cluster_size, 
              embedding_add_batch_norm, fv_couple_weights, fv_coupling_factor, 
              is_training)
            list_fv = []
            if len(model_inputs) > 1:
              for model_input in model_inputs:
                fv = netfv_cls.forward(model_input)
                list_fv.append(fv)
              fv = tf.add_n(list_fv) / len(list_fv)
            else:
              fv = netfv_cls.forward(model_inputs[0])
            fv = make_fc(fv, 'fv')

      return dbof, netvlad, fv

    dbof_video, netvlad_video, fv_video = make_embedding(sample_model_inputs_video, 1024, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, 'video')
    dbof_audio, netvlad_audio, fv_audio = make_embedding(sample_model_inputs_audio, 128, 
      dbof_cluster_size // 2, netvlad_cluster_size // 2, fv_cluster_size // 2, 'audio')

    activation = tf.concat([dbof_video, netvlad_video, fv_video, 
                            dbof_audio, netvlad_audio, fv_audio], 1)

    with tf.variable_scope('merge_video_audio'):

      if not fc_circulant:
        activation_dim = activation.get_shape().as_list()[1] 
        fc_hidden_weights = tf.get_variable("fc_hidden_weights",
          [activation_dim, fc_hidden_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size)))
        activation = tf.matmul(activation, fc_hidden_weights)
      else:
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size))
        input_dim = activation.get_shape().as_list()
        circ_layer_hidden = CirculantLayerWithFactor(input_dim, fc_hidden_size, 
                    k_factor=k_factor, initializer=initializer)
        activation = circ_layer_hidden.matmul(activation)

      if fc_add_batch_norm:
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="fc_hidden_bn")
      else:
        fc_hidden_biases = tf.get_variable("fc_hidden_biases",
          [fc_hidden_size],
          initializer = tf.random_normal_initializer(stddev=0.01))
        tf.summary.histogram("fc_hidden_biases", fc_hidden_biases)
        activation += fc_hidden_biases
     
      activation = tf.nn.relu6(activation)
      tf.summary.histogram("full_hidden_output", activation)

    if fc_gating:
      with tf.variable_scope('gating_frame_level'):
        activation = context_gating(activation, fc_add_batch_norm, is_training)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=moe_add_batch_norm,
        **unused_params)



class EnsemblePoolingEarlyConcatWithFC(models.BaseModel):
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
                   n_bagging=None,
                   embedding_add_batch_norm=None,
                   dbof_cluster_size=None,
                   netvlad_cluster_size=None,
                   fv_cluster_size=None,
                   fv_couple_weights=None,
                   fv_coupling_factor=None,
                   fc_hidden_size=None,
                   fc_add_batch_norm=None,
                   fc_circulant=None,
                   fc_gating=None,
                   k_factor=None,
                   moe_add_batch_norm=None,
                   is_training=True,
                   **unused_params):

    iterations = iterations or FLAGS.iterations
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    input_add_batch_norm = input_add_batch_norm or FLAGS.input_add_batch_norm

    n_bagging = n_bagging or FLAGS.n_bagging
    
    embedding_add_batch_norm = embedding_add_batch_norm or FLAGS.embedding_add_batch_norm

    dbof_cluster_size = dbof_cluster_size or FLAGS.dbof_cluster_size
    netvlad_cluster_size = netvlad_cluster_size or FLAGS.netvlad_cluster_size
    fv_cluster_size = fv_cluster_size or FLAGS.fv_cluster_size
    fv_couple_weights = fv_couple_weights or FLAGS.fv_couple_weights
    fv_coupling_factor = fv_coupling_factor or FLAGS.fv_coupling_factor

    fc_add_batch_norm = fc_add_batch_norm or FLAGS.fc_add_batch_norm
    fc_hidden_size = fc_hidden_size or FLAGS.fc_hidden_size
    fc_circulant = fc_circulant or FLAGS.fc_circulant
    k_factor = k_factor or FLAGS.k_factor
    fc_gating = fc_gating or FLAGS.fc_gating

    moe_add_batch_norm = moe_add_batch_norm or FLAGS.moe_add_batch_norm


    def get_input(id_):
      num_frames_cast = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      if random_frames:
        sample_model_input = utils.SampleRandomFrames(model_input, num_frames_cast, iterations)
      else:
        sample_model_input = utils.SampleRandomSequence(model_input, num_frames_cast, iterations)
      max_frames = sample_model_input.get_shape().as_list()[1]
      feature_size = sample_model_input.get_shape().as_list()[2]
      reshaped_input = tf.reshape(sample_model_input, [-1, feature_size])
      tf.summary.histogram("input_hist_{}".format(id_), reshaped_input)
      if input_add_batch_norm:
        reshaped_input = slim.batch_norm(
            reshaped_input,
            center=True,
            scale=True,
            is_training=is_training,
            scope="input_bn_{}".format(id_))
      return reshaped_input, max_frames

    sample_model_inputs_video = []
    sample_model_inputs_audio = []
    for i in range(n_bagging):
      sample_model_input, max_frames = get_input(i)
      sample_model_inputs_video.append(sample_model_input[:, 0:1024])
      sample_model_inputs_audio.append(sample_model_input[:, 1024:])


    def make_fc(input_, name):
      if not fc_circulant:
        dim = input_.get_shape().as_list()[1] 
        fc_hidden_weights = tf.get_variable("{}_fc_hidden_weights".format(name),
          [dim, fc_hidden_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size)))
        return tf.matmul(input_, fc_hidden_weights)
      else:
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size))
        input_dim = input_.get_shape().as_list()
        circ_layer_hidden = CirculantLayerWithFactor(input_dim, fc_hidden_size, 
                    k_factor=k_factor, initializer=initializer)
        return circ_layer_hidden.matmul(input_)

    def attention_pooling(input_, out_dim):
      # pooling method with low rank matrix
      feature_size = input_.get_shape().as_list()[2]
      init = tf.random_normal_initializer(mean=1.0, 
        stddev=1/math.sqrt(feature_size))
      a = tf.get_variable("a_attention_pooling",
        [feature_size, out_dim], initializer=init)
      b = tf.get_variable("b_attention_pooling",
        [feature_size, out_dim], initializer=init)
      m1 = tf.tensordot(input_, a, 1)
      m2 = tf.tensordot(input_, b, 1)
      return tf.reduce_sum(tf.multiply(m1, m2), 1)

    def make_embedding(model_inputs, size, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, name):
      with tf.variable_scope(name):

        with tf.variable_scope("DBoF_{}".format(name), reuse=tf.AUTO_REUSE):
          dbof_cls = DBof(size, max_frames, dbof_cluster_size, 
            FLAGS.dbof_pooling_method, embedding_add_batch_norm, is_training)
          list_dbof = []
          if len(model_inputs) > 1:
            for model_input in model_inputs:
              dbof = dbof_cls.forward(model_input)
              dbof = make_fc(dbof, 'dbof')
              list_dbof.append(dbof)
            # dbof = tf.add_n(list_dbof) / len(list_dbof)
            dbof = attention_pooling(tf.stack(list_dbof, 1), fc_hidden_size)
          else:
            dbof = dbof_cls.forward(model_inputs[0])
          # dbof = make_fc(dbof, 'dbof')

        with tf.variable_scope("NetVLAD_{}".format(name), reuse=tf.AUTO_REUSE):
          netvlad_cls = NetVLAD(size, max_frames, netvlad_cluster_size, 
           embedding_add_batch_norm, is_training)
          list_vlad = []
          if len(model_inputs) > 1:
            for model_input in model_inputs:
              netvlad = netvlad_cls.forward(model_input)
              netvlad = make_fc(netvlad, 'netvlad')
              list_vlad.append(netvlad)
            # netvlad = tf.add_n(list_vlad) / len(list_vlad)
            netvlad = attention_pooling(tf.stack(list_vlad, 1), fc_hidden_size)
          else:
            netvlad = netvlad_cls.forward(model_inputs[0])
          # netvlad = make_fc(netvlad, 'netvlad')

        with tf.variable_scope("Fisher_vector_{}".format(name), reuse=tf.AUTO_REUSE):
          netfv_cls = NetFV(size, max_frames, fv_cluster_size, 
            embedding_add_batch_norm, fv_couple_weights, fv_coupling_factor, 
            is_training)
          list_fv = []
          if len(model_inputs) > 1:
            for model_input in model_inputs:
              fv = netfv_cls.forward(model_input)
              fv = make_fc(fv, 'fv')
              list_fv.append(fv)
            # fv = tf.add_n(list_fv) / len(list_fv)
            fv = attention_pooling(tf.stack(list_fv, 1), fc_hidden_size)
          else:
            fv = netfv_cls.forward(model_inputs[0])
          # fv = make_fc(fv, 'fv')

      return dbof, netvlad, fv

    dbof_video, netvlad_video, fv_video = make_embedding(sample_model_inputs_video, 1024, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, 'video')
    dbof_audio, netvlad_audio, fv_audio = make_embedding(sample_model_inputs_audio, 128, 
      dbof_cluster_size // 2, netvlad_cluster_size // 2, fv_cluster_size // 2, 'audio')

    activation = tf.concat([dbof_video, netvlad_video, fv_video, 
                            dbof_audio, netvlad_audio, fv_audio], 1)

    with tf.variable_scope('merge_video_audio'):

      if not fc_circulant:
        activation_dim = activation.get_shape().as_list()[1] 
        fc_hidden_weights = tf.get_variable("fc_hidden_weights",
          [activation_dim, fc_hidden_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size)))
        activation = tf.matmul(activation, fc_hidden_weights)
      else:
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size))
        input_dim = activation.get_shape().as_list()
        circ_layer_hidden = CirculantLayerWithFactor(input_dim, fc_hidden_size, 
                    k_factor=k_factor, initializer=initializer)
        activation = circ_layer_hidden.matmul(activation)

      if fc_add_batch_norm:
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="fc_hidden_bn")
      else:
        fc_hidden_biases = tf.get_variable("fc_hidden_biases",
          [fc_hidden_size],
          initializer = tf.random_normal_initializer(stddev=0.01))
        tf.summary.histogram("fc_hidden_biases", fc_hidden_biases)
        activation += fc_hidden_biases
     
      activation = tf.nn.relu6(activation)
      tf.summary.histogram("full_hidden_output", activation)

    if fc_gating:
      with tf.variable_scope('gating_frame_level'):
        activation = context_gating(activation, fc_add_batch_norm, is_training)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=moe_add_batch_norm,
        **unused_params)



class EnsembleEarlyConcatL2NormWithFC(models.BaseModel):
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
                   n_bagging=None,
                   embedding_add_batch_norm=None,
                   dbof_cluster_size=None,
                   netvlad_cluster_size=None,
                   fv_cluster_size=None,
                   fv_couple_weights=None,
                   fv_coupling_factor=None,
                   fc_hidden_size=None,
                   fc_add_batch_norm=None,
                   fc_circulant=None,
                   fc_gating=None,
                   k_factor=None,
                   moe_add_batch_norm=None,
                   is_training=True,
                   **unused_params):

    iterations = iterations or FLAGS.iterations
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    input_add_batch_norm = input_add_batch_norm or FLAGS.input_add_batch_norm

    n_bagging = n_bagging or FLAGS.n_bagging
    
    embedding_add_batch_norm = embedding_add_batch_norm or FLAGS.embedding_add_batch_norm

    dbof_cluster_size = dbof_cluster_size or FLAGS.dbof_cluster_size
    netvlad_cluster_size = netvlad_cluster_size or FLAGS.netvlad_cluster_size
    fv_cluster_size = fv_cluster_size or FLAGS.fv_cluster_size
    fv_couple_weights = fv_couple_weights or FLAGS.fv_couple_weights
    fv_coupling_factor = fv_coupling_factor or FLAGS.fv_coupling_factor

    fc_add_batch_norm = fc_add_batch_norm or FLAGS.fc_add_batch_norm
    fc_hidden_size = fc_hidden_size or FLAGS.fc_hidden_size
    fc_circulant = fc_circulant or FLAGS.fc_circulant
    k_factor = k_factor or FLAGS.k_factor
    fc_gating = fc_gating or FLAGS.fc_gating

    moe_add_batch_norm = moe_add_batch_norm or FLAGS.moe_add_batch_norm


    def get_input(id_):
      num_frames_cast = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      if random_frames:
        sample_model_input = utils.SampleRandomFrames(model_input, num_frames_cast, iterations)
      else:
        sample_model_input = utils.SampleRandomSequence(model_input, num_frames_cast, iterations)
      max_frames = sample_model_input.get_shape().as_list()[1]
      feature_size = sample_model_input.get_shape().as_list()[2]
      reshaped_input = tf.reshape(sample_model_input, [-1, feature_size])
      tf.summary.histogram("input_hist_{}".format(id_), reshaped_input)
      if input_add_batch_norm:
        reshaped_input = slim.batch_norm(
            reshaped_input,
            center=True,
            scale=True,
            is_training=is_training,
            scope="input_bn",
            reuse=tf.AUTO_REUSE)
      return reshaped_input, max_frames

    sample_model_inputs_video = []
    sample_model_inputs_audio = []
    for i in range(n_bagging):
      sample_model_input, max_frames = get_input(i)
      sample_model_inputs_video.append(sample_model_input[:, 0:1024])
      sample_model_inputs_audio.append(sample_model_input[:, 1024:])


    def make_fc(input_, name):
      if not fc_circulant:
        dim = input_.get_shape().as_list()[1] 
        fc_hidden_weights = tf.get_variable("{}_fc_hidden_weights".format(name),
          [dim, fc_hidden_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size)))
        return tf.matmul(input_, fc_hidden_weights)
      else:
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size))
        input_dim = input_.get_shape().as_list()
        circ_layer_hidden = CirculantLayerWithFactor(input_dim, fc_hidden_size, 
                    k_factor=k_factor, initializer=initializer)
        return circ_layer_hidden.matmul(input_)


    def make_embedding(model_inputs, size, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, name):
      with tf.variable_scope(name):

        with tf.variable_scope("DBoF_{}".format(name), reuse=tf.AUTO_REUSE):
          dbof_cls = DBof(size, max_frames, dbof_cluster_size, 
            FLAGS.dbof_pooling_method, embedding_add_batch_norm, is_training)
          list_dbof = []
          if len(model_inputs) > 1:
            for model_input in model_inputs:
              dbof = dbof_cls.forward(model_input)
              list_dbof.append(dbof)
            # dbof = tf.add_n(list_dbof) / len(list_dbof)
            list_dbof = tf.stack(list_dbof, 1)
            list_dbof_max = tf.reduce_max([tf.reduce_max(list_dbof), 1e-4])
            dbof = list_dbof_max * tf.sqrt(tf.reduce_sum(tf.pow(list_dbof / list_dbof_max, 2), 1))
          else:
            dbof = dbof_cls.forward(model_inputs[0])
          dbof = make_fc(dbof, 'dbof')

        with tf.control_dependencies([dbof]):
          with tf.variable_scope("NetVLAD_{}".format(name), reuse=tf.AUTO_REUSE):
            netvlad_cls = NetVLAD(size, max_frames, netvlad_cluster_size, 
             embedding_add_batch_norm, is_training)
            list_vlad = []
            if len(model_inputs) > 1:
              for model_input in model_inputs:
                netvlad = netvlad_cls.forward(model_input)
                list_vlad.append(netvlad)
              # netvlad = tf.add_n(list_vlad) / len(list_vlad)
              list_vlad = tf.stack(list_vlad, 1)
              list_vlad_max = tf.reduce_max([tf.reduce_max(list_vlad), 1e-4])
              netvlad = list_vlad_max * tf.sqrt(tf.reduce_sum(tf.pow(list_vlad / list_vlad_max, 2), 1))
            else:
              netvlad = netvlad_cls.forward(model_inputs[0])
            netvlad = make_fc(netvlad, 'netvlad')

        with tf.control_dependencies([netvlad]):
          with tf.variable_scope("Fisher_vector_{}".format(name), reuse=tf.AUTO_REUSE):
            netfv_cls = NetFV(size, max_frames, fv_cluster_size, 
              embedding_add_batch_norm, fv_couple_weights, fv_coupling_factor, 
              is_training)
            list_fv = []
            if len(model_inputs) > 1:
              for model_input in model_inputs:
                fv = netfv_cls.forward(model_input)
                list_fv.append(fv)
              # fv = tf.add_n(list_fv) / len(list_fv)
              list_fv = tf.stack(list_fv, 1)
              list_fv_max = tf.reduce_max([tf.reduce_max(list_fv), 1e-4])
              fv = list_fv_max * tf.sqrt(tf.reduce_sum(tf.pow(list_fv / list_fv_max, 2), 1))
            else:
              fv = netfv_cls.forward(model_inputs[0])
            fv = make_fc(fv, 'fv')

      return dbof, netvlad, fv

    dbof_video, netvlad_video, fv_video = make_embedding(sample_model_inputs_video, 1024, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, 'video')
    dbof_audio, netvlad_audio, fv_audio = make_embedding(sample_model_inputs_audio, 128, 
      dbof_cluster_size // 2, netvlad_cluster_size // 2, fv_cluster_size // 2, 'audio')

    activation = tf.concat([dbof_video, netvlad_video, fv_video, dbof_audio, netvlad_audio, fv_audio], 1)

    with tf.variable_scope('merge_video_audio'):

      if not fc_circulant:
        activation_dim = activation.get_shape().as_list()[1] 
        fc_hidden_weights = tf.get_variable("fc_hidden_weights",
          [activation_dim, fc_hidden_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size)))
        activation = tf.matmul(activation, fc_hidden_weights)
      else:
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size))
        input_dim = activation.get_shape().as_list()
        circ_layer_hidden = CirculantLayerWithFactor(input_dim, fc_hidden_size, 
                    k_factor=k_factor, initializer=initializer)
        activation = circ_layer_hidden.matmul(activation)

      if fc_add_batch_norm:
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="fc_hidden_bn")
      else:
        fc_hidden_biases = tf.get_variable("fc_hidden_biases",
          [fc_hidden_size],
          initializer = tf.random_normal_initializer(stddev=0.01))
        tf.summary.histogram("fc_hidden_biases", fc_hidden_biases)
        activation += fc_hidden_biases
     
      activation = tf.nn.relu6(activation)
      tf.summary.histogram("full_hidden_output", activation)

    if fc_gating:
      with tf.variable_scope('gating_frame_level'):
        activation = context_gating(activation, fc_add_batch_norm, is_training)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=moe_add_batch_norm,
        **unused_params)



class EnsembleEarlyConcatAverageWithFCv2(models.BaseModel):
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
                   n_bagging=None,
                   embedding_add_batch_norm=None,
                   dbof_cluster_size=None,
                   netvlad_cluster_size=None,
                   fv_cluster_size=None,
                   fv_couple_weights=None,
                   fv_coupling_factor=None,
                   
                   fc_dbof_circulant=None,
                   fc_netvlad_circulant=None,
                   fc_fisher_circulant=None,

                   fc_hidden_size=None,
                   fc_add_batch_norm=None,
                   fc_circulant=None,
                   fc_gating=None,
                   k_factor=None,
                   moe_add_batch_norm=None,
                   is_training=True,
                   **unused_params):

    iterations = iterations or FLAGS.iterations
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    input_add_batch_norm = input_add_batch_norm or FLAGS.input_add_batch_norm

    n_bagging = n_bagging or FLAGS.n_bagging
    
    embedding_add_batch_norm = embedding_add_batch_norm or FLAGS.embedding_add_batch_norm

    dbof_cluster_size = dbof_cluster_size or FLAGS.dbof_cluster_size
    netvlad_cluster_size = netvlad_cluster_size or FLAGS.netvlad_cluster_size
    fv_cluster_size = fv_cluster_size or FLAGS.fv_cluster_size
    fv_couple_weights = fv_couple_weights or FLAGS.fv_couple_weights
    fv_coupling_factor = fv_coupling_factor or FLAGS.fv_coupling_factor

    fc_dbof_circulant = fc_dbof_circulant or FLAGS.fc_dbof_circulant
    fc_netvlad_circulant = fc_netvlad_circulant or FLAGS.fc_netvlad_circulant
    fc_fisher_circulant = fc_fisher_circulant or FLAGS.fc_fisher_circulant

    fc_add_batch_norm = fc_add_batch_norm or FLAGS.fc_add_batch_norm
    fc_hidden_size = fc_hidden_size or FLAGS.fc_hidden_size
    fc_circulant = fc_circulant or FLAGS.fc_circulant
    k_factor = k_factor or FLAGS.k_factor
    fc_gating = fc_gating or FLAGS.fc_gating

    moe_add_batch_norm = moe_add_batch_norm or FLAGS.moe_add_batch_norm


    def get_input(id_):
      num_frames_cast = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      if random_frames:
        sample_model_input = utils.SampleRandomFrames(model_input, num_frames_cast, iterations)
      else:
        sample_model_input = utils.SampleRandomSequence(model_input, num_frames_cast, iterations)
      max_frames = sample_model_input.get_shape().as_list()[1]
      feature_size = sample_model_input.get_shape().as_list()[2]
      reshaped_input = tf.reshape(sample_model_input, [-1, feature_size])
      tf.summary.histogram("input_hist_{}".format(id_), reshaped_input)
      if input_add_batch_norm:
        reshaped_input = slim.batch_norm(
            reshaped_input,
            center=True,
            scale=True,
            is_training=is_training,
            scope="input_bn_{}".format(id_))
      return reshaped_input, max_frames

    sample_model_inputs_video = []
    sample_model_inputs_audio = []
    for i in range(n_bagging):
      sample_model_input, max_frames = get_input(i)
      sample_model_inputs_video.append(sample_model_input[:, 0:1024])
      sample_model_inputs_audio.append(sample_model_input[:, 1024:])


    def make_fc(input_, name, circulant):
      if not circulant:
        dim = input_.get_shape().as_list()[1] 
        fc_hidden_weights = tf.get_variable("{}_fc_hidden_weights".format(name),
          [dim, fc_hidden_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size)))
        return tf.matmul(input_, fc_hidden_weights)
      else:
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size))
        input_dim = input_.get_shape().as_list()
        circ_layer_hidden = CirculantLayerWithFactor(input_dim, fc_hidden_size, 
                    k_factor=k_factor, initializer=initializer)
        return circ_layer_hidden.matmul(input_)


    def make_embedding(model_inputs, size, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, name):
      with tf.variable_scope(name):

        with tf.variable_scope("DBoF_{}".format(name), reuse=tf.AUTO_REUSE):
          dbof_cls = DBof(size, max_frames, dbof_cluster_size, 
            FLAGS.dbof_pooling_method, embedding_add_batch_norm, is_training)
          list_dbof = []
          if len(model_inputs) > 1:
            for model_input in model_inputs:
              dbof = dbof_cls.forward(model_input)
              list_dbof.append(dbof)
            dbof = tf.add_n(list_dbof) / len(list_dbof)
          else:
            dbof = dbof_cls.forward(model_inputs[0])
          dbof = make_fc(dbof, 'dbof', fc_dbof_circulant)

        with tf.control_dependencies([dbof]):
          with tf.variable_scope("NetVLAD_{}".format(name), reuse=tf.AUTO_REUSE):
            netvlad_cls = NetVLAD(size, max_frames, netvlad_cluster_size, 
             embedding_add_batch_norm, is_training)
            list_vlad = []
            if len(model_inputs) > 1:
              for model_input in model_inputs:
                netvlad = netvlad_cls.forward(model_input)
                list_vlad.append(netvlad)
              netvlad = tf.add_n(list_vlad) / len(list_vlad)
            else:
              netvlad = netvlad_cls.forward(model_inputs[0])
            netvlad = make_fc(netvlad, 'netvlad', fc_netvlad_circulant)

        with tf.control_dependencies([netvlad]):
          with tf.variable_scope("Fisher_vector_{}".format(name), reuse=tf.AUTO_REUSE):
            netfv_cls = NetFV(size, max_frames, fv_cluster_size, 
              embedding_add_batch_norm, fv_couple_weights, fv_coupling_factor, 
              is_training)
            list_fv = []
            if len(model_inputs) > 1:
              for model_input in model_inputs:
                fv = netfv_cls.forward(model_input)
                list_fv.append(fv)
              fv = tf.add_n(list_fv) / len(list_fv)
            else:
              fv = netfv_cls.forward(model_inputs[0])
            fv = make_fc(fv, 'fv', fc_fisher_circulant)

      return dbof, netvlad, fv

    dbof_video, netvlad_video, fv_video = make_embedding(sample_model_inputs_video, 1024, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, 'video')
    dbof_audio, netvlad_audio, fv_audio = make_embedding(sample_model_inputs_audio, 128, 
      dbof_cluster_size // 2, netvlad_cluster_size // 2, fv_cluster_size // 2, 'audio')

    activation = tf.concat([dbof_video, netvlad_video, fv_video, 
                            dbof_audio, netvlad_audio, fv_audio], 1)

    with tf.variable_scope('merge_video_audio'):

      if not fc_circulant:
        activation_dim = activation.get_shape().as_list()[1] 
        fc_hidden_weights = tf.get_variable("fc_hidden_weights",
          [activation_dim, fc_hidden_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size)))
        activation = tf.matmul(activation, fc_hidden_weights)
      else:
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size))
        input_dim = activation.get_shape().as_list()
        circ_layer_hidden = CirculantLayerWithFactor(input_dim, fc_hidden_size, 
                    k_factor=k_factor, initializer=initializer)
        activation = circ_layer_hidden.matmul(activation)

      if fc_add_batch_norm:
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="fc_hidden_bn")
      else:
        fc_hidden_biases = tf.get_variable("fc_hidden_biases",
          [fc_hidden_size],
          initializer = tf.random_normal_initializer(stddev=0.01))
        tf.summary.histogram("fc_hidden_biases", fc_hidden_biases)
        activation += fc_hidden_biases
     
      activation = tf.nn.relu6(activation)
      tf.summary.histogram("full_hidden_output", activation)

    if fc_gating:
      with tf.variable_scope('gating_frame_level'):
        activation = context_gating(activation, fc_add_batch_norm, is_training)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=moe_add_batch_norm,
        **unused_params)




class EmbeddingPipeline(models.BaseModel):
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
                   embedding,
                   iterations=None,
                   sample_random_frames=None,
                   input_add_batch_norm=None,
                   n_bagging=None,
                   embedding_add_batch_norm=None,
                   dbof_cluster_size=None,
                   netvlad_cluster_size=None,
                   fv_cluster_size=None,
                   fv_couple_weights=None,
                   fv_coupling_factor=None,
                   fc_hidden_size=None,
                   fc_add_batch_norm=None,
                   fc_circulant=None,
                   fc_gating=None,
                   k_factor=None,
                   moe_add_batch_norm=None,
                   is_training=True,
                   **unused_params):

    iterations = iterations or FLAGS.iterations
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    input_add_batch_norm = input_add_batch_norm or FLAGS.input_add_batch_norm

    n_bagging = n_bagging or FLAGS.n_bagging
    
    embedding_add_batch_norm = embedding_add_batch_norm or FLAGS.embedding_add_batch_norm

    dbof_cluster_size = dbof_cluster_size or FLAGS.dbof_cluster_size
    netvlad_cluster_size = netvlad_cluster_size or FLAGS.netvlad_cluster_size
    fv_cluster_size = fv_cluster_size or FLAGS.fv_cluster_size
    fv_couple_weights = fv_couple_weights or FLAGS.fv_couple_weights
    fv_coupling_factor = fv_coupling_factor or FLAGS.fv_coupling_factor

    fc_add_batch_norm = fc_add_batch_norm or FLAGS.fc_add_batch_norm
    fc_hidden_size = fc_hidden_size or FLAGS.fc_hidden_size
    fc_circulant = fc_circulant or FLAGS.fc_circulant
    k_factor = k_factor or FLAGS.k_factor
    fc_gating = fc_gating or FLAGS.fc_gating

    moe_add_batch_norm = moe_add_batch_norm or FLAGS.moe_add_batch_norm


    def get_input(id_):
      num_frames_cast = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      if random_frames:
        sample_model_input = utils.SampleRandomFrames(model_input, num_frames_cast, iterations)
      else:
        sample_model_input = utils.SampleRandomSequence(model_input, num_frames_cast, iterations)
      max_frames = sample_model_input.get_shape().as_list()[1]
      feature_size = sample_model_input.get_shape().as_list()[2]
      reshaped_input = tf.reshape(sample_model_input, [-1, feature_size])
      tf.summary.histogram("input_hist_{}".format(id_), reshaped_input)
      if input_add_batch_norm:
        reshaped_input = slim.batch_norm(
            reshaped_input,
            center=True,
            scale=True,
            is_training=is_training,
            scope="input_bn_{}".format(id_))
      return reshaped_input, max_frames

    sample_model_inputs_video = []
    sample_model_inputs_audio = []
    for i in range(n_bagging):
      sample_model_input, max_frames = get_input(i)
      sample_model_inputs_video.append(sample_model_input[:, 0:1024])
      sample_model_inputs_audio.append(sample_model_input[:, 1024:])

    def make_embedding(model_inputs, size, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, name):
      with tf.variable_scope(name):

        if embedding == 'dbof':
          with tf.variable_scope("DBoF_{}".format(name), reuse=tf.AUTO_REUSE):
            dbof_cls = DBof(size, max_frames, dbof_cluster_size, 
              FLAGS.dbof_pooling_method, embedding_add_batch_norm, is_training)
            list_dbof = []
            if len(model_inputs) > 1:
              for model_input in model_inputs:
                dbof = dbof_cls.forward(model_input)
                list_dbof.append(dbof)
              ret = tf.add_n(list_dbof) / len(list_dbof)
            else:
              ret = dbof_cls.forward(model_inputs[0])

        if embedding == 'netvlad':
          with tf.variable_scope("NetVLAD{}".format(name), reuse=tf.AUTO_REUSE):
            netvlad_cls = NetVLAD(size, max_frames, netvlad_cluster_size, 
             embedding_add_batch_norm, is_training)
            list_vlad = []
            if len(model_inputs) > 1:
              for model_input in model_inputs:
                netvlad = netvlad_cls.forward(model_input)
                list_vlad.append(netvlad)
              ret = tf.add_n(list_vlad) / len(list_vlad)
            else:
              ret = netvlad_cls.forward(model_inputs[0])

        if embedding == 'fisher':
          with tf.variable_scope("Fisher_vector{}".format(name), reuse=tf.AUTO_REUSE):
            netfv_cls = NetFV(size, max_frames, fv_cluster_size, 
              embedding_add_batch_norm, fv_couple_weights, fv_coupling_factor, 
              is_training)
            list_fv = []
            if len(model_inputs) > 1:
              for model_input in model_inputs:
                fv = netfv_cls.forward(model_input)
                list_fv.append(fv)
              ret = tf.add_n(list_fv) / len(list_fv)
            else:
              ret = netfv_cls.forward(model_inputs[0])

      return ret

    video_embedding = make_embedding(sample_model_inputs_video, 1024, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, 'video')
    audio_embedding = make_embedding(sample_model_inputs_audio, 128, 
      dbof_cluster_size // 2, netvlad_cluster_size // 2, fv_cluster_size // 2, 'audio')

    activation = tf.concat([video_embedding, audio_embedding], 1)

    with tf.variable_scope('merge_video_audio'):

      if not fc_circulant:
        activation_dim = activation.get_shape().as_list()[1] 
        fc_hidden_weights = tf.get_variable("fc_hidden_weights",
          [activation_dim, fc_hidden_size],
          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size)))
        activation = tf.matmul(activation, fc_hidden_weights)
      else:
        initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size))
        input_dim = activation.get_shape().as_list()
        circ_layer_hidden = CirculantLayerWithFactor(input_dim, fc_hidden_size, 
                    k_factor=k_factor, initializer=initializer)
        activation = circ_layer_hidden.matmul(activation)

      if fc_add_batch_norm:
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=is_training,
            scope="fc_hidden_bn")
      else:
        fc_hidden_biases = tf.get_variable("fc_hidden_biases",
          [fc_hidden_size],
          initializer = tf.random_normal_initializer(stddev=0.01))
        tf.summary.histogram("fc_hidden_biases", fc_hidden_biases)
        activation += fc_hidden_biases
     
      activation = tf.nn.relu6(activation)
      tf.summary.histogram("full_hidden_output", activation)

    if fc_gating:
      with tf.variable_scope('gating_frame_level'):
        activation = context_gating(activation, fc_add_batch_norm, is_training)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=moe_add_batch_norm,
        **unused_params)

class EnsembleLateAverage(models.BaseModel):

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   **unused_params):
    
    n_embedding = 0
    total_preds = []
    for embedding in ['dbof', 'netvlad', 'fisher']:
      with tf.variable_scope(embedding):
        predictions = EmbeddingPipeline().create_model(
          model_input, vocab_size, num_frames, embedding)
        total_preds.append(predictions['predictions'])
      n_embedding += 1

    final = tf.add_n(total_preds) / n_embedding
    return {'predictions': final}


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
                   input_add_batch_norm=None,
                   dbof_add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   gating=None,
                   moe_add_batch_norm=None,
                   k_factor=None,
                   **unused_params):
    
    iterations = iterations or FLAGS.iterations
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    input_add_batch_norm = input_add_batch_norm or FLAGS.input_add_batch_norm
    dbof_add_batch_norm = dbof_add_batch_norm or FLAGS.dbof_add_batch_norm
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size
    gating = gating or FLAGS.dbof_gating
    k_factor = k_factor or FLAGS.k_factor
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

    with tf.variable_scope("video_DBoF"):
      video_DBoF = DBof(1024, max_frames, cluster_size, 
                      FLAGS.dbof_pooling_method, dbof_add_batch_norm, is_training)
      dbof_video = video_DBoF.forward(reshaped_input[:, 0:1024]) 

    with tf.variable_scope("audio_DBoF"):
      audio_DBoF = DBof(128, max_frames, cluster_size // 2, 
                        FLAGS.dbof_pooling_method, dbof_add_batch_norm, is_training)
      dbof_audio = audio_DBoF.forward(reshaped_input[:, 1024:])

    activation = tf.concat([dbof_video, dbof_audio], 1)

    with tf.variable_scope('circulant_fc_layer'):
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(hidden1_size))
      input_dim = activation.get_shape().as_list()
      circ_layer_hidden = CirculantLayerWithFactor(input_dim, hidden1_size, 
                  k_factor=k_factor, initializer=initializer)
      activation = circ_layer_hidden.matmul(activation)

    if dbof_add_batch_norm:
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
        activation = context_gating(activation, dbof_add_batch_norm, is_training)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=moe_add_batch_norm,
        **unused_params)

class Upsampling_CirculantWithFactor_DoubleDbofModel(models.BaseModel):
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
                   input_add_batch_norm=None,
                   dbof_add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   gating=None,
                   moe_add_batch_norm=None,
                   k_factor=None,
                   **unused_params):
    
    iterations = iterations or FLAGS.iterations
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    input_add_batch_norm = input_add_batch_norm or FLAGS.input_add_batch_norm
    dbof_add_batch_norm = dbof_add_batch_norm or FLAGS.dbof_add_batch_norm
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size
    gating = gating or FLAGS.dbof_gating
    k_factor = k_factor or FLAGS.k_factor
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

    with tf.variable_scope("video_DBoF"):
      video_DBoF = DBofCirculant(1024, max_frames, cluster_size, 
                      FLAGS.dbof_pooling_method, dbof_add_batch_norm, k_factor, is_training)
      dbof_video = video_DBoF.forward(reshaped_input[:, 0:1024]) 

    with tf.variable_scope("audio_DBoF"):
      audio_DBoF = DBofCirculant(128, max_frames, cluster_size // 2, 
                        FLAGS.dbof_pooling_method, dbof_add_batch_norm, k_factor, is_training)
      dbof_audio = audio_DBoF.forward(reshaped_input[:, 1024:])

    activation = tf.concat([dbof_video, dbof_audio], 1)

    dim = activation.get_shape().as_list()[1]
    hidden1_weights = tf.get_variable("hidden1_weights",
      [dim, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(dim)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if dbof_add_batch_norm:
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
        activation = context_gating(activation, dbof_add_batch_norm, is_training)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=moe_add_batch_norm,
        **unused_params)
