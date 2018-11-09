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
from utils_layer import (CirculantLayer, CirculantLayerWithFactor, 
  CirculantLayerWithDiag, DBofCirculant)
from utils_layer import NetVLAD, DBof, NetFV, SoftDBoF, LightVLAD
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

flags.DEFINE_bool("input_add_batch_norm", False,
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
flags.DEFINE_bool("fv_couple_weights", False,
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
flags.DEFINE_bool("fc_gating", False,
                  "Activate Context Gating layer after FC from Ensemble Early Concat")


flags.DEFINE_bool('add_dbof', False, "add Embedding to model")
flags.DEFINE_bool('add_netvlad', False, "add Embedding to model")
flags.DEFINE_bool('add_fisher_vector', False, "add Embedding to model")
flags.DEFINE_bool('add_moments', False, "add Embedding to model")

flags.DEFINE_bool('fc_dbof_circulant', False, "Circulant FC after dbof")
flags.DEFINE_bool('fc_netvlad_circulant', False, "Circulant FC after NetVald")
flags.DEFINE_bool('fc_fisher_circulant', False, "Circulant FC after fisher")
flags.DEFINE_bool('fc_moment_circulant', False, "Circulant FC after Moment")

flags.DEFINE_bool('dbof_circulant', False, "Make the DBoF upsampling circulant")

flags.DEFINE_bool('no_audio', False, "remove audio for embeding")

flags.DEFINE_bool('use_d_matrix', False, "use D matrice {-1, +1} as diag")



class CirulantDiagonalNetwork(models.BaseModel):
  """Class to compare Compact vs Dense Deep Neural Network model.

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
                   fc_moment_circulant=None,
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
    fc_moment_circulant = fc_moment_circulant or FLAGS.fc_moment_circulant

    fc_add_batch_norm = fc_add_batch_norm or FLAGS.fc_add_batch_norm
    fc_hidden_size = fc_hidden_size or FLAGS.fc_hidden_size
    fc_circulant = fc_circulant or FLAGS.fc_circulant
    k_factor = k_factor or FLAGS.k_factor
    fc_gating = fc_gating or FLAGS.fc_gating

    moe_add_batch_norm = moe_add_batch_norm or FLAGS.moe_add_batch_norm


    def get_input():
      num_frames_cast = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
      if random_frames:
        sample_model_input = utils.SampleRandomFrames(model_input, num_frames_cast, iterations)
      else:
        sample_model_input = utils.SampleRandomSequence(model_input, num_frames_cast, iterations)
      max_frames = sample_model_input.get_shape().as_list()[1]
      feature_size = sample_model_input.get_shape().as_list()[2]
      reshaped_input = tf.reshape(sample_model_input, [-1, feature_size])
      tf.summary.histogram("input", reshaped_input)
      if input_add_batch_norm:
        reshaped_input = slim.batch_norm(
            reshaped_input,
            center=True,
            scale=True,
            is_training=is_training,
            scope="input_bn",
            reuse=tf.AUTO_REUSE)
      return reshaped_input, max_frames, feature_size

    sample_model_inputs_video = []
    sample_model_inputs_audio = []
    for i in range(n_bagging):
      sample_model_input, max_frames, feature_size = get_input()
      sample_model_inputs_video.append(sample_model_input[:, 0:1024])
      sample_model_inputs_audio.append(sample_model_input[:, 1024:])

    if FLAGS.use_d_matrix:
      CirculantLayer = CirculantLayerWithDiag
    else:
      CirculantLayer = CirculantLayerWithFactor


    def make_fc(input_, name, circulant, batch_norm):
      with tf.variable_scope(name):
        if not circulant:
          dim = input_.get_shape().as_list()[1] 
          fc_hidden_weights = tf.get_variable("{}_fc_hidden_weights".format(name),
            [dim, fc_hidden_size], initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size)))
          activation = tf.matmul(input_, fc_hidden_weights)
        else:
          input_dim = input_.get_shape().as_list()
          circ_layer_hidden = CirculantLayer(input_dim, fc_hidden_size, 
                      k_factor=k_factor, initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(fc_hidden_size)))
          activation = circ_layer_hidden.matmul(input_)

        if fc_add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=is_training,
              scope="{}_fc_hidden_bn".format(name))
        else:
          fc_hidden_biases = tf.get_variable("{}_fc_hidden_biases".format(name),
            [fc_hidden_size],
            initializer = tf.random_normal_initializer(stddev=0.01))
          tf.summary.histogram("fc_hidden_biases", fc_hidden_biases)
          activation += fc_hidden_biases
      activation = tf.nn.relu6(activation)
      return activation

    if FLAGS.dbof_circulant:
      DBof_ = DBofCirculant
    else:
      DBof_ = DBof

    def make_embedding(model_inputs, size, 
      dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, name):
      
      embeddings = []
      with tf.variable_scope(name):

        if FLAGS.add_dbof:
          with tf.variable_scope("DBoF_{}".format(name), reuse=tf.AUTO_REUSE):
            dbof_cls = DBof_(size, max_frames, dbof_cluster_size, 
              FLAGS.dbof_pooling_method, embedding_add_batch_norm, is_training, 
              k_factor=k_factor)
            list_dbof = []
            if len(model_inputs) > 1:
              for model_input in model_inputs:
                dbof = dbof_cls.forward(model_input)
                list_dbof.append(dbof)
              dbof = tf.add_n(list_dbof) / len(list_dbof)
            else:
              dbof = dbof_cls.forward(model_inputs[0])
            dbof = make_fc(dbof, 'dbof', fc_dbof_circulant, True)
          embeddings.append(dbof)

        if FLAGS.add_netvlad:
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
            netvlad = make_fc(netvlad, 'netvlad', fc_netvlad_circulant, True)
          embeddings.append(netvlad)

        if FLAGS.add_fisher_vector:
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
            fv = make_fc(fv, 'fv', fc_fisher_circulant, True)
          embeddings.append(fv)

        if FLAGS.add_moments:
          with tf.variable_scope("Moment_Stats_{}".format(name), reuse=tf.AUTO_REUSE):
            list_stat = []
            if len(model_inputs) > 1:
              for model_input in model_inputs:
                model_input = tf.reshape(model_input, [-1, max_frames, size])
                moment1, moment2 = tf.nn.moments(model_input, axes=1)
                moments = tf.concat([moment1, moment2], 1)
                list_stat.append(moments)
              moments = tf.add_n(list_stat) / len(list_stat)
            else:
              model_input = tf.reshape(model_input, [-1, max_frames, size])
              moment1, moment2 = tf.nn.moments(model_input, axes=1)
              moments = tf.concat([moment1, moment2], 1)
            moments = make_fc(moments, "moments", fc_moment_circulant, True)
          embeddings.append(moments)

      return embeddings


    if FLAGS.no_audio:
      embedding_video = make_embedding(sample_model_inputs_video, 1024, 
        dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, 'video')
      activation = tf.concat([*embedding_video], 1)
    else:
      embedding_video = make_embedding(sample_model_inputs_video, 1024, 
        dbof_cluster_size, netvlad_cluster_size, fv_cluster_size, 'video')
      embedding_audio = make_embedding(sample_model_inputs_audio, 128, 
        dbof_cluster_size // 2, netvlad_cluster_size // 2, fv_cluster_size // 2, 'audio')
      activation = tf.concat([*embedding_video, *embedding_audio], 1)


    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        is_training=is_training,
        add_batch_norm=moe_add_batch_norm,
        **unused_params)



