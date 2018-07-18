
import math

import model_utils as utils

import tensorflow as tf
import tensorflow.contrib.slim as slim


class CirculantLayer:
    
  def __init__(self, input_shape, out_dim, initializer=None):
    self.input_shape = input_shape
    self.out_dim = out_dim
    self.initializer = initializer or \
                    tf.random_normal_initializer(stddev=1./out_dim)

    self.max_dim_circ_vec = input_shape[-1]
        
    if self.out_dim <= self.max_dim_circ_vec:
      self.w = self._get_weights('weights')
    else:
      dim = 0
      self.ws = []
      count = 0
      while dim < self.out_dim:
        self.ws.append(self._get_weights('weights{}'.format(count)))
        dim += self.max_dim_circ_vec
        count += 1

  @property
  def weights(self):
    if self.out_dim <= self.max_dim_circ_vec:
      yield self.w
    else:
      for w in self.ws:
        yield w

  def _get_weights(self, name=None):
    return tf.get_variable(
                name=name,
                shape=(1, self.max_dim_circ_vec),
                dtype=tf.float32,
                initializer=self.initializer,
                trainable=True,
            )

  def _matmul(self, X_fft, w):
      w = tf.cast(w, tf.complex64)
      fft_w = tf.fft(w[..., ::-1])
      fft_mul = tf.multiply(X_fft, fft_w)
      ifft_val = tf.ifft(fft_mul)
      mat = tf.cast(tf.real(ifft_val), tf.float32)
      mat = tf.manip.roll(mat, shift=1, axis=1)
      return mat
    
  def matmul(self, X):
    """
      X @ W
    """
    X = tf.cast(X, tf.complex64)
    X_fft = tf.fft(X)
    if self.out_dim <= self.max_dim_circ_vec:
      ret = self._matmul(X_fft, self.w)
      return ret[..., :self.out_dim]
    else:
      mat = []
      for w in self.ws:
        ret = self._matmul(X_fft, w)
        mat.append(ret)
      return tf.concat(mat, axis=1)[..., :self.out_dim]


def context_gating(input_layer, add_batch_norm=None, is_training=True):
  """Context Gating
     https://github.com/antoine77340/LOUPE/blob/master/loupe.py#L59

    Args:
      input_layer: Input layer in the following shape:
      'batch_size' x 'number_of_activation'
    Returns:
      activation: gated layer in the following shape:
      'batch_size' x 'number_of_activation'
  """
  input_dim = input_layer.get_shape().as_list()[1] 
  
  gating_weights = tf.get_variable("gating_weights",
    [input_dim, input_dim],
    initializer = tf.random_normal_initializer(
    stddev=1 / math.sqrt(input_dim)))
  
  gates = tf.matmul(input_layer, gating_weights)

  if add_batch_norm:
    gates = slim.batch_norm(
        gates,
        center=True,
        scale=True,
        is_training=is_training,
        scope="gating_bn")
  else:
    gating_biases = tf.get_variable("gating_biases", [input_dim],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(input_dim)))
    gates += gating_biases

  gates = tf.nn.sigmoid(gates)
  activation = tf.multiply(input_layer, gates)
  return activation

class NetVLAD():
  """
    Thanks to Willow for this NetVLAD Class
    Taken from https://github.com/antoine77340/Youtube-8M-WILLOW
  """
  def __init__(self, feature_size, max_frames, cluster_size, 
              add_batch_norm, is_training):
    self.feature_size = feature_size
    self.max_frames = max_frames
    self.is_training = is_training
    self.add_batch_norm = add_batch_norm
    self.cluster_size = cluster_size

  def forward(self, reshaped_input):

    cluster_weights = tf.get_variable("cluster_weights",
          [self.feature_size, self.cluster_size],
          initializer = tf.random_normal_initializer(
            stddev=1 / math.sqrt(self.feature_size)))
     
    tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    
    if self.add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=self.is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases", [self.cluster_size],
        initializer = tf.random_normal_initializer(
          stddev=1 / math.sqrt(self.feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
  
    activation = tf.nn.softmax(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

    a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

    cluster_weights2 = tf.get_variable("cluster_weights2",
        [1, self.feature_size, self.cluster_size],
        initializer = tf.random_normal_initializer(
          stddev=1 / math.sqrt(self.feature_size)))
    
    a = tf.multiply(a_sum, cluster_weights2)
    
    activation = tf.transpose(activation, perm=[0, 2, 1])
    
    reshaped_input = tf.reshape(reshaped_input, 
      [-1, self.max_frames, self.feature_size])
    vlad = tf.matmul(activation, reshaped_input)
    vlad = tf.transpose(vlad, perm=[0,2,1])
    vlad = tf.subtract(vlad, a)
    vlad = tf.nn.l2_normalize(vlad, 1)
    vlad = tf.reshape(vlad,[-1, self.cluster_size * self.feature_size])
    vlad = tf.nn.l2_normalize(vlad,1)

    return vlad


class DBof():

  def __init__(self, feature_size, max_frames, cluster_size, dbof_pooling_method,
              add_batch_norm, is_training):
    self.feature_size = feature_size
    self.max_frames = max_frames
    self.cluster_size = cluster_size
    self.dbof_pooling_method = dbof_pooling_method
    self.add_batch_norm = add_batch_norm
    self.is_training = is_training

  def forward(self, reshaped_input):

    cluster_weights = tf.get_variable("cluster_weights",
      [self.feature_size, self.cluster_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
    tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    
    if self.add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=self.is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
        [self.cluster_size],
        initializer = tf.random_normal(stddev=1 / math.sqrt(self.feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])
    activation = utils.FramePooling(activation, self.dbof_pooling_method)

    return activation
