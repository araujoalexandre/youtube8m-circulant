
import math
import numpy as np

import model_utils as utils

import tensorflow as tf
import tensorflow.contrib.slim as slim

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

class CirculantLayerWithFactor:
    
  def __init__(self, input_shape, out_dim, k_factor=1, initializer=None):
    self.input_shape = input_shape
    self.out_dim = out_dim
    self.k_factor = k_factor
    self.initializer = initializer
    self.max_dim = input_shape[-1]
            
    dim = 0
    self.parameters = []
    count = 0
    while dim < self.out_dim:
      factor = []
      for k in range(self.k_factor):
        w = self._get_weights('weights_circ{}_f{}'.format(count, k))
        d = self._get_weights('weights_diag{}_f{}'.format(count, k))
        factor.append((w, d))
      count += 1
      self.parameters.append(factor)
      dim += self.max_dim

  def _get_weights(self, name=None):
    return tf.get_variable(
                name=name,
                shape=(1, self.max_dim),
                dtype=tf.float32,
                initializer=self.initializer,
                trainable=True)

  def matmul(self, X):
    mat = []
    batch_size = X.get_shape()[0]
    for params in self.parameters:
      ret = X
      for weights, diag in params:
        ret = tf.multiply(ret, diag)
        fft1 = tf.spectral.rfft(ret)
        fft2 = tf.spectral.rfft(weights[..., ::-1])
        fft_mul = tf.multiply(fft1, fft2)
        ifft_val = tf.spectral.irfft(fft_mul)
        ret = tf.cast(tf.real(ifft_val), tf.float32)
        ret = tf.manip.roll(ret, 1, axis=1)
      mat.append(ret)
    return tf.concat(mat, axis=1)[..., :self.out_dim]


class CirculantLayerWithDiag:
    
  def __init__(self, input_shape, out_dim, k_factor=1, initializer=None):
    self.input_shape = input_shape
    self.out_dim = out_dim
    self.k_factor = k_factor
    self.initializer = initializer
    self.max_dim = input_shape[-1]
            
    dim = 0
    self.parameters = []
    count = 0
    while dim < self.out_dim:
      factor = []
      for k in range(self.k_factor):
        w = self._get_weights('weights_circ{}_f{}'.format(count, k))
        d = tf.Variable(np.random.choice([-1, 1], size=(1, self.max_dim)), 
          dtype=tf.float32, trainable=False, name='weights_diag{}_f{}'.format(count, k))
        factor.append((w, d))
      count += 1
      self.parameters.append(factor)
      dim += self.max_dim

  def _get_weights(self, name=None):
    return tf.get_variable(
                name=name,
                shape=(1, self.max_dim),
                dtype=tf.float32,
                initializer=self.initializer,
                trainable=True)

  def matmul(self, X):
    mat = []
    batch_size = X.get_shape()[0]
    for params in self.parameters:
      ret = X
      for weights, diag in params:
        fft1 = tf.spectral.rfft(ret)
        fft2 = tf.spectral.rfft(weights[..., ::-1])
        fft_mul = tf.multiply(fft1, fft2)
        ifft_val = tf.spectral.irfft(fft_mul)
        ret = tf.cast(tf.real(ifft_val), tf.float32)
        ret = tf.multiply(ret, diag)
        ret = tf.manip.roll(ret, 1, axis=1)
      mat.append(ret)
    return tf.concat(mat, axis=1)[..., :self.out_dim]


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

    a_sum = tf.reduce_sum(activation,-2, keepdims=True)

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

class NetFV():
  
  def __init__(self, 
               feature_size, 
               max_frames, 
               cluster_size, 
               add_batch_norm, 
               fv_couple_weights,
               fv_coupling_factor,
               is_training):

    self.feature_size = feature_size
    self.max_frames = max_frames
    self.cluster_size = cluster_size
    self.add_batch_norm = add_batch_norm
    self.fv_couple_weights = fv_couple_weights
    self.fv_coupling_factor = fv_coupling_factor
    self.is_training = is_training

  def forward(self,reshaped_input):
    
    cluster_weights = tf.get_variable("cluster_weights",
      [self.feature_size, self.cluster_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
 
    covar_weights = tf.get_variable("covar_weights",
      [self.feature_size, self.cluster_size],
      initializer = tf.random_normal_initializer(mean=1.0, stddev=1 /math.sqrt(self.feature_size)))
  
    covar_weights = tf.square(covar_weights)
    eps = tf.constant([1e-6])
    covar_weights = tf.add(covar_weights, eps)

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
    
    activation = tf.nn.softmax(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

    a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

    if not self.fv_couple_weights:
        cluster_weights2 = tf.get_variable("cluster_weights2",
          [1,self.feature_size, self.cluster_size],
          initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
    else:
        cluster_weights2 = tf.scalar_mul(self.fv_coupling_factor, cluster_weights)

    a = tf.multiply(a_sum, cluster_weights2)
    
    activation = tf.transpose(activation, perm=[0,2,1])
    
    reshaped_input = tf.reshape(reshaped_input,[-1, self.max_frames, self.feature_size])
    fv1 = tf.matmul(activation, reshaped_input)
    
    fv1 = tf.transpose(fv1, perm=[0,2,1])

    # computing second order FV
    a2 = tf.multiply(a_sum, tf.square(cluster_weights2)) 

    b2 = tf.multiply(fv1, cluster_weights2) 
    fv2 = tf.matmul(activation, tf.square(reshaped_input)) 
 
    fv2 = tf.transpose(fv2, perm=[0,2,1])
    fv2 = tf.add_n([a2, fv2, tf.scalar_mul(-2, b2)])

    fv2 = tf.divide(fv2, tf.square(covar_weights))
    fv2 = tf.subtract(fv2, a_sum)

    fv2 = tf.reshape(fv2,[-1, self.cluster_size * self.feature_size])
  
    fv2 = tf.nn.l2_normalize(fv2, 1)
    fv2 = tf.reshape(fv2,[-1, self.cluster_size * self.feature_size])
    fv2 = tf.nn.l2_normalize(fv2, 1)

    fv1 = tf.subtract(fv1, a)
    fv1 = tf.divide(fv1, covar_weights) 

    fv1 = tf.nn.l2_normalize(fv1, 1)
    fv1 = tf.reshape(fv1,[-1, self.cluster_size * self.feature_size])
    fv1 = tf.nn.l2_normalize(fv1, 1)

    return tf.concat([fv1, fv2], 1)

class DBof():

  def __init__(self, feature_size, max_frames, cluster_size, 
    dbof_pooling_method, add_batch_norm, is_training, **kargs):
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


class SoftDBoF():

    def __init__(self, feature_size, max_frames, cluster_size, max_pool, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.max_pool = max_pool

    def forward(self, reshaped_input):

        feature_size = self.feature_size
        cluster_size = self.cluster_size
        add_batch_norm = self.add_batch_norm
        max_frames = self.max_frames
        is_training = self.is_training
        max_pool = self.max_pool

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

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])

        activation_sum = tf.reduce_sum(activation,1)
        activation_sum = tf.nn.l2_normalize(activation_sum,1)

        if max_pool:
            activation_max = tf.reduce_max(activation,1)
            activation_max = tf.nn.l2_normalize(activation_max,1)
            activation = tf.concat([activation_sum,activation_max],1)
        else:
            activation = activation_sum
        
        return activation

class LightVLAD():
    def __init__(self, feature_size,max_frames,cluster_size, add_batch_norm, is_training):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

    def forward(self,reshaped_input):


        cluster_weights = tf.get_variable("cluster_weights",
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
       
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
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases", cluster_biases)
          activation += cluster_biases
        
        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])
       
        activation = tf.transpose(activation,perm=[0,2,1])
        
        reshaped_input = tf.reshape(reshaped_input,[-1,self.max_frames,self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)
        
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        return vlad



class DBofCirculant():

  def __init__(self, feature_size, max_frames, cluster_size, 
    dbof_pooling_method, add_batch_norm, is_training, k_factor=1):
    self.feature_size = feature_size
    self.max_frames = max_frames
    self.cluster_size = cluster_size
    self.dbof_pooling_method = dbof_pooling_method
    self.add_batch_norm = add_batch_norm
    self.k_factor = k_factor
    self.is_training = is_training

  def forward(self, reshaped_input):

    initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.cluster_size))
    input_dim = reshaped_input.get_shape().as_list()
    circ_layer_hidden = CirculantLayerWithFactor(
                          (None, self.feature_size), 
                          self.cluster_size, 
                          k_factor=self.k_factor, 
                          initializer=initializer)
    activation = circ_layer_hidden.matmul(reshaped_input)

    # cluster_weights = tf.get_variable("cluster_weights",
    #   [self.feature_size, self.cluster_size],
    #   initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(self.cluster_size)))
    # tf.summary.histogram("cluster_weights", cluster_weights)
    # activation = tf.matmul(reshaped_input, cluster_weights)
    
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
