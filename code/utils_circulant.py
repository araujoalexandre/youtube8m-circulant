
import tensorflow as tf

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
