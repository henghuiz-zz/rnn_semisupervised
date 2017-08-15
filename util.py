import tensorflow as tf


def _mask_by_length(t, length):
  """Mask t, 3-D [batch, time, dim], by length, 1-D [batch,]."""
  maxlen = t.get_shape().as_list()[1]

  # Subtract 1 from length to prevent the perturbation from going on 'eos'
  mask = tf.sequence_mask(length - 1, maxlen=maxlen)
  mask = tf.expand_dims(tf.cast(mask, tf.float32), -1)
  # shape(mask) = (batch, num_timesteps, 1)
  return t * mask


def _scale_l2(x, norm_length):
  # shape(x) = (batch, num_timesteps, d)
  # Divide x by max(abs(x)) for a numerically stable L2 norm.
  # 2norm(x) = a * 2norm(x/a)
  # Scale over the full sequence, dims (1, 2)
  alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
  l2_norm = alpha * tf.sqrt(
      tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
  x_unit = x / l2_norm
  return norm_length * x_unit


def _kl_divergence_with_logits(q_logits, p_logits, num_classes):
  """Returns weighted KL divergence between distributions q and p.

  Args:
    q_logits: logits for 1st argument of KL divergence shape
              [num_timesteps * batch_size, num_classes] if num_classes > 2, and
              [num_timesteps * batch_size] if num_classes == 2.
    p_logits: logits for 2nd argument of KL divergence with same shape q_logits.
    weights: 1-D float tensor with shape [num_timesteps * batch_size].
             Elements should be 1.0 only on end of sequences

  Returns:
    KL: float scalar.
  """
  # For logistic regression
  if num_classes == 2:
    q = tf.nn.sigmoid(q_logits)
    kl = (-tf.nn.sigmoid_cross_entropy_with_logits(logits=q_logits, labels=q) +
          tf.nn.sigmoid_cross_entropy_with_logits(logits=p_logits, labels=q))
    kl = tf.squeeze(kl)

  # For softmax regression
  else:
    q = tf.nn.softmax(q_logits)
    kl = tf.reduce_sum(
        q * (tf.nn.log_softmax(q_logits) - tf.nn.log_softmax(p_logits)), 1)

  loss = tf.identity(tf.reduce_sum(kl), name='kl')
  return loss