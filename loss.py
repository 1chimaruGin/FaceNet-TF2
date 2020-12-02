import math
import tensorflow as tf 
from tensorflow.keras import layers

def triplet_loss(gt, pred, alpha=0.2):
  anchor, pos, neg = pred[0], pred[1], pred[2]
  d_pos = tf.reduce_sum(tf.square(tf.subtract(anchor, pos)), axis=-1)
  d_neg = tf.reduce_sum(tf.square(tf.subtract(anchor, neg)), axis=-1)
  base_loss = d_pos - d_neg + alpha
  loss = tf.reduce_mean(tf.maximum(base_loss, 0))

  return loss

class ArcFaceSoftmaxLinear(layers.Layer):
  def __init__(self, units, input_dim, margin, feature_scale=64):
    super(ArcFaceSoftmaxLinear, self).__init__()
    self.m = margin  # m
    self.s = feature_scale
    self.cos_m = tf.math.cos(self.m)
    self.sin_m = tf.math.sin(self.m)
    self.threshold = tf.math.cos(math.pi-self.m)
    self.weight = self.add_weight(
      shape=(input_dim, units),
      nitializer='he_normal',
      trainable=True
    )

  def __call__(self, embedding, labels):
    x, w = embedding, self.weight
    w = tf.math.l2_normalize(w, axis=0)
    x = tf.math.l2_normalize(x, axis=1)
    logits = tf.matmul(x, w)
    indices_m = tf.expand_dims(tf.Variable(range(embedding.shape[0])), axis=1)
    indices_n = tf.expand_dims(labels, axis=1)
    indices = tf.concat([indices_m, indices_n], 1)
    selected_logits = tf.gather_nd(logits, indices)
    cos_theta = selected_logits
    sin_theta = tf.math.sqrt((1.0-tf.math.square(cos_theta)))
    logit_target = self.s * (cos_theta*self.cos_m-sin_theta*self.sin_m)
    keep_val = self.s*(cos_theta - self.m*self.sin_m)
    logit_target_updated = tf.where(cos_theta > self.threshold, logit_target, keep_val)
    logit_target_updated = self.s*(selected_logits-self.m)
    logits = tf.tensor_scatter_nd_update(logits, indices, logit_target_updated)
    return logits
