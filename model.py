import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Generator:
  def __init__(self, learning_rate=1e-4, num_blocks=6):
    self.learning_rate = learning_rate
    self.num_blocks = num_blocks

  def pelu(self, x):
    with tf.variable_scope(x.op.name + '_activation', initializer=tf.constant_initializer(1.0), reuse=tf.AUTO_REUSE):
        shape = x.get_shape().as_list()[1:]
        alpha = tf.get_variable('alpha', 1, constraint=lambda t: tf.maximum(t, 0.1))
        beta = tf.get_variable('beta', 1, constraint=lambda t: tf.maximum(t, 0.1))
        positive = tf.nn.relu(x) * alpha / (beta + 1e-9)
        negative = alpha * (tf.exp((-tf.nn.relu(-x)) / (beta + 1e-9)) - 1)
        return negative + positive

  def adaptive_global_average_pool_2d(self, x):
    c = x.get_shape()[-1]
    ADAP2d = tf.reshape(tf.reduce_mean(x, axis=[1, 2]), (-1, 1, 1, c))
    return ADAP2d

  def channel_attention(self, x, f, reduction):
    skip_conn = tf.identity(x, name='identity')

    x = self.adaptive_global_average_pool_2d(x)

    x = tf.layers.conv2d(x, kernel_size=1, filters=f//reduction, strides=1, padding='same')
    x = self.pelu(x)

    x = tf.layers.conv2d(x, kernel_size=1, filters=f, strides=1, padding='same')
    x = tf.nn.sigmoid(x)
    CA = tf.multiply(skip_conn, x)
    return CA

  def ResidualBlock(self, x, kernel_size, filters, strides=1):
    x = tf.layers.conv2d(x, kernel_size=1, filters=filters, strides=1, padding='same')
    skip = x
    x1 = x
    for i in range(3):
      tm1 = slim.conv2d(x1, num_outputs=filters, kernel_size=[3, 3], stride=1)
      tm1 = self.pelu(tm1)
      tm1 = slim.conv2d(tm1, num_outputs=filters, kernel_size=[1, 1], stride=1)
      tm1 = self.pelu(tm1)
      tm1 = slim.conv2d(tm1, num_outputs=filters, kernel_size=[1, 1], stride=1)
      tm1 = self.channel_attention(tm1, f=filters, reduction=4)
      x1 = tf.concat([x1,tm1], axis=3)
    
    x2 = x
    for i in range(3):
      tm2 = slim.conv2d(x2, num_outputs=filters, kernel_size=[3, 3], stride=1)
      tm2 = self.pelu(tm2)
      tm2 = slim.conv2d(tm2, num_outputs=filters, kernel_size=[1, 1], stride=1)
      tm2 = self.pelu(tm2)
      tm2 = slim.conv2d(tm2, num_outputs=filters, kernel_size=[1, 1], stride=1)
      tm2 = self.channel_attention(tm2, f=filters, reduction=4)
      x2 = tf.concat([x2,tm2], axis=3)
    
    x3 = x
    for i in range(3):
      tm3 = slim.conv2d(x3, num_outputs=filters, kernel_size=[3, 3], stride=1)
      tm3 = self.pelu(tm3)
      tm3 = slim.conv2d(tm3, num_outputs=filters, kernel_size=[1, 1], stride=1)
      tm3 = self.pelu(tm3)
      tm3 = slim.conv2d(tm3, num_outputs=filters, kernel_size=[1, 1], stride=1)
      tm3 = self.channel_attention(tm3, f=filters, reduction=4)
      x3 = tf.concat([x3,tm3], axis=3)
    
    x5 = tf.concat(values=[x1, x2, x3], axis=3, name='stack0')
    x6 = tf.layers.conv2d(x5, kernel_size=1, filters=filters, strides=strides, padding='same', use_bias=False)
    x7 = skip + x6
    return x7

  def Upsample2xBlock(self, x, kernel_size, filters, strides):
    #size = tf.shape(x)
    #h = size[1]
    #w = size[2]
    #x = tf.image.resize_nearest_neighbor(x, size=[h * 3, w * 3], align_corners=False, name=None)
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same')
    x = tf.depth_to_space(x, 2)
    x = self.pelu(x)
    return x

  def ThermalSR(self, x, reuse=False, isTraining=True):
    with tf.variable_scope("ThermalSR", reuse=reuse) as scope:
      x4 = tf.layers.conv2d(x, kernel_size=7, filters=64, strides=1, padding='same')
      x4 = self.pelu(x4)
      skip = x4

      # Global Residual Learning
      size = tf.shape(x)
      h = size[1]
      w = size[2]
      x_GRL = tf.image.resize_bicubic(x, size=[h * 4, w * 4], align_corners=False, name=None)
      x_GRL = tf.layers.conv2d(x_GRL, kernel_size=1, filters=64, strides=1, padding='same')
      x_GRL = self.pelu(x_GRL)
      x_GRL = tf.layers.conv2d(x_GRL, kernel_size=1, filters=16, strides=1, padding='same')
      x_GRL = self.pelu(x_GRL)
      x_GRL = tf.layers.conv2d(x_GRL, kernel_size=1, filters=3, strides=1, padding='same')
      x_GRL = self.pelu(x_GRL)

      for i in range(4):
        x4 = self.ResidualBlock(x4, kernel_size=1, filters=64, strides=1)
        x4 = tf.layers.conv2d(x4, kernel_size=1, filters=64, strides=1, padding='same', use_bias=False)
        x4 = self.pelu(x4)
        x4 = tf.concat([x4, skip], axis=3)

      x4 = tf.layers.conv2d(x4, kernel_size=3, filters=64, strides=1, padding='same', use_bias=False)
      x4 = self.pelu(x4)
      x4 = x4 + skip

      with tf.variable_scope('Upsamplingconv_stage_1'):
        xUP = self.Upsample2xBlock(x4, kernel_size=3, filters=64, strides = 1)
        xUP = self.Upsample2xBlock(xUP, kernel_size=3, filters=64, strides = 1)
        
      xUP = tf.layers.conv2d(xUP, kernel_size=1, filters=64, strides=1, padding='same', use_bias=False)
      xUP = self.pelu(xUP)
      skip1 = xUP 
      for i in range(2):
        x5 = self.ResidualBlock(xUP, kernel_size=1, filters=32, strides=1)
        x5 = tf.layers.conv2d(x5, kernel_size=1, filters=32, strides=1, padding='same', use_bias=False)
        x5 = self.pelu(x5)
        x5 = tf.concat([x5, skip1], axis=3)

      x5 = tf.layers.conv2d(x5, kernel_size=3, filters=64, strides=1, padding='same', use_bias=False)
      x5 = self.pelu(x5)
      x5 = x5 + skip1

      with tf.variable_scope('Upsamplingconv_stage_2'):
        x6 = self.Upsample2xBlock(x5, kernel_size=3, filters=64, strides = 1)
        
      x6 = tf.layers.conv2d(x6, kernel_size=3, filters=64, strides=1, padding='same', name='forward_4')
      x6 = self.pelu(x6)
      x6 = tf.layers.conv2d(x6, kernel_size=3, filters=3, strides=1, padding='same', name='forward_5')
      x6 = self.pelu(x6)
      x_final = x6 + x_GRL
      return x_final

