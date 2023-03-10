
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf



##########---tf bilinear UpSampling3D
def up_sampling(input_tensor, scale):
    net = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D(size=(scale, scale), interpolation='bilinear'))(input_tensor)
    net = tf.keras.layers.Permute((2, 1, 3, 4))(net)  # (B, z, H, W, C) -> (B, H, z, w, c)
    net = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D(size=(scale, 1), interpolation='bilinear'))(net)
    net = tf.keras.layers.Permute((2, 1, 3, 4))(net)  # (B, z, H, W, C) -> (B, H, z, w, c)
    return net

#######-----Bottleneck