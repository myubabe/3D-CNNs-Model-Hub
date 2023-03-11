
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
def Bottleneck(x, nb_filter, increase_factor=4., weight_decay=1e-4):
    inter_channel = int(nb_filter * increase_factor)
    x = tf.keras.layers.Conv3D(inter_channel, (1, 1, 1),
                               kernel_initializer='he_normal',
                               padding='same',
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1.1e-5)(x)
    x = tf.nn.relu6(x)
    return x

#####------------>>> Convolutional Block
def conv_block(input, nb_filter, kernal_size=(3, 3, 3), dilation_rate=1,
                 bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu, 3x3X3 Conv3D, optional bottleneck block and dropout
    Args:
        input: Input tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: tensor with batch_norm, relu and convolution3D added (optional bottleneck)
    '''


    x = tf.keras.layers.BatchNormalization(epsilon=1.1e-5)(input)
    x = tf.nn.relu6(x)

    if bottleneck:
        inter_channel = nb_filter  # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua
        x = tf.keras.layers.Conv3D(inter_channel, (1, 1, 1),
                   kernel_initializer='he_normal',
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1.1e-5)(x)
        x = tf.nn.relu6(x)

    x = tf.keras.layers.Conv3D(nb_filter, kernal_size,
               dilation_rate=dilation_rate,
               kernel_initializer='he_normal',
               padding='same',
               use_bias=False)(x)
    if dropout_rate:
        x = tf.keras.layers.SpatialDropout3D(dropout_rate)(x)
    return x

##--------------------DenseBlock-------####
def dense_block(x, nb_layers, growth_rate, kernal_size=(3, 3, 3),
                  dilation_list=None,
                  bottleneck=True, dropout_rate=None, weight_decay=1e-4,
                  return_concat_list=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: input tensor