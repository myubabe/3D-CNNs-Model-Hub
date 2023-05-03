import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dense, GlobalMaxPooling3D, Dropout, BatchNormalization,Concatenate, AveragePooling3D, Activation
from tensorflow.keras.optimizers import Adam
from config import *

def conv_bn_relu(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same'):
    x = Conv3D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def inception_base(x):
    x = conv_bn_relu(x, filters=32)
    x = conv_bn_relu(x, filters=32)
    x = conv_bn_relu(x, filters=64)

    b0 = MaxPooling3D(pool_size=(2, 2, 2))(x)
    b1 = conv_bn_relu(x, 64, strides=(2, 2, 2))
    x = Concatenate(axis=4)([b0, b1])

    print('inception_base')
    print(b0.get_shape())
    print(b1.get_shape())
    print(x.get_shape())

    return x

def inception_block(x, filters=256):
    shrinkaged_filters = int(filters * INCEPTION_ENABLE_DEPTHWISE_SEPARABLE_CONV_SHRINKAGE)
    b0 = conv_bn_relu(x, filters=filters, kernel_size=(1, 1, 1))

    b1 = conv_bn_relu(x, filters=shrinkaged_filters, kernel_size=(1, 1, 1))
    b1 = conv_bn_relu(b1, filters=filters, kernel_size=(3, 3, 3))

    b2 = conv_bn_relu(x, filters=shrinkaged_filters, kernel_size=(1, 1, 1))
    b2 = conv_bn_relu(b2, filters=filters, kernel_size=(3, 3, 3))
    b2 = conv_bn_relu(b2, filters=filters, kernel_size=(3, 3, 3))

    b3 = AveragePooling3D(pool_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    b3 = conv_bn_relu(b3, filters=filters, kernel_size=(1, 1, 1))

    bs = [b0, b1, b2, b3]

    print('inception_bl