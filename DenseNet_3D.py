
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Dense, BatchNormalization, Concatenate, Dropout, AveragePooling3D, GlobalAveragePooling3D, Activation
from config import*

def bn_relu_conv(x, filters, kernel_size=(3, 3, 3)):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters=filters, kernel_size=kernel_size, padding='same')(x)

    return x

def dense_block(x):
    print('dense block')
    print(x.get_shape())

    for _ in range(DENSE_NET_BLOCK_LAYERS):
        y = x

        if DENSE_NET_ENABLE_BOTTLENETCK:
            y = bn_relu_conv(y, filters=DENSE_NET_GROWTH_RATE, kernel_size=(1, 1, 1))

        y = bn_relu_conv(y, filters=DENSE_NET_GROWTH_RATE, kernel_size=(3, 3, 3))
        x = Concatenate(axis=4)([x, y])
        print(x.get_shape())

    return x