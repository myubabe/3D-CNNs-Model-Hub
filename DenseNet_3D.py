
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