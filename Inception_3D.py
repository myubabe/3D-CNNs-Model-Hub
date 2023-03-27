import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dense, GlobalMaxPooling3D, Dropout, BatchNormalization,Concatenate, AveragePooling3D, Activation
from tensorflow.keras.optimizers import Adam
from config import *

def conv_bn_relu(x, filters, kernel