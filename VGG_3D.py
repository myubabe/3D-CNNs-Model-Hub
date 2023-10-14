
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Dense, GlobalMaxPooling3D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from config import *



def VGG3D(inputs,num_classes):
    inputs = inputs
    x = inputs