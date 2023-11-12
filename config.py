import tensorflow as tf
import math
###---Number-of-GPU

##-----Network Configuration----#####
NUMBER_OF_CLASSES=5
INPUT_PATCH_SIZE=(96,128,128, 1)
##------Resnet3D----####
TRAIN_NUM_RES_UNIT=3
TRAIN_NUM_FILTERS=(16, 32, 64, 128)
TRAIN_STRIDES=((1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2))
TRAIN_CLASSIFY_ACTICATION=tf.nn.relu6
TRAIN_KERNAL_INITIALIZER=tf.keras.initializers.VarianceScaling(distribution='uniform')
#-------DenseNet13D----#####
# DenseNet
DENSE_NET_BLOCKS = 3
DENSE_NET_BLOCK_LAYERS = 5
DENSE_NET_INITIAL_CONV_DIM = 16
DENSE_NET_GROWTH_RATE = DENSE_NET_INITIAL_CONV_DIM // 2
DENSE_NET_ENABLE_BOTTLENETCK = False # called DenseNet-BC if ENABLE_BOTTLENETCK and COMPRESSION < 1 in paper
DENSE_NET_TRANSITION_COMPRESSION = 1.0
DENSE_NET_ENABLE_DROPO