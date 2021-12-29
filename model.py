import math
import os
import random
import sys
import struct
import warnings

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from library import pc, quantize, sample_points, augment_pc_with_offsets
from library import VFE_FCN, ElementwiseMaxpool, PointwiseConcat, VFE, VFE_out
from library import ConvMiddleLayer, RPNConvBlock


input_shape = (10, 400, 352, 35, 7)

# 1. Make VFE
vfe_input = keras.Input(shape=input_shape)
vfe_h1 = VFE(cin = 7, cout = 32, name="VFE_7-32")(vfe_input)
vfe_h2 = VFE(cin = 32, cout = 128, name="VFE_32-128")(vfe_h1)
vfe_out = VFE_out(CC = 128, axis = 4)(vfe_h2)

# 2. Conv middle layers
conv_1 = ConvMiddleLayer(
    MM = 3, c_in = 128, c_out = 64,
    kk = 3, ss = (2, 1, 1), pp = (1, 1, 1),
    name = "ConvMiddle_1_"
)(vfe_out)

conv_2 = ConvMiddleLayer(
    MM = 3, c_in = 64, c_out = 64,
    kk = 3, ss = (1, 1, 1), pp = (0, 1, 1),
    name = "ConvMiddle_2_"
)(conv_1)

conv_3 = ConvMiddleLayer(
    MM = 3, c_in = 64, c_out = 64,
    kk = 3, ss = (2, 1, 1), pp = (1, 1, 1),
    name = "ConvMiddle_3_"
)(conv_2)

# 3. RPN
# 3.1. Reshape previous layer
rpn_in_permute = tf.keras.layers.Permute((2, 3, 1, 4))(conv_3)

# reshape to: Get rid of the batch axis, multiply the last two
reshape_to = rpn_in_permute.shape[1:-2] + (rpn_in_permute.shape[-2] * rpn_in_permute.shape[-1],)
rpn_in_reshape = tf.keras.layers.Reshape(
    input_shape = rpn_in_permute.shape,
    target_shape = reshape_to
)(rpn_in_permute)

# 3.2. The main three conv blocks
rpn_block_1 = RPNConvBlock(qq = 4, name = 'rpn_block_1')(rpn_in_reshape)
rpn_block_2 = RPNConvBlock(qq = 6, name = 'rpn_block_2')(rpn_block_1)
rpn_block_3 = RPNConvBlock(qq = 6, name = 'rpn_block_3')(rpn_block_2)

# 3.3. Deconv blocks, to be concatenated...
rpn_deconv_1 = keras.layers.Conv2DTranspose(
    filters = 256, kernel_size = 3, strides = 1, padding = 'same', name = 'rpn_deconv_1'
)(rpn_block_1)
rpn_deconv_2 = keras.layers.Conv2DTranspose(
    filters = 256, kernel_size = 2, strides = 2, padding = 'same', name = 'rpn_deconv_2'
)(rpn_block_2)
rpn_deconv_3 = keras.layers.Conv2DTranspose(
    filters = 256, kernel_size = 4, strides = 4, padding = 'same', name = 'rpn_deconv_3'
)(rpn_block_3)

# 3.4. Concat andmap to two learning targets
deconv_layers = [rpn_deconv_1, rpn_deconv_2, rpn_deconv_3]
rpn_penultimate = keras.layers.Concatenate(name = 'rpn_penultimate')(deconv_layers)
probability_score_map = keras.layers.Convolution2D(
    filters = 2,
    kernel_size = 1,
    name = "RPN_prob_score_map",
)(rpn_penultimate)
regression_map = keras.layers.Convolution2D(
    filters = 14,
    kernel_size = 1
)(rpn_penultimate)

RPN_outputs = [probability_score_map, regression_map]

# 4. Ta-da! The model now.
voxelnet = keras.Model(
    inputs = vfe_input,
    outputs = RPN_outputs
)

