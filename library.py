'''library.py

A dumping ground for functions and networks we've defined.

List:

pc
    Numpy array with shape (*, 4). Sample pointcloud

quantize(x, a = - 3, b = 1, v = 0.4)
    Given a floating value x, quantize its value within
        range [a, b] with spacing v.

sample_points(
    pointcloud,
    D = [-3, 1], H = [-40, 40],  W = [0, 70.4], T = 35,
    vD = 0.4, vH = 0.2, vW = 0.2,
    flip_xyz = True, room_for_augmentation = True
)
    Given a pointcloud (i.e. array of (n, 4)  points)
        and quantization parameters D, H, W, T, vD, vH, vW,
        1. quantize according to the parameters D, H, W, vD, vH, vW,
        2. sample per-cell according to T, and
        3. Augment by adding the offset features (point - offset from mean)
        return the (D, H, W, T, 7) array of selected points.

augment_pc_with_offsets(voxelgrid):
    Given a voxel from sample_points(...),
    we want to augment each point in each cell
    with (z - z_mean, y - y_mean, x - x_mean),
    i.e. the offset from the centroid of the cell.
    Operates IN PLACE.

equi_hash(n, a = 0.618033988749895, K = 131072)
    Hash integer n to range [0, K) using Equidistribution Theorem Hash, using irrational a.

ravel_multi_index(
    index = (0, 0, 0),
    shape = (10, 400, 352)
):
    Flatten an index for a given range



VFE_FCN, ElementwiseMaxpool, PointwiseConcat, VFE, VFE_out, ConvMiddleLayer, RPNConvBlock


Quick import:

from library import pc, quantize, sample_points, augment_pc_with_offsets
from library import equi_hash, ravel_multi_index
from library import VFE_FCN, ElementwiseMaxpool, PointwiseConcat, VFE, VFE_out
from library import ConvMiddleLayer, RPNConvBlock
'''

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


# Define a pointcloud for us to use
with open('example_pc.bytes', 'rb') as ff:
    pc_bytes = ff.read()

pc = np.array(list(struct.iter_unpack('f', pc_bytes))).reshape(-1, 4)

#Part 1: Voxel Feature Encoder functions
def quantize(x, a = - 3, b = 1, v = 0.4):
    '''Given a floating value x, quantize its value within
    range [a, b] with spacing v.
    
    :param x: The value to quantize.
    :type x: float
    :param a: The left-bound of the range.
    :type a: float
    :param b: The right-bound of the range.
    :type b: float
    :param v: The size of each quantize
    :type v: float
    
    :return: The quantized index of x
    :rtype: int
    
    Examples:
    >>> quantize(x = -3.0, a = - 3, b = 1, v = 0.4)
    0
    >>> quantize(x =  1.0, a = - 3, b = 1, v = 0.4)
    9
    >>> quantize(x =  0.3, a = - 3, b = 1, v = 0.4)
    8
    '''
    return int((x - a) // v)

def sample_points(
    pointcloud,
    D = [-3, 1],
    H = [-40, 40], 
    W = [0, 70.4],
    T = 35,
    vD = 0.4,
    vH = 0.2,
    vW = 0.2,
    flip_xyz = True,
    room_for_augmentation = True
):
    '''Given a pointcloud (i.e. array of (n, 4)  points)
    and quantization parameters D, H, W, T, vD, vH, vW,
    1. quantize according to the parameters D, H, W, vD, vH, vW,
    2. sample per-cell according to T, and
    3. Augment by adding the offset features (point - offset from mean)
    return the (D, H, W, T, 7) array of selected points.
    
    :param pointcloud: Array of floats, size (n, 4) or (n, 3).
        Assuming (x, y, z).
    :type pointcloud: numpy.ndarray
    :param D: Range over Z axis, list of two floats
    :type D: list
    :param H: Range over Y axis, list of two floats
    :type H: list
    :param W: Range over X axis, list of two floats
    :type W: list
    :param T: Maximum number of points to sample
    :type T: int
    :param vD: Quantization size over Z axis
    :type vD: float
    :param vH: Quantization size over Y axis
    :type vH: float
    :param vW: Quantization size over X axis
    :type vW: float
    :param flip_xyz: If True, assume the input array is (x, y, z)
        rather than (z, y, x).
    :type flip_xyz: bool
    :param room_for_augmentation: If True, instantiage G with
        size 7 instead of 4, leaving room for augmentation
        described in 2.1.1. 
    :type flip_xyz: bool
    
    :return: Array of size (D, H, W, T, 4) (or 3)
        containing selection of points.
    :rtype: numpy.ndarray
    '''
    #pointcloud,
    #D = [-3, 1],H = [-40, 40],W = [0, 70.4],
    #T = 35,
    #vD = 0.4,vH = 0.2,vW = 0.2,
    #flip_xyz = True
    
    # Get the sizes of the grid
    Dsize = int((D[1] - D[0]) / vD)
    Hsize = int((H[1] - H[0]) / vH)
    Wsize = int((W[1] - W[0]) / vW)
    
    assert len(pointcloud.shape) == 2, "Pointcloud should be of shape (n, 4)"
    assert pointcloud.shape[1] == 4, \
        f"Expected points in 4 dimensions, not {pointcloud.shape[1]}"
    
    # 1 and 2: Define our arrays
    point_size = 7 if room_for_augmentation else 4
    G = np.zeros((Dsize, Hsize, Wsize, T, point_size))
    N = np.zeros((Dsize, Hsize, Wsize), dtype=np.int)
    
    # 3. Randomly shuffle incoming points P
    number_of_points = pointcloud.shape[0]
    random_indices = np.random.permutation(number_of_points)
    
    # 4. For each point in the pointcloud to add to the grid,
    for pc_index in random_indices:
        # 1. Get the indices `(i, j, k)`
        x, y, z, r = pointcloud[pc_index]
        # skip if any are out of bounds
        if not D[0] < z < D[1]:
            continue
        if not H[0] < y < H[1]:
            continue
        if not W[0] < x < W[1]:
            continue
        
        # note (i, j, k) corresponds to (Z, Y, X)
        i = quantize(z, a=D[0], b=D[1], v=vD)
        j = quantize(y, a=H[0], b=H[1], v=vH)
        k = quantize(x, a=W[0], b=W[1], v=vW)
        
        # 2. Get t and increase N
        t = N[i, j, k]
        N[i, j, k] += 1
        
        # 3 and 4 Skip if t >= T, add the point to G
        if t < T:
            G[i, j, k, t, :4] = z, y, x, r
        
    return G, N

def augment_pc_with_offsets(voxelgrid):
    '''Given a voxel from sample_points(...),
    we want to augment each point in each cell
    with (z - z_mean, y - y_mean, x - x_mean),
    i.e. the offset from the centroid of the cell.
    
    Operates IN PLACE.
    
    voxelgrid[:,:,:,:,4:7] should be empty.
    
    :param voxelgrid: Voxelgrid returned from sample_points,
        shape (D, H, W, T, 7)
    :type voxelgrid: numpy.ndarray
    '''
    
    voxelgrid[:,:,:,:,4:7] = \
        voxelgrid[:,:,:,:,:3] \
        - voxelgrid[:,:,:,:,:3].mean(axis=3)[:,:,:,None,:]
    
    # Operates in-place! Return is only for convenience.
    return voxelgrid

def equi_hash(n, a = 0.618033988749895, K = 131072):
    """Hash integer n to range [0, K) using Equidistribution Theorem Hash, using irrational a.

    Is optimal for a = Phi = 1.618034.
    Because of mod 1, we use 0.618034.

    :param n: Input to hash
    :type n: int
    :param a: Irrational number, defaults to 0.618033988749895
    :type a: float, optional
    :param K: [description], defaults to 131072
    :type K: int, optional

    :return: Result of the hash: An int in range [0, K)
    :rtype: int
    """
    return math.floor(((n * a) % 1)*K)

def ravel_multi_index(
    index = (0, 0, 0),
    shape = (10, 400, 352)
):
    """Flatten an index for a given range

    :param index: Multi-axis index, defaults to (0, 0, 0)
    :type index: tuple, optional
    :param shape: Range of the index, defaults to (10, 400, 352)
    :type shape: tuple, optional
    :return: Flat index for the given shape
    :rtype: int
    """
    assert len(index) == len(shape)
    flat_index = 0
    for axis in range(len(index)):
        flat_index = flat_index * shape[axis]
        flat_index = flat_index + index[axis]
    
    return flat_index


#Part 1.2. VFE neural layers
class VFE_FCN(keras.layers.Layer):
    '''The fully-connected layer used for the VFE.
    
    :param CC: Dimension of input
    :type CC: int
    :param name: Suffix of names to be given to layers
    :type name: str
    '''
    def __init__(self, units = 32, name = "VFE_FCN"):
        super(VFE_FCN, self).__init__(name=name)
        
        self.linear = keras.layers.Dense(units, name = f"{name}_linear")
        self.bn = keras.layers.BatchNormalization(name = f"{name}_bn")
        self.relu = keras.layers.ReLU(name = f"{name}_fcn")
    
    def call(self, xx):
        '''
        :param xx: Input tensor
        :type xx: Tensor
        '''
        # TODO: tflow can't autograph this. report on github
        return self.relu(self.bn(self.linear(xx)))

class ElementwiseMaxpool(keras.layers.Layer):
    '''
    :param axis: Axis to maxpool over
    :type axis: int
    :param keepdims: If true, reduced axes are not deleted.
    E.g. (3, 3, 35) over axis 2 becomes (3, 3, 1).
    :type keepdims: bool
    :param name: Suffix of names to be given to layers
    :type name: str
    '''
    def __init__(self, axis = 3, keepdims = True, name = "VFE_ElementwiseMaxpool"):
        super(ElementwiseMaxpool, self).__init__(name=name)
        self.axis = axis
        self.keepdims = keepdims
        
    def call(self, xx):
        '''
        :param xx: Input tensor to be maxpooled
        :type xx: Tensor
        :return: Output tensor after maxpooling operation
        :rtype: Tensor
        '''
        return tf.reduce_max(xx, axis = self.axis, keepdims = self.keepdims)

class PointwiseConcat(keras.layers.Layer):
    '''Given two inputs, broadcast the second over a given axis to match 
    the first, and then concatenate them.
    
    :param axis: Axis to repeat over
    :type axis: int
    :param expand_dims: If True, expand at the axis on the second input.
        Use False if the axis is already there (shape 1).
    :type expand_dims: bool
    :param name: Suffix of names to be given to layers
    :type name: str
    '''
    def __init__(self, axis = 4, expand_dims = False, name = "pointwise_concat_"):
        super(PointwiseConcat, self).__init__(name=name)
        self.axis = axis
        self.expand_dims = expand_dims
        
    def call(self, xx):
        '''
        :param xx: Input tensor
        :type xx: Tensor
        '''
        # 1. make sure shape makes sense
        X1 = xx[0] # pointwise
        X2 = xx[1] # aggregate
        
        if self.expand_dims:
            assert len(X1.shape) == len(X2.shape) + 1
        else:
            assert len(X1.shape) == len(X2.shape)
        
        # 2. Get num to repeat
        num_to_repeat = X1.shape[self.axis]
        
        # 3. expand_dims, repeat,
        if self.expand_dims:
            X2_wip = tf.expand_dims(X2, axis = self.axis)
        
        X2_wip = tf.repeat(X2, num_to_repeat, axis = self.axis)
        #todo: assert fails even though both are equal.
        #debug later...
        #assert X2_wip.shape == X1.shape, f"{X2_wip.shape} != {X1.shape}"
        
        return tf.concat([X1, X2_wip], axis = -1)

class VFE(keras.layers.Layer):
    '''
    :param cin: Dimensionality of input points
    :type cin: int
    :param cout:  Dimension of output
    :type cout: int  
    :param cout: Max number of points in voxel, needed for repeat
    :type cout: int  
    :param name: Suffix of names to be given to layers
    :type name: str  
    '''
    def __init__(
        self,
        cin = 7,
        cout = 32,
        T = 35,
        name = "VFE"
    ):
        super(VFE, self).__init__(name=name)
        self.fcn = VFE_FCN(units = cout//2, name = f"{name}_fcn")
        self.elementwise_maxpool = ElementwiseMaxpool(
            axis = 4,
            keepdims = True,
            name = f"{name}_elementwise_maxpool"
        )
        self.pointwise_concat = PointwiseConcat(
            axis = 4,
            name = f"{name}_pointwise_concat"
        )
        
    def call(
        self,
        xx
    ):
        '''
        :param xx: Input tensor
        :type xx: Tensor
        
        :return: Output of layer
        :rtype: Tensor
        '''
        pointwise_features = self.fcn(xx)
        aggregate_features = self.elementwise_maxpool(pointwise_features)
        
        return self.pointwise_concat([pointwise_features, aggregate_features])

class VFE_out(keras.layers.Layer):
    '''Connects VFE-n layer to the convolutional middle layers.
    
    :param CC: Dimension of input
    :type CC: int
    :param axis: Axis to maxpool over
    :type axis: int
    :param name: Suffix of names to be given to layers
    :type name: str
    '''
    def __init__(self, CC = 128, axis = 3, name = "VFE_out"):
        super(VFE_out, self).__init__(name=name)
        
        self.fcn = VFE_FCN(units=CC, name = f"{name}_fcn")
        self.elementwise_maxpool = ElementwiseMaxpool(
            axis = 4,
            keepdims = False,
            name = f"{name}_elementwise_maxpool"
        )
    
    def call(self, xx):
        '''
        :param xx: Input tensor
        :type xx: Tensor
        
        
        :return: Output of layer
        :rtype: Tensor
        '''
        return self.elementwise_maxpool(self.fcn(xx))



#Part 2. Convolutional middle layers
class ConvMiddleLayer(keras.layers.Layer):
    '''
    See section 2.1.2 "Convolutional Middle Layers" of the VoxelNet paper.
    
    For parameters kk, ss, and pp, these are vectors of size MM,
    but can be instantiated as a scalar.
    
    (E.g. for MM = 3 and kk = 2, this code changes it to kk = (2, 2, 2)
    
    :param MM: Dimension of the convolution, default 3
    :type MM: int
    :param c_in: Number of input channels
    :type c_in: int
    :param c_out: Number of output channels
    :type c_out: int
    :param kk: Kernel size, vector of size MM
    :type kk: tuple or int
    :param ss: Stride size, vector of size MM
    :type ss: tuple or int
    :param pp: Padding size, vector of size MM
    :type pp: tuple or int
    :param name: Suffix of names to be given to layers
    :type name: str
    '''
    
    def __init__(
        self,
        MM = 3,
        c_in = 64,
        c_out = 64,
        kk =  3,
        ss = (2, 1, 1),
        pp = (1, 1, 1),
        name = "ConvMiddleBlock"
    ):
        
        super(ConvMiddleLayer, self).__init__(name=name)

        # Check all values first
        # (todo: type hint, how to do iterables and union in 3.7?)
        # (todo: consider using pattern matching, if you're willing to break for old python version)

        if not isinstance(MM, int):
            raise TypeError("MM must be int")
        if not isinstance(c_in, int):
            raise TypeError("c_in must be int")
        if not isinstance(c_out, int):
            raise TypeError("c_out must be int")
        
        self.MM = MM
        self.c_in = c_in
        self.c_out = c_out

        if type(kk) is int:
            self.kk = [kk for _ in range(MM)]
        elif not hasattr(kk, '__iter__'):
            raise TypeError("kk must be int or iterable of ints")
        else:
            if not len(kk) == MM:
                raise ValueError("kk must be of length MM")
            for val in kk:
                if not isinstance(val, int):
                    raise TypeError("kk must be int or iterable of ints")
            self.kk = kk

        if type(ss) is int:
            self.ss = [ss for _ in range(MM)]
        elif not hasattr(ss, '__iter__'):
            raise TypeError("ss must be int or iterable of ints")
        else:
            if not len(ss) == MM:
                raise ValueError("ss must be of length MM")
            for val in ss:
                if not isinstance(val, int):
                    raise TypeError("ss must be int or iterable of ints")
            self.ss = ss

        if type(pp) is int:
            self.pp = [pp for _ in range(MM)]
        elif not hasattr(pp, '__iter__'):
            raise TypeError("pp must be int or iterable of ints")
        else:
            if not len(pp) == MM:
                raise ValueError("pp must be of length MM")
            for val in pp:
                if not isinstance(val, int):
                    raise TypeError("pp must be int or iterable of ints")
            self.pp = pp
        
        # TODO: define data format in padding and conv layers?
        if MM == 3:
            self.pad_layer  = keras.layers.ZeroPadding3D(
                padding = self.pp,
                name = f"{self.name}_padding"
            )
            self.conv_layer = keras.layers.Conv3D(
                filters     = self.c_out,
                kernel_size = self.kk,
                strides     = self.ss,
                padding     = "valid",  # TODO: Padding self.pp applies during call
                name        = f"{self.name}_conv"
            )
        elif MM == 2:
            self.pad_layer  = keras.layers.ZeroPadding2D(
                padding = self.pp,
                name = f"{self.name}_padding"
            )
            self.conv_layer = keras.layers.Conv2D(
                filters     = self.c_out,
                kernel_size = self.kk,
                strides     = self.ss,
                padding     = "valid",  # TODO: Padding self.pp applies during call
                name        = f"{self.name}_conv"
            )
        elif MM == 1:
            self.pad_layer  = keras.layers.ZeroPadding1D(
                padding = self.pp,
                name = f"{self.name}_padding"
            )
            self.conv_layer = keras.layers.Conv1D(
                filters     = self.c_out,
                kernel_size = self.kk,
                strides     = self.ss,
                padding     = "valid",  # TODO: Padding self.pp applies during call
                name        = f"{self.name}_conv"
            )
        elif MM == 4:
            raise NotImplementedError("4D convolutions not supported! Implement this in ConvMiddleLayer.")
        else:
            raise ValueError("Dimension MM must be 1, 2, or 3..")
        
        self.batchnorm_layer = keras.layers.BatchNormalization(
            name = f"{self.name}_batchnorm"
        )
        
        self.relu_layer = keras.layers.ReLU(
            name = f"{self.name}_relu"
        )
        
    def call(
        self,
        tf_input
    ):
        if not tf_input.shape[-1] == self.c_in:
            print(f"Warning, {self.name} expected input with {self.c_in} channels, got {tf_input.shape[-1]} instead.")
        
        conved = self.conv_layer(tf_input)
        h1 = self.pad_layer(conved)
        h2 = self.batchnorm_layer(h1)
        h3 = self.relu_layer(h2)
        
        return h3

#3. RPN block
class RPNConvBlock(keras.layers.Layer):
    '''
    See section 2.1.3 "Convolutional Middle Layers" of the VoxelNet paper.
    
    This defines a 2D convolution block used in the RPN.
    
    :param filters: Number of filters in each convolution block
    :type filters: int
    :param qq: Number of stacks of convolutions
    :type qq: int
    :param kernel_size:
    :type kernel_size:
    :param get_strides: A function mapping integers [0, ..., q) to integers.
        (By default, follows values for car detection. Use `lambda ii: 1` to return constant 1.)
    :type strides: function
    :param padding: 
    :type padding: 
    :param name: Suffix of names to be given to layers
    :type name: str
    '''
    
    def __init__(
        self,
        qq      = 6,
        filters = 128,
        kernel_size = 3,
        get_strides = lambda ii: 2 if ii == 0 else 1,
        padding = 1,
        name = "RPN_Conv_Block",
    ):
        
        super(RPNConvBlock, self).__init__(name=name)
        
        if qq <= 0:
            raise ValueError(f"Expected positive number of layers, but RPN conv block \"{name}\" got qq = {qq}")
        
        self.qq = qq
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.get_strides = get_strides
        self.block_layers = []
        
        for ii in range(self.qq):
            strides = self.get_strides(ii)
            
            new_layers = [
                keras.layers.ZeroPadding2D(
                    padding = self.padding,
                    name = f"{self.name}_zeropadding_{ii}"
                ),
                keras.layers.Conv2D(
                    filters = self.filters,
                    kernel_size = self.kernel_size,
                    strides = strides,
                    padding = 'valid',
                    name = f"{self.name}_conv_{ii}"   
                ),
                keras.layers.BatchNormalization(
                    name = f"{self.name}_bn_{ii}"
                ),
                keras.layers.ReLU(
                    name = f"{self.name}_relu_{ii}"
                )
            ]
            
            
            self.block_layers.extend(new_layers)
        
    def call(
        self,
        xx
    ):
        for layer in self.block_layers:
            xx = layer(xx)
        
        return xx

# TODO: Loss, training, etc.
# See nb102 for connecting conv, RPN layers, stacking.

