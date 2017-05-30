"""
Discrete Fourier Transforms - helper.py

"""
from __future__ import division, absolute_import, print_function

import numpy as np
from theano import gof
import theano.tensor as T
from theano.gradient import DisconnectedType


class FFTSHIFTOp(gof.Op):
    __props__ = ()

    def output_type(self, inp):
        return T.TensorType(inp.dtype,
                            broadcastable=[False] * (inp.type.ndim))

    def make_node(self, x, axes=None):

        x = T.as_tensor_variable(x)
        if x.ndim < 2:
            raise TypeError('%s: input must have dimension >= 2. For example,' %
                            self.__class__.__name__ +
                            '(n_batches, 2, nx, ny[, nt])')
        if axes is None:
            axes = list(range(x.ndim))
        elif isinstance(axes, int):
            axes = (axes,)

        axes = T.as_tensor_variable(axes)
        if (not axes.dtype.startswith('int')) and \
           (not axes.dtype.startswith('uint')):
                raise TypeError('%s: length of the transformed axis must be'
                                ' of type integer' % self.__class__.__name__)
        return gof.Apply(self, [x, axes], [self.output_type(x)()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        axes = inputs[1]
        # combine the input into single complex map
        out = np.fft.fftshift(x, axes)
        output_storage[0][0] = out

    def grad(self, inputs, output_grads):
        # Gradient is just Inverse Fourier Transform
        gout, = output_grads
        s = inputs[1]
        return [ifftshift_op(gout, s), DisconnectedType()()]

    def connection_pattern(self, node):
        # Specificy that shape input parameter has no connection to graph and
        # gradients.
        return [[True], [False]]

fftshift_op = FFTSHIFTOp()


class IFFTSHIFTOp(gof.Op):

    __props__ = ()

    def output_type(self, inp):
        return T.TensorType(inp.dtype,
                            broadcastable=[False] * (inp.type.ndim))

    def make_node(self, x, axes=None):

        x = T.as_tensor_variable(x)
        if x.ndim < 2:
            raise TypeError('%s: input must have dimension >= 2. For example' %
                            self.__class__.__name__ +
                            '(n_batches, 2, nx, ny[, nt])')
    
        if axes is None:
            axes = list(range(x.ndim))
        elif isinstance(axes, int):
            axes = (axes,)

        axes = T.as_tensor_variable(axes)
        if (not axes.dtype.startswith('int')) and \
           (not axes.dtype.startswith('uint')):
                raise TypeError('%s: length of the transformed axis must be'
                                ' of type integer' % self.__class__.__name__)
        return gof.Apply(self, [x, axes], [self.output_type(x)()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        axes = inputs[1]
        # combine the input into single complex map
        out = np.fft.ifftshift(x, axes)
        output_storage[0][0] = out

    def grad(self, inputs, output_grads):
        # Gradient is just Inverse Fourier Transform
        gout, = output_grads
        s = inputs[1]
        return [fftshift_op(gout, s), DisconnectedType()()]

    def connection_pattern(self, node):
        # Specificy that shape input parameter has no connection to graph and
        # gradients.
        return [[True], [False]]

ifftshift_op = IFFTSHIFTOp()


def fftshift(x, axes=None):
    """
    Performs np.fft.fftshift. Gradient is implemented as ifftshift

    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or shape tuple, optional
        Axes over which to calculate.  Defaults to None, which shifts all axes.

    Returns
    -------
    y : ndarray
        The shifted array.

    """
    return fftshift_op(x, axes)


def ifftshift(x, axes=None):
    """
    Performs np.fft.ifftshift. Gradient is implemented as fftshift

    Parameters
    ----------
    x : array_like
        Input array.
    axes : int or shape tuple, optional
        Axes over which to calculate.  Defaults to None, which shifts all axes.

    Returns
    -------
    y : ndarray
        The shifted array.

    """
    return ifftshift_op(x, axes)
