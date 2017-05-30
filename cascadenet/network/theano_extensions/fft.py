from __future__ import absolute_import, print_function, division
import numpy as np
from theano import gof
import theano.tensor as T
from theano.gradient import DisconnectedType


class FFTOp(gof.Op):

    __props__ = ()

    def output_type(self, inp):
        # Assume input is already of the shape [n1, ..., nm, 2]
        return T.TensorType(inp.dtype,
                            broadcastable=[False] * (inp.type.ndim))

    def make_node(self, a, s=None):

        a = T.as_tensor_variable(a)
        if a.ndim < 3:
            raise TypeError('%s: input must have dimension >= 3,  with ' %
                            self.__class__.__name__ +
                            'first dimension batches and last real/imag parts')
        if s is None:
            s = a.shape[-2]
            s = T.as_tensor_variable(s)
        else:
            s = T.as_tensor_variable(s)
            if (not s.dtype.startswith('int')) and \
               (not s.dtype.startswith('uint')):
                raise TypeError('%s: length of the transformed axis must be'
                                ' of type integer' % self.__class__.__name__)
        return gof.Apply(self, [a, s], [self.output_type(a)()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        s = inputs[1]

        # combine the input into single complex map
        a_in = a[..., 0] + 1j * a[..., 1]
        A = np.fft.fft(a_in)
        # Format output with two extra dimensions for real and imaginary
        # parts.
        out = np.zeros(A.shape + (2,), dtype=a.dtype)
        out[..., 0], out[..., 1] = np.real(A), np.imag(A)
        output_storage[0][0] = out

    def grad(self, inputs, output_grads):
        # Gradient is just Inverse Fourier Transform
        gout, = output_grads
        s = inputs[1]
        # There are no symmetry present in complex-FFT because both real and imaginary part
        # interact to create distinct values every time.

        # idx = [slice(None)] * (gout.ndim - 2) \
        #     + [slice(1, (s[-1] // 2) + (s[-1] % 2))] + [slice(None)]
        # gout = T.set_subtensor(gout[idx], gout[idx] * 0.5)
        return [ifft_op(gout, s), DisconnectedType()()]

    def connection_pattern(self, node):
        # Specificy that shape input parameter has no connection to graph and gradients.
        return [[True], [False]]

fft_op = FFTOp()


class IFFTOp(gof.Op):

    __props__ = ()

    def output_type(self, inp):
        # remove extra dim for real/imag
        return T.TensorType(inp.dtype,
                            broadcastable=[False] * (inp.type.ndim))

    def make_node(self, a, s=None):
        a = T.as_tensor_variable(a)
        if a.ndim < 3:
            raise TypeError('%s: input must have dimension >= 3,  with ' %
                            self.__class__.__name__ +
                            'first dimension batches and last real/imag parts')

        if s is None:
            s = a.shape[-2]
            s = T.as_tensor_variable(s)
        else:
            s = T.as_tensor_variable(s)
            if (not s.dtype.startswith('int')) and \
               (not s.dtype.startswith('uint')):
                raise TypeError('%s: length of the transformed axis must be'
                                ' of type integer' % self.__class__.__name__)
        return gof.Apply(self, [a, s], [self.output_type(a)()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        s = inputs[1]

        # Reconstruct complex array from two float dimensions
        inp = a[..., 0] + 1j * a[..., 1]
        A = np.fft.ifft(inp) 
        # Remove numpy's default normalization (by n=s)
        # Cast to input type (numpy outputs float64 by default)
        out = np.zeros(A.shape + (2,), dtype=a.dtype)
        out[..., 0], out[..., 1] = np.real(A), np.imag(A)
        output_storage[0][0] = (out * s).astype(a.dtype)

    def grad(self, inputs, output_grads):
        gout, = output_grads
        s = inputs[1]
        gf = fft_op(gout, s)
        return [gf, DisconnectedType()()]

    def connection_pattern(self, node):
        # Specificy that shape input parameter has no connection to graph and gradients.
        return [[True], [False]]

ifft_op = IFFTOp()


def fft(inp, norm=None):
    """
    Performs the fast Fourier transform of a complex-valued input simulated by R^2.

    The input must be a real-valued variable of dimensions (m, ..., n, 2).
    It performs FFTs of size n along the last axis. 

    The output is a tensor of dimensions (m, ..., n, 2).
    The real and imaginary parts are stored as a pair of
    float arrays.

    Parameters
    ----------
    inp
        Array of floats of size (m, ..., n, 2)
    norm : {None, 'ortho', 'no_norm'}
        Normalization of transform. Following numpy, default *None* normalizes
        only the inverse transform by n, 'ortho' yields the unitary transform
        (:math:`1/\sqrt n` forward and inverse). In addition, 'no_norm' leaves
        the transform unnormalized.

    """

    s = inp.shape[-2]
    cond_norm = _unitary(norm)
    scaling = 1
    if cond_norm == "ortho":
        scaling = T.sqrt(s.astype(inp.dtype))

    return fft_op(inp, s) / scaling


def ifft(inp, norm=None):
    """
    Performs the inverse fast Fourier Transform with complex-valued input simulated by R^2.

    The input is a variable of dimensions (m, ..., n, 2)
    The real and imaginary parts are stored as a
    pair of float arrays.

    The output is a real-valued variable of dimensions (m, ..., n, 2)
    giving the inverse FFTs along the last axis.

    Parameters
    ----------
    inp
        Array of size (m, ..., n, 2), containing m inputs

    norm : {None, 'ortho', 'no_norm'}
        Normalization of transform. Following numpy, default *None* normalizes
        only the inverse transform by n, 'ortho' yields the unitary transform
        (:math:`1/\sqrt n` forward and inverse). In addition, 'no_norm' leaves
        the transform unnormalized.

    """

    s = inp.shape[-2]
    cond_norm = _unitary(norm)
    scaling = 1
    # Numpy's default normalization is 1/N on the inverse transform.
    if cond_norm is None:
        scaling = s.astype(inp.dtype)
    elif cond_norm == "ortho":
        scaling = T.sqrt(s.astype(inp.dtype))

    return ifft_op(inp, s) / scaling


def _unitary(norm):
    if norm not in (None, "ortho", "no_norm"):
        raise ValueError("Invalid value %s for norm, must be None, 'ortho' or "
                         "'no norm'" % norm)
    return norm
