
from __future__ import absolute_import, print_function, division

import numpy as np
import theano
from theano import Op
import theano.tensor as T
from theano.gradient import DisconnectedType

from theano.gpuarray import (basic_ops, GpuArrayType)

import theano.tensor.fft
import cascadenet.network.theano_extensions.fft2
from theano.gpuarray.opt import register_opt, op_lifter, register_opt2

try:
    import pygpu
    pygpu_available = True
except ImportError:
    pygpu_available = False

try:
    import pycuda.driver
    pycuda_available = True
except ImportError:
    pycuda_available = False

try:
    import skcuda
    from skcuda import fft
    scikits_cuda_available = True
except (ImportError, Exception):
    scikits_cuda_available = False


class CuFFT2Op(Op):

    __props__ = ()

    def output_type(self, inp):
    
        return GpuArrayType(inp.dtype,
                            broadcastable=[False] * (inp.type.ndim),
                            context_name=inp.type.context_name)

    def make_node(self, inp, s=None):
        # A shape parameter s can be provided as an input. For now this is used to
        # manage odd transform sizes.
        # Later this could be extended to handle padding and trunkation,
        # following numpy's interface. However, cuFFT expects array that match
        # the shape given to the plan, so padding will have to be done in the op.
        # The effect of padding on gradients has yet to be investigated.

        if not scikits_cuda_available:
            raise RuntimeError("skcuda is needed for CuFFTOp")

        if not pygpu_available:
            raise RuntimeError("pygpu is needed for CuFFTOp")

        if not pycuda_available:
            raise RuntimeError("pycuda is needed for CuFFTOp")

        inp = basic_ops.gpu_contiguous(
            basic_ops.as_gpuarray_variable(inp,
                                           basic_ops.infer_context_name(inp)))

        # If no shape is provided as input, default to input data shape.
        if s is None:
            s = inp.shape[-3:-1]
            # s = inp.shape[1:]
        s = T.as_tensor_variable(s)

        assert inp.dtype == "float32"
        assert s.ndim == 1
        assert 'int' in s.dtype

        return theano.Apply(self, [inp, s], [self.output_type(inp)()])

    def make_thunk(self, node, storage_map, _, _2, impl=None):

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        # Initiliaze cuda context to the input's.
        with node.inputs[0].type.context:
            skcuda.misc.init()

        plan_input_shape = [None]
        plan = [None]

        def thunk():
            input_shape = inputs[0][0].shape
            s = inputs[1][0]

            # Since padding is not supported, assert s matches input shape.
            # assert (input_shape[1:-1] == s).all()
            assert (input_shape[-3:-1] == s).all()
            output_shape = input_shape

            z = outputs[0]

            # only allocate if there is no previous allocation of the
            # right size.
            if z[0] is None or z[0].shape != output_shape:
                z[0] = pygpu.zeros(output_shape, context=inputs[0][0].context,
                                   dtype='float32')

            input_pycuda = inputs[0][0]
            output_pycuda = z[0]

            with input_pycuda.context:
                # only initialise plan if necessary
                if plan[0] is None or plan_input_shape[0] != input_shape:
                    plan_input_shape[0] = input_shape
                    plan[0] = fft.Plan(s, np.complex64, np.complex64,
                                       batch=np.prod(input_shape[:-3]))

                # Sync GPU variables before computation
                input_pycuda.sync()
                output_pycuda.sync()

                fft.fft(input_pycuda, output_pycuda, plan[0])

                # Sync results to ensure output contains completed computation
                pycuda.driver.Context.synchronize()

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False
        
        return thunk

    def grad(self, inputs, output_grads):
        gout, = output_grads
        s = inputs[1]
        # Divide the last dimension of the output gradients by 2, they are
        # double-counted by the real-IFFT due to symmetry, except the first
        # and last elements (for even transforms) which are unique.
        # idx = [slice(None)] * (gout.ndim - 2) \
        #     + [slice(1, (s[-1] // 2) + (s[-1] % 2))] + [slice(None)]
        # gout = T.set_subtensor(gout[idx], gout[idx] * 0.5)
        return [cuifft2_op(gout, s), DisconnectedType()()]

    def connection_pattern(self, node):
        # Specificy that shape input parameter has no connection to graph and gradients.
        return [[True], [False]]

cufft2_op = CuFFT2Op()


class CuIFFT2Op(Op):

    __props__ = ()

    def output_type(self, inp):
        # remove extra dim for real/imag
        return GpuArrayType(inp.dtype,
                            broadcastable=[False] * (inp.type.ndim),
                            context_name=inp.type.context_name)

    def make_node(self, inp, s=None):
        # A shape parameter is expected as an input. For now this is used to
        # manage odd transform sizes.
        # Later this could be extended to handle padding and trunkation,
        # following numpy's interface. However, cuFFT expects array that match
        # the shape given to the plan, so padding will have to be done in the op.
        # The effect of padding on gradients has yet to be investigated.

        if not scikits_cuda_available:
            raise RuntimeError("skcuda is needed for CuIFFTOp")

        if not pygpu_available:
            raise RuntimeError("pygpu is needed for CuIFFTOp")

        if not pycuda_available:
            raise RuntimeError("pycuda is needed for CuIFFTOp")

        inp = basic_ops.gpu_contiguous(
            basic_ops.as_gpuarray_variable(inp,
                                           basic_ops.infer_context_name(inp)))

        # If no shape is provided as input, calculate shape assuming even real transform.
        if s is None:
            s = inp.shape[-3:-1]
        s = T.as_tensor_variable(s)

        assert inp.dtype == "float32"
        assert s.ndim == 1

        return theano.Apply(self, [inp, s], [self.output_type(inp)()])

    def make_thunk(self, node, storage_map, _, _2, impl=None):

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        # Initiliaze cuda context to the input's.
        with node.inputs[0].type.context:
            skcuda.misc.init()

        plan_input_shape = [None]
        plan = [None]

        def thunk():
            input_shape = inputs[0][0].shape
            s = inputs[1][0]
            output_shape = input_shape

            z = outputs[0]

            # only allocate if there is no previous allocation of the
            # right size.
            if z[0] is None or z[0].shape != output_shape:
                z[0] = pygpu.zeros(output_shape, context=inputs[0][0].context,
                                   dtype='float32')

            input_pycuda = inputs[0][0]
            # input_pycuda is a float32 array with an extra dimension,
            # but will be interpreted by skcuda as a complex64
            # array instead.
            output_pycuda = z[0]

            with input_pycuda.context:
                # only initialise plan if necessary
                if plan[0] is None or plan_input_shape[0] != input_shape:
                    plan_input_shape[0] = input_shape
                    plan[0] = fft.Plan(s, np.complex64, np.complex64,
                                       batch=np.prod(input_shape[:-3]))

                # Sync GPU variables before computation
                input_pycuda.sync()
                output_pycuda.sync()

                fft.ifft(input_pycuda, output_pycuda, plan[0])
                # strangely enough, enabling rescaling here makes it run
                # very, very slowly, so do this rescaling manually
                # afterwards!

                # Sync results to ensure output contains completed computation
                pycuda.driver.Context.synchronize()

        thunk.inputs = inputs
        thunk.outputs = outputs
        thunk.lazy = False

        return thunk

    def grad(self, inputs, output_grads):
        gout, = output_grads
        s = inputs[1]
        # # Multiply the last dimension of the gradient by 2, they represent
        # # both positive and negative frequencies, except the first
        # # and last elements (for even transforms) which are unique.
        # idx = [slice(None)] * (gf.ndim - 2) \
        #     + [slice(1, (s[-1] // 2) + (s[-1] % 2))] + [slice(None)]
        # gf = T.set_subtensor(gf[idx], gf[idx] * 2)
        return [cufft2_op(gout, s), DisconnectedType()()]

    def connection_pattern(self, node):
        # Specificy that shape input parameter has no connection to graph and gradients.
        return [[True], [False]]

cuifft2_op = CuIFFT2Op()


def cufft2(inp, norm=None):
    """
    Performs the 2D fast Fourier transform of a simulated complex-valued input on the GPU.

    The input must be a real-valued float32 variable of dimensions (m, ..., nx, ny, 2).
    It performs 2D FFTs of size (..., nx, ny) on m batches.

    The output is a GpuArray of dimensions (m, ..., nx, ny, 2). 

    Parameters
    ----------
    inp
        Array of real-valued float32 of size (m, ..., nx, ny, 2).
    norm : {None, 'ortho', 'no_norm'}
        Normalization of transform. Following numpy, default *None* normalizes
        only the inverse transform by n, 'ortho' yields the unitary transform
        (:math:`1/\sqrt n` forward and inverse). In addition, 'no_norm' leaves
        the transform unnormalized.

    """
    # For Debugging purpose
    print('... using GPU implementation for fft2')

    s = inp.shape[-3:-1] # get (nx, ny)
    cond_norm = _unitary(norm)
    scaling = 1
    if cond_norm == "ortho":
        scaling = T.sqrt(s.prod().astype('float32'))

    return cufft2_op(inp, s) / scaling


def cuifft2(inp, norm=None):
    """
    Performs the 2D inverse fast Fourier transform of a simulated complex-valued input on the GPU.

    The input must be a real-valued float32 variable of dimensions (m, ..., nx, ny, 2).
    It performs 2D IFFTs of size (..., nx, ny) on m batches.

    The output is a GpuArray of dimensions (m, ..., nx, ny, 2). 

    Parameters
    ----------
    inp
        Array of real-valued float32 of size (m, ..., nx, ny, 2).
    norm : {None, 'ortho', 'no_norm'}
        Normalization of transform. Following numpy, default *None* normalizes
        only the inverse transform by n, 'ortho' yields the unitary transform
        (:math:`1/\sqrt n` forward and inverse). In addition, 'no_norm' leaves
        the transform unnormalized.

    """
    # For Debugging purpose
    print('... using GPU implementation for ifft2')

    s = inp.shape[-3:-1]

    cond_norm = _unitary(norm)
    scaling = 1
    if cond_norm is None:
        scaling = s.prod().astype('float32')
    elif cond_norm == "ortho":
        scaling = T.sqrt(s.prod().astype('float32'))

    return cuifft2_op(inp, s) / scaling


def _unitary(norm):
    if norm not in (None, "ortho", "no_norm"):
        raise ValueError("Invalid value %s for norm, must be None, 'ortho' or "
                         "'no norm'" % norm)
    return norm

if scikits_cuda_available:
    # @op_lifter([theano.tensor.fft.FFTOp])
    # @register_opt2([theano.tensor.fft.FFTOp], 'fast_compile')
    @register_opt('fast_compile')
    @op_lifter([cascadenet.network.theano_extensions.fft2.FFT2Op])
    @register_opt2([cascadenet.network.theano_extensions.fft2.FFT2Op], 'fast_compile')
    def local_gpua_cufft2_op(op, ctx_name, inputs, outputs):
        return cufft2_op

    # @op_lifter([theano.tensor.fft.IFFTOp])
    # @register_opt2([theano.tensor.fft.IFFTOp], 'fast_compile')
    @register_opt('fast_compile')
    @op_lifter([cascadenet.network.theano_extensions.fft2.IFFT2Op])
    @register_opt2([cascadenet.network.theano_extensions.fft2.IFFT2Op], 'fast_compile')
    def local_gpua_cuifft2_op(op, ctx_name, inputs, outputs):
        return cuifft2_op
