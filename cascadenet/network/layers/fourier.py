import numpy as np
import theano
import theano.tensor as T
from utils.mymath import fourier_matrix, inverse_fourier_matrix
from lasagne.layers import Layer

# Ugly but works for now
try:
    import pygpu
    import pycuda.driver
    import skcuda
    from skcuda import fft
    cufft_available = True
except ImportError:
    cufft_available = False

if theano.config.device == 'cuda' and cufft_available:
    from cascadenet.network.theano_extensions.gpuarray.fft2 import cufft2 as fft2g
    from cascadenet.network.theano_extensions.gpuarray.fft2 import cuifft2 as ifft2g
    use_cuda = True
    print "Using GPU version of fft layers"
else:
    from cascadenet.network.theano_extensions.fft2_lasagne import fft2, ifft2
    use_cuda = False
    print "Using CPU version of fft layers"


from cascadenet.network.theano_extensions.fft_helper import fftshift, ifftshift
from cascadenet.network.theano_extensions.fft import fft, ifft


class FFTLayer(Layer):
    def __init__(self, incoming, data_shape, inv=False, **kwargs):
        '''
        Need to take input shape of the fft, since it needs to
        precompute fft matrix
        '''
        super(FFTLayer, self).__init__(incoming, **kwargs)
        self.data_shape = data_shape
        n, _, nx, ny = data_shape
        # create matrix which performs fft

        if inv:
            fourier_mat = inverse_fourier_matrix(nx, ny)
        else:
            fourier_mat = fourier_matrix(nx, ny)

        self.real_fft = np.real(fourier_mat)
        self.complex_fft = np.imag(fourier_mat)

    def transform(self, input):
        '''
        Perform fourier transform using Fourier matrix

        Parameters
        ------------------------------
        input must be of 4d tensor
        with shape [n, 2, nx, ny] where [nx, ny] == self.data_shape. n means
        number of data. 2 means channels for real and complex part of the input
        (channel 1 == real, channel 2 = complex)
        uses real values to simulate the complex operation

        Returns
        ------------------------------
        tensor of the shape [n, 2, nx, ny] which is equivalent to
        fourier transform

        '''
        in_r = input[0]
        in_c = input[1]
        real_fft = self.real_fft
        complex_fft = self.complex_fft
        out_r = T.dot(real_fft, in_r) - T.dot(complex_fft, in_c)
        out_c = T.dot(complex_fft, in_r) + T.dot(real_fft, in_c)
        return T.stack([out_r, out_c])

    def get_output_for(self, input, **kwargs):
        '''
        Computes FFT. Input layer must have dimension [n, 2, nx, ny]
        '''
        out, updates = theano.scan(self.transform, sequences=input)
        return out


class FFT2CPULayer(Layer):
    def __init__(self, incoming, data_shape, inv=False,
                 norm='ortho', **kwargs):
        '''
        Need to take input shape of the fft,
        since it needs to precompute fft matrix

        if nx != ny, we need to matrices

        '''
        super(FFT2Layer, self).__init__(incoming, **kwargs)
        self.fn = fft2 if not inv else ifft2
        self.norm = norm

    def get_output_for(self, input, **kwargs):
        '''
        Computes 2D FFT. Input layer must have dimension (n, 2, nx, ny[, nt])
        '''
        return self.fn(input, norm=self.norm)


class FFT2GPULayer(Layer):
    def __init__(self, incoming, data_shape, inv=False,
                 norm='ortho', **kwargs):
        '''
        Need to take input shape of the fft,
        since it needs to precompute fft matrix

        if nx != ny, we need to matrices

        '''
        super(FFT2Layer, self).__init__(incoming, **kwargs)
        self.fn = fft2g if not inv else ifft2g
        self.is_3d = len(data_shape) == 5
        self.norm = norm

    def get_output_for(self, input, **kwargs):
        '''
        Computes 2D FFT. Input layer must have dimension (n, 2, nx, ny[, nt])
        '''
        if self.is_3d:
            input_fft = input.dimshuffle((0, 4, 2, 3, 1))
            res = self.fn(input_fft, norm=self.norm)
            return res.dimshuffle((0, 4, 2, 3, 1))
        else:
            input_fft = input.dimshuffle((0, 2, 3, 1))
            res = self.fn(input_fft, norm=self.norm)
            return res.dimshuffle((0, 3, 1, 2))


class FT2Layer(Layer):
    def __init__(self, incoming, data_shape, inv=False, **kwargs):
        '''
        Need to take input shape of the fft,
        since it needs to precompute fft matrix

        if nx != ny, we need to matrices

        '''
        super(FFT2Layer, self).__init__(incoming, **kwargs)
        self.is_3d = len(data_shape) == 5
        self.data_shape = data_shape
        if self.is_3d:
            n, _, nx, ny, nt = data_shape
        else:
            n, _, nx, ny = data_shape
        # create matrix which performs fft

        if inv:
            fourier_mat_x = inverse_fourier_matrix(nx, nx)
            fourier_mat_y = inverse_fourier_matrix(ny, ny) if nx != ny else fourier_mat_x
        else:
            fourier_mat_x = fourier_matrix(nx, nx)
            fourier_mat_y = fourier_matrix(ny, ny) if nx != ny else fourier_mat_x

        self.real_fft_x = np.real(fourier_mat_x).astype(theano.config.floatX)
        self.complex_fft_x = np.imag(fourier_mat_x).astype(theano.config.floatX)
        self.real_fft_y = np.real(fourier_mat_y).astype(theano.config.floatX)
        self.complex_fft_y = np.imag(fourier_mat_y).astype(theano.config.floatX)

    def transform(self, input):
        '''
        Perform fourier transform using Fourier matrix

        Parameters
        ------------------------------
        input must be of 4d tensor
        with shape [n, 2, nx, ny] where [nx, ny] == self.data_shape. n means
        number of data. 2 means channels for real and complex part of the input
        (channel 1 == real, channel 2 = complex)
        uses real values to simulate the complex operation

        Returns
        ------------------------------
        tensor of the shape [n, 2, nx, ny] which is equivalent to fourier
        transform
        '''
        u = input[0]
        v = input[1]
        real_fft_x = self.real_fft_x
        complex_fft_x = self.complex_fft_x
        real_fft_y = self.real_fft_y
        complex_fft_y = self.complex_fft_y

        out_u = T.dot(u, real_fft_y.T) - T.dot(v, complex_fft_y.T)
        out_v = T.dot(u, complex_fft_y.T) + T.dot(v, real_fft_y.T)
        out_u2 = T.dot(real_fft_x, out_u) - T.dot(complex_fft_x, out_v)
        out_v2 = T.dot(complex_fft_x, out_u) + T.dot(real_fft_x, out_v)

        return T.stack([out_u2, out_v2])

    def get_output_for(self, input, **kwargs):
        '''
        Computes 2D FFT. Input layer must have dimension [n, 2, nx, ny]
        '''
        if self.is_3d:

            n, nc, nx, ny, nt = self.data_shape
            lin = T.transpose(input, axes=(0, 4, 1, 2, 3))
            lin = lin.reshape((-1, nc, nx, ny))
            lout, updates = theano.scan(self.transform, sequences=lin)
            lout = lout.reshape((-1, nt, nc, nx, ny))
            out = T.transpose(lout, axes=(0, 2, 3, 4, 1))
            return out

            # def loop_over_n(i, arr):
            #     out, updates = theano.scan(self.transform,
            #                                sequences=arr[:, :, i])[0]
            #     return out

            # nt = self.data_shape[-1]
            # out, updates = theano.scan(loop_over_n,
            #                            non_sequences=input,
            #                            sequences=xrange(nt))
            # return out

        out, updates = theano.scan(self.transform, sequences=input)
        return out


class FFTCLayer(Layer):
    def __init__(self, incoming, data_shape, norm=None,
                 inv=False, **kwargs):
        '''

        Assumes data is in the format of (n, 2, nx, ny[, nt])

        Applies FFTC along the last axis

        '''
        super(FFTCLayer, self).__init__(incoming, **kwargs)
        self.fn = fft if not inv else ifft
        self.is_3d = len(data_shape) == 5
        self.norm = norm
        # if isinstance(axes, int):
        #     axes = (axes,)

        # # Note that because we are simulating complex number, with 2 channels,
        # # we need to be careful when we invoke axes=(-1) and so we need to make
        # # sure we fix all -n to -n-1.
        # axes_list = []
        # for ax in axes:
        #     if ax < 0:
        #         axes_list.append(ax-1)
        #     else:
        #         axes_list.append(ax)
        # axes = tuple(axes_list)
        # print(axes)

        # self.axes = axes

    def get_output_for(self, input, **kwargs):
        '''
        Computes FFTC. Input layer must have dimension
        '''
        if self.is_3d:
            #  Convert to (n, nx, ny[, nt], 2) for fft
            tmp = input.dimshuffle((0, 2, 3, 4, 1))
            tmp_shifted = ifftshift(tmp, axes=(-2,))
            tmp_tfx_shifted = self.fn(tmp_shifted, norm=self.norm)
            tmp_tfx = fftshift(tmp_tfx_shifted, axes=(-2,))
            #  Convert back to (n, 2, nx, ny[, nt])
            return tmp_tfx.dimshuffle((0, 4, 1, 2, 3))

        else:
            # shape: [n, nc, nx, nt]
            tmp = input.dimshuffle((0, 2, 3, 1))
            data_xf = ifftshift(tmp, axes=(-2,))
            data_xt = self.fn(data_xf, norm=self.norm)
            data_xt = fftshift(data_xt, axes=(-2,))
            return data_xt.dimshuffle((0, 3, 1, 2))

if use_cuda:
    FFT2Layer = FFT2GPULayer
else:
    # FFT2Layer = FT2Layer
    FFT2Layer = FFT2CPULayer

# def FT2Layer(incoming, data_shape):
#     net = FFTLayer(incoming, data_shape)
#     net = TransposeLayer(net)
#     net = FFTLayer(net, data_shape)
#     net = TransposeLayer(net)
#     return net


# def IFT2Layer(incoming, data_shape):
#     net = FFTLayer(incoming, data_shape, inv=True)
#     net = TransposeLayer(net)
#     net = FFTLayer(net, data_shape, inv=True)
#     net = TransposeLayer(net)
#     return net


# def fftshift(x):
#     return x


# def ifftshift(x):
#     return x

