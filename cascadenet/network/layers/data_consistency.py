import theano.tensor as T
from lasagne.layers import MergeLayer, get_output
from fourier import FFT2Layer, FFTCLayer
from cascadenet.network.theano_extensions.fft_helper import fftshift, ifftshift
from cascadenet.network.theano_extensions.fft import fft, ifft 


class DataConsistencyLayer(MergeLayer):
    '''
    Data consistency layer
    '''

    def __init__(self, incomings, inv_noise_level=None, **kwargs):
        super(DataConsistencyLayer, self).__init__(incomings, **kwargs)
        self.inv_noise_level = inv_noise_level

    def get_output_for(self, inputs, **kwargs):
        '''

        Parameters
        ------------------------------
        inputs: 2 4d tensors, first is data, second is the k-space samples

        Returns
        ------------------------------
        output: 4d tensor, data input with entries replaced with sampled vals
        '''
        x = inputs[0]
        x_sampled = inputs[1]
        v = self.inv_noise_level
        if v:  # noisy case
            out = (x + v * x_sampled) / (1 + v)
        else:  # noiseless case
            mask = T.set_subtensor(x_sampled[T.neq(x_sampled, 0).nonzero()], 1)
            out = (1 - mask) * x + x_sampled
        return out

    def get_output_shape_for(self, input_shapes, **kwargs):
        return input_shapes[0]


class DataConsistencyWithMaskLayer(MergeLayer):
    '''
    Data consistency layer
    '''

    def __init__(self, incomings, inv_noise_level=None, **kwargs):
        super(DataConsistencyWithMaskLayer, self).__init__(incomings, **kwargs)
        self.inv_noise_level = inv_noise_level

    def get_output_for(self, inputs, **kwargs):
        '''

        Parameters
        ------------------------------
        inputs: 3 4d tensors
            First is data, second is the mask, third is the k-space samples

        Returns
        ------------------------------
        output: 4d tensor, data input with entries replaced with the sampled
        values
        '''
        x = inputs[0]
        mask = inputs[1]
        x_sampled = inputs[2]
        v = self.inv_noise_level
        if v:  # noisy case
            out = (x + v * x_sampled) / (1 + v)
        else:  # noiseless case
            out = (1 - mask) * x + x_sampled
        return out

    def get_output_shape_for(self, input_shapes, **kwargs):
        return input_shapes[0]


class DCLayer(MergeLayer):
    '''
    Data consistency layer
    '''
    def __init__(self, incomings, data_shape, inv_noise_level=None, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'dc'

        super(DCLayer, self).__init__(incomings, **kwargs)
        self.inv_noise_level = inv_noise_level
        data, mask, sampled = incomings
        self.data = data
        self.mask = mask
        self.sampled = sampled
        self.dft2 = FFT2Layer(data, data_shape, name='dc_dft2')
        self.dc = DataConsistencyWithMaskLayer([self.dft2, mask, sampled],
                                               name='dc_consistency')
        self.idft2 = FFT2Layer(self.dc, data_shape, inv=True, name='dc_idft2')

    def get_output_for(self, inputs, **kwargs):
        x = inputs[0]
        mask = inputs[1]
        x_sampled = inputs[2]
        return get_output(self.idft2,
                          {self.data: x,
                           self.mask: mask,
                           self.sampled: x_sampled})

    def get_output_shape_for(self, input_shapes, **kwargs):
        return input_shapes[0]
