import theano
import lasagne
from lasagne.layers import Layer, prelu
from helper import ensure_set_name

if theano.config.device == 'cuda':
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    # MaxPool2DDNNLayer as MaxPool2DLayer
else:
    from lasagne.layers import Conv2DLayer as ConvLayer


def Conv(incoming, num_filters, filter_size=3,
         stride=(1, 1), pad='same', W=lasagne.init.HeNormal(),
         b=None, nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
    """
    Overrides the default parameters for ConvLayer
    """
    ensure_set_name('conv', kwargs)

    return ConvLayer(incoming, num_filters, filter_size, stride, pad, W=W, b=b,
                     nonlinearity=nonlinearity, **kwargs)


class ConvPrelu(Layer):
    def __init__(self, incoming, num_filters, filter_size=3, stride=(1, 1),
                 pad='same', W=lasagne.init.HeNormal(), b=None, **kwargs):
        # Enforce name
        ensure_set_name('conv_prelu', kwargs)

        super(ConvPrelu, self).__init__(incoming, **kwargs)
        self.conv = Conv(incoming, num_filters, filter_size, stride,
                         pad=pad, W=W, b=b, nonlinearity=None, **kwargs)
        self.prelu = prelu(self.conv, **kwargs)

        self.params = self.conv.params.copy()
        self.params.update(self.prelu.params)

    def get_output_for(self, input, **kwargs):
        out_conv = self.conv.get_output_for(input)
        out_prelu = self.prelu.get_output_for(out_conv)
        # return get_output(self.prelu, {self.conv: input})
        return out_prelu

    def get_output_shape_for(self, input, **kwargs):
        return self.conv.get_output_shape_for(input)


class ConvAggr(Layer):
    def __init__(self, incoming, num_channels, filter_size=3, stride=(1, 1),
                 pad='same', W=lasagne.init.HeNormal(), b=None, **kwargs):
        ensure_set_name('conv_aggr', kwargs)
        super(ConvAggr, self).__init__(incoming, **kwargs)
        self.conv = Conv(incoming, num_channels, filter_size, stride, pad=pad,
                         W=W, b=b, nonlinearity=None, **kwargs)

        # copy params
        self.params = self.conv.params.copy()

    def get_output_for(self, input, **kwargs):
        return self.conv.get_output_for(input)

    def get_output_shape_for(self, input_shape):
        return self.conv.get_output_shape_for(input_shape)
