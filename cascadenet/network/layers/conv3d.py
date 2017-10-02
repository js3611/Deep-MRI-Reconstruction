import theano
import lasagne
from lasagne.layers import Layer, prelu
from helper import ensure_set_name

if 'cuda' in theano.config.device:
    from lasagne.layers.dnn import Conv3DDNNLayer as Conv3DLayer
    # MaxPool2DDNNLayer as MaxPool2DLayer
else:
    raise ImportError(("\n"
                       "+-----------------------------------------------+\n"
                       "| In order to use Conv3D,                       |\n"
                       "| theano.config.device == cuda is required.     |\n"
                       "+-----------------------------------------------+\n"))


def Conv3D(incoming, num_filters, filter_size=3,
           stride=(1, 1, 1), pad='same', W=lasagne.init.HeNormal(),
           b=lasagne.init.Constant(),
           nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
    """
    Overrides the default parameters for Conv3DLayer
    """
    ensure_set_name('conv3d', kwargs)
    return Conv3DLayer(incoming, num_filters, filter_size, stride, pad,
                       W=W, b=b, nonlinearity=nonlinearity, **kwargs)


class Conv3DPrelu(Layer):
    def __init__(self, incoming, num_filters, filter_size=3, stride=(1, 1, 1),
                 pad='same', W=lasagne.init.HeNormal(), b=None, **kwargs):
        # Enforce name
        ensure_set_name('conv3d_prelu', kwargs)

        super(Conv3DPrelu, self).__init__(incoming, **kwargs)
        self.conv = Conv3D(incoming, num_filters, filter_size, stride,
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


class Conv3DAggr(Layer):
    def __init__(self, incoming, num_channels, filter_size=3,
                 stride=(1, 1, 1),
                 pad='same',
                 W=lasagne.init.HeNormal(),
                 b=None,
                 **kwargs):
        ensure_set_name('conv3d_aggr', kwargs)
        super(Conv3DAggr, self).__init__(incoming, **kwargs)
        self.conv = Conv3D(incoming, num_channels, filter_size, stride,
                           pad=pad, W=W, b=b, nonlinearity=None, **kwargs)

        # copy params
        self.params = self.conv.params.copy()

    def get_output_for(self, input, **kwargs):
        return self.conv.get_output_for(input)

    def get_output_shape_for(self, input_shape):
        return self.conv.get_output_shape_for(input_shape)
