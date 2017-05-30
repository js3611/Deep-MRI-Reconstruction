import numpy as np
import theano.tensor as T
import lasagne
from lasagne.layers import Layer


class TransposeLayer(Layer):
    def get_output_for(self, input, **kwargs):
        transposed = T.transpose(input, axes=(0, 1, 3, 2))
        return transposed


class SubpixelLayer(Layer):
    def __init__(self, incoming, r, c, **kwargs):
        super(SubpixelLayer, self).__init__(incoming, **kwargs)
        self.r = r
        self.c = c

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.c, self.r*input_shape[2],
                self.r*input_shape[3])

    def get_output_for(self, input, deterministic=False, **kwargs):
        out = T.zeros((input.shape[0], self.output_shape[1],
                       self.output_shape[2], self.output_shape[3]))

        for x in xrange(self.r):
            # loop across all feature maps belonging to this channel
            for y in xrange(self.r):
                out = T.inc_subtensor(out[:, :, x::self.r, y::self.r],
                                      input[:, self.r*x+y::self.r*self.r, :, :])
        return out


class Subpixel3DLayer(Layer):
    """
    r: upscale factor
    c: number of channels left after upscaling (which usually should be c_in / (r**3))
    """
    def __init__(self, incoming, r, c, **kwargs):
        super(SubpixelLayer, self).__init__(incoming, **kwargs)
        self.r = r
        self.c = c

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.c, self.r*input_shape[2],
                self.r*input_shape[3], self.r*input_shape[4])

    def get_output_for(self, input, deterministic=False, **kwargs):
        r = self.r
        out = T.zeros((input.shape[0], self.output_shape[1],
                       self.output_shape[2], self.output_shape[3], self.output_shape[4]))

        for x in xrange(r):
            # loop across all feature maps belonging to this channel
            for y in xrange(r):
                for z in xrange(r):
                    out = T.inc_subtensor(out[..., x::r, y::r, z::r],
                                          input[:, (r**2)*x+(r*y)+z::r*r*r, ...])
        return out


class ShuffleLayer(Layer):
    """
    Slices the input at a specific axis and at specific indices.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    indices : int or slice instance
        If an ``int``, selects a single element from the given axis, dropping
        the axis. If a slice, selects all elements in the given range, keeping
        the axis.

    axis : int
        Specifies the axis from which the indices are selected.

    Examples
    --------
    >>> from lasagne.layers import SliceLayer, InputLayer
    >>> l_in = InputLayer((2, 3, 4))
    >>> SliceLayer(l_in, indices=0, axis=1).output_shape
    ... # equals input[:, 0]
    (2, 4)
    >>> SliceLayer(l_in, indices=slice(0, 1), axis=1).output_shape
    ... # equals input[:, 0:1]
    (2, 1, 4)
    >>> SliceLayer(l_in, indices=slice(-2, None), axis=-1).output_shape
    ... # equals input[..., -2:]
    (2, 3, 2)
    """
    def __init__(self, incoming, order=None, axis=1, **kwargs):
        super(ShuffleLayer, self).__init__(incoming, **kwargs)
        self.axis = axis
        self.order = order
        if not order:
            n = lasagne.layers.get_output_shape(incoming)[axis]
            self.order = np.random.permutation(np.arange(n))

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        axis = self.axis
        if axis < 0:
            axis += input.ndim
        return input[(slice(None),) * axis + (self.order,)]
