import numpy as np
import theano.tensor as T
import lasagne
from lasagne.layers import Layer, pool


class PoolNDLayer(Layer):
    """
    ND pooling layer

    Performs ND mean or max-pooling over the trailing axes
    of a ND input tensor.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.

    pool_size : integer or iterable
        The length of the pooling region in each dimension.  If an integer, it
        is promoted to a square pooling region. If an iterable, it should have
        n elements.

    stride : integer, iterable or ``None``
        The strides between sucessive pooling regions in each dimension.
        If ``None`` then ``stride = pool_size``.

    pad : integer or iterable
        Number of elements to be added on each side of the input
        in each dimension. Each value must be less than
        the corresponding stride.

    ignore_border : bool
        If ``True``, partial pooling regions will be ignored.
        Must be ``True`` if ``pad != (0, 0)``.

    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        Pooling mode: max-pooling or mean-pooling including/excluding zeros
        from partially padded pooling regions. Default is 'max'.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    See Also
    --------
    MaxPool2DLayer : Shortcut for max pooling layer.

    Notes
    -----
    The value used to pad the input is chosen to be less than
    the minimum of the input, so that the output of each pooling region
    always corresponds to some element in the unpadded input region.

    Using ``ignore_border=False`` prevents Theano from using cuDNN for the
    operation, so it will fall back to a slower implementation.
    """

    def __init__(self, incoming, n, pool_size, stride=None, pad=0,
                 ignore_border=True, mode='max', **kwargs):
        super(PoolNDLayer, self).__init__(incoming, **kwargs)

        self.n = n
        self.pool_size = lasagne.utils.as_tuple(pool_size, n)

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = lasagne.utils.as_tuple(stride, n)

        self.pad = lasagne.utils.as_tuple(pad, n)

        self.ignore_border = ignore_border
        self.mode = mode

        # since it uses pool_2d, if n is odd, we append dummy dimension
        if n % 2 == 1:
            self.pool_size += (1, )
            self.pad += (0, )
            self.stride += (1, )

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        tr = len(output_shape) - self.n

        for i in xrange(self.n):
            output_shape[tr+i] = pool.pool_output_length(input_shape[tr+i],
                                                         pool_size=self.pool_size[i],
                                                         stride=self.stride[i],
                                                         pad=self.pad[i],
                                                         ignore_border=self.ignore_border,)

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        """ Uses pool_2d to pool each dimension."""
        input_shape = input.shape
        n = self.n
        if n % 2 == 1:
            n += 1
            input = T.shape_padright(input, 1)

        # aggregate axis for clearer code?
        n_axis = input.ndim - n
        # map_shape = input.shape[axis:]
        # new_shape = T.cast(T.join(0, T.as_tensor([-1]), map_shape), 'int64')
        # input = T.reshape(input, new_shape, n+1)

        # loop  reshape -> pool for n trailing axis
        for i in np.arange(0, n, 2):

            # extract parameters for the corresponding axes
            i1 = (n-2 + i) % n
            i2 = (n-1 + i) % n

            # pool last 2 axis
            input = pool.pool_2d(input,
                                 ds=(self.pool_size[i1], self.pool_size[i2]),
                                 st=(self.stride[i1], self.stride[i2]),
                                 ignore_border=self.ignore_border,
                                 padding=(self.pad[i1], self.pad[i2]),
                                 mode=self.mode, )

            # Get next permutation, which shifts by 2 (+1 is for first axis)
            fixed = tuple(np.arange(n_axis))
            perm = tuple((np.arange(2, n+2) % n) + n_axis)

            # include the first axis from input
            shuffle = fixed + perm

            # shuffle
            input = input.dimshuffle(shuffle)

        # restore original shape
        input = input.reshape(self.get_output_shape_for(input_shape))

        return input


class Upscale3DLayer(Layer):
    """
    3D upscaling layer
    Performs 3D upscaling over the two trailing axes of a 4D input tensor.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.
    scale_factor : integer or iterable
        The scale factor in each dimension. If an integer, it is promoted to
        a square scale factor region. If an iterable, it should have two
        elements.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, scale_factor, **kwargs):
        super(Upscale3DLayer, self).__init__(incoming, **kwargs)

        self.scale_factor = lasagne.utils.as_tuple(scale_factor, 3)

        if self.scale_factor[0] < 1 or self.scale_factor[1] < 1 or self.scale_factor[2] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list
        if output_shape[2] is not None:
            output_shape[2] *= self.scale_factor[0]
        if output_shape[3] is not None:
            output_shape[3] *= self.scale_factor[1]
        if output_shape[4] is not None:
            output_shape[4] *= self.scale_factor[2]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        a, b, c = self.scale_factor
        upscaled = input
        if c > 1:
            upscaled = T.extra_ops.repeat(upscaled, b, 4)
        if b > 1:
            upscaled = T.extra_ops.repeat(upscaled, b, 3)
        if a > 1:
            upscaled = T.extra_ops.repeat(upscaled, a, 2)
        return upscaled
