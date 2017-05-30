import theano.tensor as T
import lasagne
from lasagne.layers import Layer
from helper import ensure_set_name


class IdLayer(Layer):
    def get_output_for(self, input, **kwargs):
        return input


class SumLayer(Layer):
    def get_output_for(self, input, **kwargs):
        return input.sum(axis=-1)

    def get_output_shape_for(self, input_shape):
        return input_shape[:-1]


class SHLULayer(Layer):
    def get_output_for(self, input, **kwargs):
        return T.sgn(input) * T.maximum(input - 1, 0)


class ResidualLayer(lasagne.layers.ElemwiseSumLayer):
    '''
    Residual Layer, which just wraps around ElemwiseSumLayer
    '''

    def __init__(self, incomings, **kwargs):
        ensure_set_name('res', kwargs)
        super(ResidualLayer, self).__init__(incomings, **kwargs)
        # store names
        input_names = []
        for l in incomings:
            if isinstance(l, lasagne.layers.InputLayer):
                input_names.append(l.name if l.name else l.input_var.name)
            elif l.name:
                input_names.append(l.name)
            else:
                input_names.append(str(l))

        self.input_names = input_names

    def get_output_for(self, inputs, **kwargs):
        return super(lasagne.layers.ElemwiseSumLayer,
                     self).get_output_for(inputs, **kwargs)
