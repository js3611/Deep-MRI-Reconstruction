"""DEPRECATED -- tensor5 is already introduced in the new version of Theano"""
import theano
import theano.tensor as T


def tensor5(name=None, dtype=None):
    if dtype is None:
        dtype = theano.config.floatX
    type = T.TensorType(dtype, (False, )*5)
    return type(name)
