import theano.tensor as T
from lasagne.layers import InputLayer
from cascadenet.network.theano_extensions.tensor import tensor5


def get_dc_input_layers(shape):
    """
    Creates input layer for the CNN. Works for 2D and 3D input.

    Returns
    -------
    net: Ordered Dictionary
       net config with 3 entries: input, kspace_input, mask.
    """
    
    if len(shape) > 4:
        # 5D
        input_var = tensor5('input_var')
        kspace_input_var = tensor5('kspace_input_var')
        mask_var = tensor5('mask')
    else:
        input_var = T.tensor4('input_var')
        kspace_input_var = T.tensor4('kspace_input_var')
        mask_var = T.tensor4('mask')

    input_layer = InputLayer(shape, input_var=input_var, name='input')
    kspace_input_layer = InputLayer(shape, input_var=kspace_input_var,
                                    name='kspace_input')
    mask_layer = InputLayer(shape, input_var=mask_var, name='mask')
    return input_layer, kspace_input_layer, mask_layer
