import lasagne
import theano
import theano.tensor as T
import numpy as np
from lasagne.layers import MergeLayer, get_output
from cascadenet.network.layers.fourier import FFT2Layer


def roll_and_sum(prior_result, orig):
    res = prior_result + orig
    res = T.roll(res, 1, axis=-1)
    return res


class KspaceFillNeighbourLayer(MergeLayer):
    '''
    k-space fill layer - The input data is assumed to be in k-space grid.

    The input data is assumed to be in k-space grid.
    This layer should be invoked from AverageInKspaceLayer
    '''
    def __init__(self, incomings, frame_dist=range(5), divide_by_n=False,
                 **kwargs):
        super(KspaceFillNeighbourLayer, self).__init__(incomings, **kwargs)
        self.frame_dist = frame_dist

        n_samples = [1 + 2*i for i in self.frame_dist]
        self.n_samples = n_samples
        self.divide_by_n = divide_by_n

    def get_output_for(self, inputs, **kwargs):
        '''

        Parameters
        ------------------------------
        inputs: two 5d tensors, [kspace_data, mask], each of shape (n, 2, nx, ny, nt)

        Returns
        ------------------------------
        output: 5d tensor, missing lines of k-space are filled using neighbouring frames.
        shape becomes (n* (len(frame_dist), 2, nx, ny, nt)
        '''
        x = inputs[0]
        mask = inputs[1]

        result, _ = theano.scan(fn=roll_and_sum,
                                outputs_info=T.zeros_like(x),
                                non_sequences=(x),
                                n_steps=T.constant(np.max(self.n_samples)))

        mask_result, _ = theano.scan(fn=roll_and_sum,
                                     outputs_info=T.zeros_like(x),
                                     non_sequences=(mask),
                                     n_steps=T.constant(np.max(self.n_samples)))

        results = [x]
        for i, t in enumerate(self.n_samples):
            # divide unbiasedly
            if self.divide_by_n:
                c = float(t)
            else:
                c = 1.0

            acc = result[t-1]
            mask_acc = mask_result[t-1]
            # when rolling back, need extra 1 because roll_and_sum rolls after adding a val.
            avg = T.roll(acc / T.maximum(c, mask_acc),
                         -self.frame_dist[i]-1,
                         axis=-1)
            res = avg * (1-mask) + x * mask
            results.append(res)

        return T.concatenate(results, axis=1)  # concatenate along channels

    def get_output_shape_for(self, input_shapes, **kwargs):
        n, nc, nx, ny, nt = input_shapes[0]
        nc_new = (len(self.frame_dist) + 1) * nc
        return (n, nc_new, nx, ny, nt)


class KspaceFillNeighbourLayer_Clipped(MergeLayer):
    '''
    k-space fill layer with clipping at the edge.

    The input data is assumed to be in k-space grid.
    This layer should be invoked from AverageInKspaceLayer
    '''

    def __init__(self, incomings, nt, frame_dist=range(5), divide_by_n=False,
                 **kwargs):
        super(KspaceFillNeighbourLayer_Clipped, self).__init__(incomings, **kwargs)
        self.frame_dist = frame_dist

        n_samples = [1 + 2*i for i in self.frame_dist]
        self.n_samples = n_samples
        self.divide_by_n = divide_by_n
        self.nt = nt

    def get_output_for(self, inputs, **kwargs):
        '''

        Parameters
        ------------------------------
        inputs: two 5d tensors, [kspace_data, mask], each of shape (n, 2, nx, ny, nt)

        Returns
        ------------------------------
        output: 5d tensor, missing lines of k-space are filled using neighbouring frames.
        shape becomes (n* (len(frame_dist), 2, nx, ny, nt)
        '''
        x = inputs[0]
        mask = inputs[1]

        results = [x]
        for i, t in enumerate(self.n_samples):
            dist = t/2
            # divide unbiasedly
            if self.divide_by_n:
                c = float(t)
            else:
                c = 1.0

            def fn(i, input):
                s = slice(T.maximum(0, i-dist), T.minimum(self.nt, i+dist+1))
                return input[..., s].sum(axis=-1)

            result, _ = theano.scan(fn,
                                    non_sequences=(x),
                                    sequences=np.arange(self.nt))

            mask_result, _ = theano.scan(fn,
                                         non_sequences=(mask),
                                         sequences=np.arange(self.nt))

            acc = T.transpose(result, axes=(1, 2, 3, 4, 0))
            mask_acc = T.transpose(mask_result, axes=(1, 2, 3, 4, 0))

            # when rolling back, need extra 1 because roll_and_sum rolls after adding a val.
            avg = acc / T.maximum(c, mask_acc)

            # concatenate results in avg
            res = avg * (1-mask) + x * mask
            results.append(res)

        return T.concatenate(results, axis=1)  # concatenate along channels

    def get_output_shape_for(self, input_shapes, **kwargs):
        n, nc, nx, ny, nt = input_shapes[0]
        nc_new = (len(self.frame_dist) + 1) * nc
        return (n, nc_new, nx, ny, nt)


class AverageInKspaceLayer(MergeLayer):
    '''
    Average-in-k-space layer

    First transforms the representation in Fourier domain,
    then performs averaging along temporal axis, then transforms back to image
    domain. Works only for 5D tensor (see parameter descriptions).


    Parameters
    -----------------------------
    incomings: two 5d tensors, [kspace_data, mask], each of shape (n, 2, nx, ny, nt)

    data_shape: shape of the incoming tensors: (n, 2, nx, ny, nt) (This is for convenience)

    frame_dist: a list of distances of neighbours to sample for each averaging channel
        if frame_dist=[1], samples from [-1, 1] for each temporal frames
        if frame_dist=[3, 5], samples from [-3,-2,...,0,1,...,3] for one,
                                           [-5,-4,...,0,1,...,5] for the second one

    divide_by_n: bool - Decides how averaging will be done.
        True => divide by number of neighbours (=#2*frame_dist+1)
        False => divide by number of nonzero contributions

    clipped: bool - By default the layer assumes periodic boundary condition along temporal axis.
        True => Averaging will be clipped at the boundary, no circular references.
        False => Averages with circular referencing (i.e. at t=0, gets contribution from t=nt-1, so on).

    Returns
    ------------------------------
    output: 5d tensor, missing lines of k-space are filled using neighbouring frames.
            shape becomes (n* (len(frame_dist)), 2, nx, ny, nt)
    '''

    def __init__(self, incomings, data_shape, frame_dist=[1, 3, 5],
                 divide_by_n=False, clipped=False, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'kspace_averaging_layer'

        super(AverageInKspaceLayer, self).__init__(incomings, **kwargs)
        data, mask = incomings
        n, nc, nx, ny, nt = data_shape
        nc_new = (len(frame_dist)+1)*2
        self.data = data
        self.mask = mask
        self.frame_dist = frame_dist
        self.divide_by_n = divide_by_n
        self.dft2 = FFT2Layer(data, data_shape, name='kavg_dft2')
        if clipped:
            self.kavg = KspaceFillNeighbourLayer_Clipped([self.dft2, mask],
                                                         nt,
                                                         frame_dist,
                                                         divide_by_n,
                                                         name='kavg_avg')
        else:
            self.kavg = KspaceFillNeighbourLayer([self.dft2, mask],
                                                 frame_dist, divide_by_n,
                                                 name='kavg_avg')
        self.kavg_tmp = lasagne.layers.reshape(self.kavg, (-1, 2, nx, ny, nt))
        self.idft2 = FFT2Layer(self.kavg_tmp, data_shape, inv=True, name='kavg_idft2')
        self.out = lasagne.layers.reshape(self.idft2, (-1, nc_new, nx, ny, nt))

    def get_output_for(self, inputs, **kwargs):
        x = inputs[0]
        mask = inputs[1]
        res = get_output(self.out,
                         {self.data: x,
                          self.mask: mask})
        return res

    def get_output_shape_for(self, input_shapes, **kwargs):
        return self.kavg.get_output_shape_for(input_shapes)
