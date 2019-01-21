import numpy as np
import torch
import torch.nn as nn


def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out


class DataConsistencyInKspace(nn.Module):
    """ Create data consistency operator

    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.

    """

    def __init__(self, noise_lvl=None, norm='ortho'):
        super(DataConsistencyInKspace, self).__init__()
        self.normalized = norm == 'ortho'
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        if x.dim() == 4: # input is 2D
            x    = x.permute(0, 2, 3, 1)
            k0   = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
        elif x.dim() == 5: # input is 3D
            x    = x.permute(0, 4, 2, 3, 1)
            k0   = k0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)

        k = torch.fft(x, 2, normalized=self.normalized)
        out = data_consistency(k, k0, mask, self.noise_lvl)
        x_res = torch.ifft(out, 2, normalized=self.normalized)

        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 4, 2, 3, 1)

        return x_res


def get_add_neighbour_op(nc, frame_dist, divide_by_n, clipped):
    max_sample = max(frame_dist) *2 + 1

    # for non-clipping, increase the input circularly
    if clipped:
        padding = (max_sample//2, 0, 0)
    else:
        padding = 0

    # expect data to be in this format: (n, nc, nt, nx, ny) (due to FFT)
    conv = nn.Conv3d(in_channels=nc, out_channels=nc*len(frame_dist),
                     kernel_size=(max_sample, 1, 1),
                     stride=1, padding=padding, bias=False)

    # Although there is only 1 parameter, need to iterate as parameters return generator
    conv.weight.requires_grad = False

    # kernel has size nc=2, nc'=8, kt, kx, ky
    for i, n in enumerate(frame_dist):
        m = max_sample // 2
        #c = 1 / (n * 2 + 1) if divide_by_n else 1
        c = 1
        wt = np.zeros((2, max_sample, 1, 1), dtype=np.float32)
        wt[0, m-n:m+n+1] = c
        wt2 = np.zeros((2, max_sample, 1, 1), dtype=np.float32)
        wt2[1, m-n:m+n+1] = c

        conv.weight.data[2*i] = torch.from_numpy(wt)
        conv.weight.data[2*i+1] = torch.from_numpy(wt2)

    conv.cuda()
    return conv


class KspaceFillNeighbourLayer(nn.Module):
    '''
    k-space fill layer - The input data is assumed to be in k-space grid.

    The input data is assumed to be in k-space grid.
    This layer should be invoked from AverageInKspaceLayer
    '''
    def __init__(self, frame_dist, divide_by_n=False, clipped=True, **kwargs):
        # frame_dist is the extent that data sharing goes.
        # e.g. current frame is 3, frame_dist = 2, then 1,2, and 4,5 are added for reconstructing 3
        super(KspaceFillNeighbourLayer, self).__init__()
        print("fr_d={}, divide_by_n={}, clippd={}".format(frame_dist, divide_by_n, clipped))
        if 0 not in frame_dist:
            raise ValueError("There suppose to be a 0 in fr_d in config file!")
            frame_dist = [0] + frame_dist # include ID

        self.frame_dist  = frame_dist
        self.n_samples   = [1 + 2*i for i in self.frame_dist]
        self.divide_by_n = divide_by_n
        self.clipped     = clipped
        self.op = get_add_neighbour_op(2, frame_dist, divide_by_n, clipped)

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, k, mask):
        '''

        Parameters
        ------------------------------
        inputs: two 5d tensors, [kspace_data, mask], each of shape (n, 2, NT, nx, ny)

        Returns
        ------------------------------
        output: 5d tensor, missing lines of k-space are filled using neighbouring frames.
        shape becomes (n* (len(frame_dist), 2, nt, nx, ny)
        '''
        max_d = max(self.frame_dist)
        k_orig = k
        mask_orig = mask
        if not self.clipped:
            # pad input along nt direction, which is circular boundary condition. Otherwise, just pad outside
            # places with 0 (zero-boundary condition)
            k = torch.cat([k[:,:,-max_d:], k, k[:,:,:max_d]], 2)
            mask = torch.cat([mask[:,:,-max_d:], mask, mask[:,:,:max_d]], 2)

        # start with x, then copy over accumulatedly...
        res = self.op(k)
        if not self.divide_by_n:
            # divide by n basically means for each kspace location, if n non-zero values from neighboring
            # time frames contributes to it, then divide this entry by n (like a normalization)
            res_mask = self.op(mask)
            res = res / res_mask.clamp(min=1)
        else:
            res_mask = self.op(torch.ones_like(mask))
            res = res / res_mask.clamp(min=1)

        res = data_consistency(res,
                               k_orig.repeat(1,len(self.frame_dist),1,1,1),
                               mask_orig.repeat(1,len(self.frame_dist),1,1,1))

        nb, nc_ri, nt, nx, ny = res.shape # here ri_nc is complicated with data sharing replica and real-img dimension
        res = res.reshape(nb, nc_ri//2, 2, nt, nx, ny)
        return res


class AveragingInKspace(nn.Module):
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
            shape becomes (n, (len(frame_dist))* 2, nx, ny, nt)
    '''

    def __init__(self, frame_dist, divide_by_n=False, clipped=True, norm='ortho'):
        super(AveragingInKspace, self).__init__()
        self.normalized = norm == 'ortho'
        self.frame_dist = frame_dist
        self.divide_by_n = divide_by_n
        self.kavg = KspaceFillNeighbourLayer(frame_dist, divide_by_n, clipped)

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, mask):
        """
        x    - input in image space, shape (n, 2, nx, ny, nt)
        mask - corresponding nonzero location
        """
        mask = mask.permute(0, 1, 4, 2, 3)

        x = x.permute(0, 4, 2, 3, 1) # put t to front, in convenience for fft
        k = torch.fft(x, 2, normalized=self.normalized)
        k = k.permute(0, 4, 1, 2, 3) # then put ri to the front, then t

        # data sharing
        # nc is the numpy of copies of kspace, specified by frame_dist
        out = self.kavg.perform(k, mask)
        # after datasharing, it is nb, nc, 2, nt, nx, ny

        nb, nc, _, nt, nx, ny = out.shape # , jo's version

        # out.shape: [nb, 2*len(frame_dist), nt, nx, ny]
        # we then detatch confused real/img channel and replica kspace channel due to datasharing (nc)
        out = out.permute(0,1,3,4,5,2) # jo version, split ri and nc, put ri to the back for ifft
        x_res = torch.ifft(out, 2, normalized=self.normalized)


        # now nb, nc, nt, nx, ny, ri, put ri to channel position, and after nc (i.e. within each nc)
        x_res = x_res.permute(0,1,5,3,4,2).reshape(nb, nc*2, nx,ny, nt)# jo version

        return x_res