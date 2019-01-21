import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np
import cascadenet_pytorch.kspace_pytorch as cl


def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)


def relu():
    return nn.ReLU(inplace=True)


def conv_block(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=None):

    # convolution dimension (2D or 3D)
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    # output dim: If None, it is assumed to be the same as n_ch
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1
    if dilation > 1:
        # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
        # pad = dilation
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv

    def conv_i():
        return conv(nf,   nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    # relu
    nll = relu if nl == 'relu' else lrelu

    layers = [conv_1, nll()]
    for i in range(nd-2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)


class DnCn(nn.Module):
    def __init__(self, n_channels=2, nc=5, nd=5, **kwargs):
        super(DnCn, self).__init__()
        self.nc = nc
        self.nd = nd
        print('Creating D{}C{}'.format(nd, nc))
        conv_blocks = []
        dcs = []

        conv_layer = conv_block

        for i in range(nc):
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            dcs.append(cl.DataConsistencyInKspace(norm='ortho'))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self, x, k, m):
        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            x = self.dcs[i].perform(x, k, m)

        return x


class StochasticDnCn(DnCn):
    def __init__(self, n_channels=2, nc=5, nd=5, p=None, **kwargs):
        super(StochasticDnCn, self).__init__(n_channels, nc, nd, **kwargs)

        self.sample = False
        self.p = p
        if not p:
            self.p = np.linspace(0, 0.5, nc)
        print(self.p)

    def forward(self, x, k, m):
        for i in range(self.nc):

            # stochastically drop connection
            if self.training or self.sample:
                if np.random.random() <= self.p[i]:
                    continue

            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            x = self.dcs[i].perform(x, k, m)

        return x

    def set_sample(self, flag=True):
        self.sample = flag


class DnCn3D(nn.Module):
    def __init__(self, n_channels=2, nc=5, nd=5, **kwargs):
        super(DnCn3D, self).__init__()
        self.nc = nc
        self.nd = nd
        print('Creating D{}C{} (3D)'.format(nd, nc))
        conv_blocks = []
        dcs = []

        conv_layer = conv_block

        for i in range(nc):
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            dcs.append(cl.DataConsistencyInKspace(norm='ortho'))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = nn.ModuleList(dcs)

    def forward(self, x, k, m):
        for i in range(self.nc):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            x = self.dcs[i].perform(x, k, m)

        return x


class DnCn3DDS(nn.Module):
    def __init__(self, n_channels=2, nc=5, nd=5, fr_d=None, clipped=False, mode='pytorch', **kwargs):
        """

        Parameters
        ----------

        fr_d: frame distance for data sharing layer. e.g. [1, 3, 5]

        """
        super(DnCn3DDS, self).__init__()
        self.nc = nc
        self.nd = nd
        self.mode = mode
        print('Creating D{}C{}-DS (3D)'.format(nd, nc))
        if self.mode == 'theano':
            print('Initialised with theano mode (backward-compatibility)')
        conv_blocks = []
        dcs = []
        kavgs = []

        if not fr_d:
            fr_d = list(range(10))
        self.fr_d = fr_d

        conv_layer = conv_block

        # update input-output channels for data sharing
        n_channels = 2 * len(fr_d)
        n_out = 2
        kwargs.update({'n_out': 2})

        for i in range(nc):
            kavgs.append(cl.AveragingInKspace(fr_d, i>0, clipped, norm='ortho'))
            conv_blocks.append(conv_layer(n_channels, nd, **kwargs))
            dcs.append(cl.DataConsistencyInKspace(norm='ortho'))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = nn.ModuleList(dcs)
        self.kavgs = nn.ModuleList(kavgs)

    def forward(self, x, k, m):
        for i in range(self.nc):
            x_ds = self.kavgs[i](x, m)
            if self.mode == 'theano':
                # transpose the layes
                x_ds_tmp = torch.zeros_like(x_ds)
                nneigh = len(self.fr_d)
                for j in range(nneigh):
                    x_ds_tmp[:,2*j] = x_ds[:,j]
                    x_ds_tmp[:,2*j+1] = x_ds[:,j+nneigh]
                x_ds = x_ds_tmp

            x_cnn = self.conv_blocks[i](x_ds)
            x = x + x_cnn
            x = self.dcs[i](x, k, m)

        return x


class DnCn3DShared(nn.Module):
    def __init__(self, n_channels=2, nc=5, nd=5, **kwargs):
        super(DnCn3DShared, self).__init__()
        self.nc = nc
        self.nd = nd
        print('Creating D{}C{}-S (3D)'.format(nd, nc))

        self.conv_block = conv_block(n_channels, nd, **kwargs)
        self.dc = cl.DataConsistencyInKspace(norm='ortho')

    def forward(self, x, k, m):
        for i in range(self.nc):
            x_cnn = self.conv_block(x)
            x = x + x_cnn
            x = self.dc.perform(x, k, m)

        return x


class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations

    Parameters
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    hidden_iteration: hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)

    Returns
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # add iteration hidden connection
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_iteration, hidden):
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        ih_to_ih = self.ih2ih(hidden_iteration)

        hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)

        return hidden


class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer

    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode

    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)

    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(BCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)

    def forward(self, input, input_iteration, test=False):
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        if test:
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            hid_init = Variable(torch.zeros(size_h)).cuda()

        output_f = []
        output_b = []
        # forward
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[i], input_iteration[i], hidden)
            output_f.append(hidden)

        output_f = torch.cat(output_f)

        # backward
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[nt - i - 1], input_iteration[nt - i -1], hidden)

            output_b.append(hidden)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b

        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)

        return output


class CRNN_MRI(nn.Module):
    """
    Model for Dynamic MRI Reconstruction using Convolutional Neural Networks

    Parameters
    -----------------------
    incomings: three 5d tensors, [input_image, kspace_data, mask], each of shape (batch_size, 2, width, height, n_seq)

    Returns
    ------------------------------
    output: 5d tensor, [output_image] with shape (batch_size, 2, width, height, n_seq)
    """
    def __init__(self, n_ch=2, nf=64, ks=3, nc=5, nd=5):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CRNN_MRI, self).__init__()
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks

        self.bcrnn = BCRNNlayer(n_ch, nf, ks)
        self.conv1_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv1_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv2_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv2_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv4_x = nn.Conv2d(nf, n_ch, ks, padding = ks//2)
        self.relu = nn.ReLU(inplace=True)

        dcs = []
        for i in range(nc):
            dcs.append(cl.DataConsistencyInKspace(norm='ortho'))
        self.dcs = dcs

    def forward(self, x, k, m, test=False):
        """
        x   - input in image domain, of shape (n, 2, nx, ny, n_seq)
        k   - initially sampled elements in k-space
        m   - corresponding nonzero location
        test - True: the model is in test mode, False: train mode
        """
        net = {}
        n_batch, n_ch, width, height, n_seq = x.size()
        size_h = [n_seq*n_batch, self.nf, width, height]
        if test:
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            hid_init = Variable(torch.zeros(size_h)).cuda()

        for j in range(self.nd-1):
            net['t0_x%d'%j]=hid_init

        for i in range(1,self.nc+1):

            x = x.permute(4,0,1,2,3)
            x = x.contiguous()
            net['t%d_x0' % (i - 1)] = net['t%d_x0' % (i - 1)].view(n_seq, n_batch,self.nf,width, height)
            net['t%d_x0'%i] = self.bcrnn(x, net['t%d_x0'%(i-1)], test)
            net['t%d_x0'%i] = net['t%d_x0'%i].view(-1,self.nf,width, height)

            net['t%d_x1'%i] = self.conv1_x(net['t%d_x0'%i])
            net['t%d_h1'%i] = self.conv1_h(net['t%d_x1'%(i-1)])
            net['t%d_x1'%i] = self.relu(net['t%d_h1'%i]+net['t%d_x1'%i])

            net['t%d_x2'%i] = self.conv2_x(net['t%d_x1'%i])
            net['t%d_h2'%i] = self.conv2_h(net['t%d_x2'%(i-1)])
            net['t%d_x2'%i] = self.relu(net['t%d_h2'%i]+net['t%d_x2'%i])

            net['t%d_x3'%i] = self.conv3_x(net['t%d_x2'%i])
            net['t%d_h3'%i] = self.conv3_h(net['t%d_x3'%(i-1)])
            net['t%d_x3'%i] = self.relu(net['t%d_h3'%i]+net['t%d_x3'%i])

            net['t%d_x4'%i] = self.conv4_x(net['t%d_x3'%i])

            x = x.view(-1,n_ch,width, height)
            net['t%d_out'%i] = x + net['t%d_x4'%i]

            net['t%d_out'%i] = net['t%d_out'%i].view(-1,n_batch, n_ch, width, height)
            net['t%d_out'%i] = net['t%d_out'%i].permute(1,2,3,4,0)
            net['t%d_out'%i].contiguous()
            net['t%d_out'%i] = self.dcs[i-1].perform(net['t%d_out'%i], k, m)
            x = net['t%d_out'%i]

            # clean up i-1
            if test:
                to_delete = [ key for key in net if ('t%d'%(i-1)) in key ]

                for elt in to_delete:
                    del net[elt]

                torch.cuda.empty_cache()

        return net['t%d_out'%i]


