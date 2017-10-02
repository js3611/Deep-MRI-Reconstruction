import numpy as np
from . import mymath
from numpy.lib.stride_tricks import as_strided


def soft_thresh(u, lmda):
    """Soft-threshing operator for complex valued input"""
    Su = (abs(u) - lmda) / abs(u) * u
    Su[abs(u) < lmda] = 0
    return Su


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)


def var_dens_mask(shape, ivar, sample_high_freq=True):
    """Variable Density Mask (2D undersampling)"""
    if len(shape) == 3:
        Nt, Nx, Ny = shape
    else:
        Nx, Ny = shape
        Nt = 1

    pdf_x = normal_pdf(Nx, ivar)
    pdf_y = normal_pdf(Ny, ivar)
    pdf = np.outer(pdf_x, pdf_y)

    size = pdf.itemsize
    strided_pdf = as_strided(pdf, (Nt, Nx, Ny), (0, Ny * size, size))
    # this must be false if undersampling rate is very low (around 90%~ish)
    if sample_high_freq:
        strided_pdf = strided_pdf / 1.25 + 0.02
    mask = np.random.binomial(1, strided_pdf)

    xc = Nx / 2
    yc = Ny / 2
    mask[:, xc - 4:xc + 5, yc - 4:yc + 5] = True

    if Nt == 1:
        return mask.reshape((Nx, Ny))

    return mask


def cartesian_mask(shape, ivar, centred=False,
                   sample_high_freq=True, sample_centre=True, sample_n=10):
    """Undersamples along Nx

    Parameters
    ----------

    shape: tuple - [nt, nx, ny]

    ivar: sensitivity parameter for Gaussian distribution

    """
    if len(shape) == 3:
        Nt, Nx, Ny = shape
    else:
        Nx, Ny = shape
        Nt = 1

    pdf_x = normal_pdf(Nx, ivar)

    # this must be false if undersampling rate is very low (around 90%~ish)
    if sample_high_freq:
        pdf_x = pdf_x / 1.25 + 0.02

    size = pdf_x.itemsize
    strided_pdf = as_strided(pdf_x, (Nt, Nx, 1), (0, size, 0))
    mask = np.random.binomial(1, strided_pdf)
    size = mask.itemsize
    mask = as_strided(mask, (Nt, Nx, Ny), (size * Nx, size, 0))

    if sample_centre:
        s = sample_n / 2
        xc = Nx / 2
        yc = Ny / 2
        mask[:, xc - s:xc + s, :] = True

    if not centred:
        mask = mymath.ifftshift(mask, axes=(-1, -2))

    return mask


def shear_grid_mask(shape, acceleration_rate, sample_low_freq=True,
                    centred=False, sample_n=10):
    '''
    Creates undersampling mask which samples in sheer grid

    Parameters
    ----------

    shape: (nt, nx, ny)

    acceleration_rate: int

    Returns
    -------

    array

    '''
    Nt, Nx, Ny = shape
    start = np.random.randint(0, acceleration_rate)
    mask = np.zeros((Nt, Nx))
    for t in xrange(Nt):
        mask[t, (start+t)%acceleration_rate::acceleration_rate] = 1

    xc = Nx / 2
    xl = sample_n / 2
    if sample_low_freq and centred:
        xh = xl
        if sample_n % 2 == 0:
            xh += 1
        mask[:, xc - xl:xc + xh+1] = 1

    elif sample_low_freq:
        xh = xl
        if sample_n % 2 == 1:
            xh -= 1

        if xl > 0:
            mask[:, :xl] = 1
        if xh > 0:
            mask[:, -xh:] = 1

    mask_rep = np.repeat(mask[..., np.newaxis], Ny, axis=-1)
    return mask_rep


def perturbed_shear_grid_mask(shape, acceleration_rate, sample_low_freq=True,
                              centred=False,
                              sample_n=10):
    Nt, Nx, Ny = shape
    start = np.random.randint(0, acceleration_rate)
    mask = np.zeros((Nt, Nx))
    for t in xrange(Nt):
        mask[t, (start+t)%acceleration_rate::acceleration_rate] = 1

    # brute force
    rand_code = np.random.randint(0, 3, size=Nt*Nx)
    shift = np.array([-1, 0, 1])[rand_code]
    new_mask = np.zeros_like(mask)
    for t in xrange(Nt):
        for x in xrange(Nx):
            if mask[t, x]:
                new_mask[t, (x + shift[t*x])%Nx] = 1

    xc = Nx / 2
    xl = sample_n / 2
    if sample_low_freq and centred:
        xh = xl
        if sample_n % 2 == 0:
            xh += 1
        new_mask[:, xc - xl:xc + xh+1] = 1

    elif sample_low_freq:
        xh = xl
        if sample_n % 2 == 1:
            xh -= 1

        new_mask[:, :xl] = 1
        new_mask[:, -xh:] = 1
    mask_rep = np.repeat(new_mask[..., np.newaxis], Ny, axis=-1)

    return mask_rep


def undersample(x, mask, centred=False, norm='ortho'):
    '''
    Undersample x. FFT2 will be applied to the last 2 axis
    Parameters
    ----------
    x: array_like
        data
    mask: array_like
        undersampling mask in fourier domain
    Returns
    -------
    xu: array_like
        undersampled image in image domain. Note that it is complex valued

    x_fu: array_like
        undersampled data in kspace

    '''
    assert x.shape == mask.shape
    if centred:
        x_f = mymath.fft2c(x, norm=norm)
        x_fu = x_f * mask
        x_u = mymath.ifft2c(x_fu, norm=norm)
        return x_u, x_fu
    else:
        x_f = mymath.fft2(x, norm=norm)
        x_fu = x_f * mask
        x_u = mymath.ifft2(x_fu, norm=norm)
        return x_u, x_fu


def data_consistency(x, y, mask, centered=False, norm='ortho'):
    '''
    x is in image space,
    y is in k-space
    '''
    if centered:
        xf = mymath.fft2c(x, norm=norm)
        xm = (1 - mask) * xf + y
        xd = mymath.ifft2c(xm, norm=norm)
    else:
        xf = mymath.fft2(x, norm=norm)
        xm = (1 - mask) * xf + y
        xd = mymath.ifft2(xm, norm=norm)

    return xd


def get_phase(x):
    xr = np.real(x)
    xi = np.imag(x)
    phase = np.arctan(xi / (xr + 1e-12))
    return phase


def undersampling_rate(mask):
    return float(mask.sum()) / mask.size
