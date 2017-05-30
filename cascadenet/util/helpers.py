import lasagne
import theano
import numpy as np


def complex2real(x):
    '''
    Parameter
    ---------
    x: ndarray
        assumes at least 2d. Last 2D axes are split in terms of real and imag
        2d/3d/4d complex valued tensor (n, nx, ny) or (n, nx, ny, nt)

    Returns
    -------
    y: 4d tensor (n, 2, nx, ny)
    '''
    x_real = np.real(x)
    x_imag = np.imag(x)
    y = np.array([x_real, x_imag]).astype(theano.config.floatX)
    # re-order in convenient order
    if x.ndim >= 3:
        y = y.swapaxes(0, 1)
    return y


def real2complex(x):
    '''
    Converts from array of the form ([n, ]2, nx, ny[, nt]) to ([n, ]nx, ny[, nt])
    '''
    x = np.asarray(x)
    if x.shape[0] == 2 and x.shape[1] != 2:  # Hacky check
        return x[0] + x[1] * 1j
    elif x.shape[1] == 2:
        y = x[:, 0] + x[:, 1] * 1j
        return y
    else:
        raise ValueError('Invalid dimension')


def mask_c2r(m):
    return complex2real(m * (1+1j))


def mask_r2c(m):
    return m[0] if m.ndim == 3 else m[:, 0]


def to_lasagne_format(x, mask=False):
    """
    Assumes data is of shape (n[, nt], nx, ny).
    Reshapes to (n, n_channels, nx, ny[, nt])
    Note: Depth must be the last axis, the dimensions will be reordered
    """
    if x.ndim == 4:  # n 3D inputs. reorder axes
        x = np.transpose(x, (0, 2, 3, 1))

    if mask:  # Hacky solution
        x = x*(1+1j)

    x = complex2real(x)

    return x


def from_lasagne_format(x, mask=False):
    """
    Assumes data is of shape (n, 2, nx, ny[, nt]).
    Reshapes to (n, [nt, ]nx, ny)
    """
    if x.ndim == 5:  # n 3D inputs. reorder axes
        x = np.transpose(x, (0, 1, 4, 2, 3))

    if mask:
        x = mask_r2c(x)
    else:
        x = real2complex(x)

    return x
