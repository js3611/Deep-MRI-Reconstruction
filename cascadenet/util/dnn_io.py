import numpy as np


def complex2real(x, swap=True):
    '''
    Parameter
    ---------
    x: ndarray
        assumes at least 2d. Last 2D axes are split in terms of real and imag
        2d/3d/4d complex valued tensor (n, nx, ny) or (n, nx, ny, nt)

    swap: if swap, then transposes

    Returns
    -------
    y: 4d tensor If swap==True (default), then (n, 2, nx, ny), else (2, n, nx, ny).
    '''
    x_real = np.real(x)
    x_imag = np.imag(x)
    y = np.array([x_real, x_imag]).astype(np.float32)
    # re-order in convenient order
    if x.ndim >= 3 and swap:
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


def r2c(x, axis=1):
    """Convert pseudo-complex data (2 real channels) to complex data

    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    shape = x.shape
    if axis < 0: axis = x.ndim + axis
    ctype = np.complex64 if x.dtype == np.float32 else np.complex128

    if axis < len(shape):
        newshape = tuple([i for i in range(0, axis)]) \
                   + tuple([i for i in range(axis+1, x.ndim)]) + (axis,)

        x = x.transpose(newshape)

    x = np.ascontiguousarray(x).view(dtype=ctype)
    return x.reshape(x.shape[:-1])


def c2r(x, axis=0):
    """Convert complex data to pseudo-complex data (2 real channels)

    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    shape = x.shape
    dtype = np.float32 if x.dtype == np.complex64 else np.float64

    x = np.ascontiguousarray(x).view(dtype=dtype).reshape(shape + (2,))

    n = x.ndim
    if axis < 0: axis = n + axis
    if axis < n:
        newshape = tuple([i for i in range(0, axis)]) + (n-1,) \
                   + tuple([i for i in range(axis, n-1)])
        x = x.transpose(newshape)

    return x



def mask_c2r(m):
    return complex2real(m * (1+1j))


def mask_r2c(m):
    return m[0] if m.ndim == 3 else m[:, 0]


def to_tensor_format(x, mask=False):
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


def from_tensor_format(x, mask=False):
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
