__author__ = 'Jo Schlemper'

import numpy as np
sqrt = np.sqrt
from numpy.fft import fft, fft2, ifft2, ifft, ifftshift, fftshift


def fftc(x, axis=-1, norm='ortho'):
    ''' expect x as m*n matrix '''
    return fftshift(fft(ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)


def ifftc(x, axis=-1, norm='ortho'):
    ''' expect x as m*n matrix '''
    return fftshift(ifft(ifftshift(x, axes=axis), axis=axis, norm=norm), axes=axis)


def fft2c(x):
    '''
    Centered fft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    # axes = (len(x.shape)-2, len(x.shape)-1)  # get last 2 axes
    axes = (-2, -1)  # get last 2 axes
    res = fftshift(fft2(ifftshift(x, axes=axes), norm='ortho'), axes=axes)
    return res


def ifft2c(x):
    '''
    Centered ifft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    axes = (-2, -1)  # get last 2 axes
    res = fftshift(ifft2(ifftshift(x, axes=axes), norm='ortho'), axes=axes)
    return res


def fourier_matrix(rows, cols):
    '''
    parameters:
    rows: number or rows
    cols: number of columns

    return unitary (rows x cols) fourier matrix
    '''
    # from scipy.linalg import dft
    # return dft(rows,scale='sqrtn')

    col_range = np.arange(cols)
    row_range = np.arange(rows)
    scale = 1 / np.sqrt(cols)

    coeffs = np.outer(row_range, col_range)
    fourier_matrix = np.exp(coeffs * (-2. * np.pi * 1j / cols)) * scale

    return fourier_matrix


def inverse_fourier_matrix(rows, cols):
    return np.array(np.matrix(fourier_matrix(rows, cols)).getH())


def flip(m, axis):
    """
    ==== > Only in numpy 1.12 < =====

    Reverse the order of elements in an array along the given axis.
    The shape of the array is preserved, but the elements are reordered.
    .. versionadded:: 1.12.0
    Parameters
    ----------
    m : array_like
        Input array.
    axis : integer
        Axis in array, which entries are reversed.
    Returns
    -------
    out : array_like
        A view of `m` with the entries of axis reversed.  Since a view is
        returned, this operation is done in constant time.
    See Also
    --------
    flipud : Flip an array vertically (axis=0).
    fliplr : Flip an array horizontally (axis=1).
    Notes
    -----
    flip(m, 0) is equivalent to flipud(m).
    flip(m, 1) is equivalent to fliplr(m).
    flip(m, n) corresponds to ``m[...,::-1,...]`` with ``::-1`` at position n.
    Examples
    --------
    >>> A = np.arange(8).reshape((2,2,2))
    >>> A
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> flip(A, 0)
    array([[[4, 5],
            [6, 7]],
           [[0, 1],
            [2, 3]]])
    >>> flip(A, 1)
    array([[[2, 3],
            [0, 1]],
           [[6, 7],
            [4, 5]]])
    >>> A = np.random.randn(3,4,5)
    >>> np.all(flip(A,2) == A[:,:,::-1,...])
    True
    """
    if not hasattr(m, 'ndim'):
        m = np.asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]


def rot90_nd(x, axes=(-2, -1), k=1):
    """Rotates selected axes"""
    def flipud(x):
        return flip(x, axes[0])

    def fliplr(x):
        return flip(x, axes[1])

    x = np.asanyarray(x)
    if x.ndim < 2:
        raise ValueError("Input must >= 2-d.")
    k = k % 4
    if k == 0:
        return x
    elif k == 1:
        return fliplr(x).swapaxes(*axes)
    elif k == 2:
        return fliplr(flipud(x))
    else:
        # k == 3
        return fliplr(x.swapaxes(*axes))
