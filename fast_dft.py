import numpy as np


def fast_dft(amp, x_loc, y_loc, x_size=None, y_size=None, no_fft=False):
    """!A fast DFT approximation from floating-point locations to a regular 2D grid
        that computes the sinc-interpolated values in image space.
    """
    pi = np.pi

    amp = input_type_check(amp)
    x_loc = input_type_check(x_loc)
    y_loc = input_type_check(y_loc)
    if amp.ndim > 1:
        n_cat = amp.shape[1]
        multi_catalog = True
    else:
        n_cat = 1
        multi_catalog = False

    if y_size is None:
        y_size = x_size

    n_src = len(x_loc)

    def kernel_1d(locs, size):
        pix = np.arange(size, dtype=np.float64)
        sign = np.power(-1.0, pix)
        for loc in locs:
            offset = np.floor(loc)
            delta = loc - offset
            if delta == 0:
                kernel = np.zeros(size, dtype=np.float64)
                kernel[offset] = 1.0
            else:
                kernel = np.sin(-pi * loc) / (pi * (pix - loc))
                kernel *= sign
            # kernel[0: size - high] += kernel_full[high:]
            # kernel[-low:] += kernel_full[0: low]
            yield np.matrix(kernel)
    kernel_x_gen = kernel_1d(x_loc, x_size)
    kernel_y_gen = kernel_1d(y_loc, y_size)

    if multi_catalog:
        model_img = [np.zeros((y_size, x_size), dtype=np.float64) for _i in range(n_cat)]
        for _i in range(n_src):
            kernel_single = np.outer(next(kernel_x_gen), next(kernel_y_gen))
            for ci in range(n_cat):
                model_img[ci] += amp[_i, ci] * kernel_single
    else:
        model_img = np.zeros((y_size, x_size), dtype=np.float64)
        for _i in range(n_src):
            # If there is only a single set of amplitudes it is more efficient to multiply by amp in 1D
            kernel_single = np.outer(amp[_i] * next(kernel_y_gen), next(kernel_x_gen))
            model_img += kernel_single
    return(model_img)


def input_type_check(var):
    """Ensure that the parameters are iterable."""
    if not hasattr(var, '__iter__'):
        var = [var]
    if type(var) != np.ndarray:
        var = np.array(var)
    return(var)
