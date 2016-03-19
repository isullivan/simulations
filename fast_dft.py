from __future__ import division
import numpy as np


def fast_dft(amplitudes, x_loc, y_loc, x_size=None, y_size=None, no_fft=True, kernel_radius=None, **kwargs):
    """!A fast DFT approximation from floating-point locations to a regular 2D grid
        that computes the sinc-interpolated values in image space.
    """
    pi = np.pi
    if kernel_radius is None:
        kernel_radius = 10

    amplitudes = input_type_check(amplitudes)
    x_loc = input_type_check(x_loc)
    y_loc = input_type_check(y_loc)
    if amplitudes.ndim > 1:
        n_cat = amplitudes.shape[1]
        multi_catalog = True
    else:
        n_cat = 1
        multi_catalog = False

    if y_size is None:
        y_size = x_size

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
            yield kernel
    kernel_x_gen = kernel_1d(x_loc, x_size)
    kernel_y_gen = kernel_1d(y_loc, y_size)

    if multi_catalog:
        def kernel_circle_inds(x_loc, y_loc, kernel_radius=None):
            ind_radius = int(4 * kernel_radius)
            x_i0, y_i0 = np.meshgrid(np.arange(2.0 * ind_radius), np.arange(2.0 * ind_radius))
            x_pix_arr = np.round(x_loc)
            y_pix_arr = np.round(y_loc)
            taper_filter = np.hanning(2 * ind_radius)
            taper_filter /= taper_filter[ind_radius - kernel_radius]
            for src_i in range(len(x_loc)):
                x_pix = int(x_pix_arr[src_i])
                y_pix = int(y_pix_arr[src_i])
                dx = x_loc[src_i] - x_pix + ind_radius
                dy = y_loc[src_i] - y_pix + ind_radius

                test_image = np.sqrt((x_i0 - dx)**2.0 + (y_i0 - dy)**2.0)
                test_image[ind_radius - kernel_radius: ind_radius + kernel_radius, :] = ind_radius
                test_image[:, ind_radius - kernel_radius: ind_radius + kernel_radius] = ind_radius
                if x_pix < ind_radius:
                    test_image[:, 0: ind_radius - x_pix] = ind_radius
                if x_pix > x_size - ind_radius:
                    test_image[:, x_size - ind_radius - x_pix:] = ind_radius
                if y_pix < ind_radius:
                    test_image[0: ind_radius - y_pix, :] = ind_radius
                if y_pix > y_size - ind_radius:
                    test_image[y_size - ind_radius - y_pix:, :] = ind_radius
                y_i, x_i = np.where(test_image < ind_radius)
                taper = taper_filter[y_i] * taper_filter[x_i]
                x_i += x_pix - ind_radius
                y_i += y_pix - ind_radius
                yield x_i
                yield y_i
                yield taper

        amp_arr = [amplitudes[_i, :] for _i in range(len(x_loc))]
        model_img = [np.zeros((y_size, x_size)) for c_i in range(n_cat)]
        x_pix = (int(np.round(xv)) for xv in x_loc)
        y_pix = (int(np.round(yv)) for yv in y_loc)
        kernel_ind_gen = kernel_circle_inds(x_loc, y_loc, kernel_radius=kernel_radius)
        for amp in amp_arr:
            kernel_x = next(kernel_x_gen)
            kernel_y = next(kernel_y_gen)
            kernel_single = np.outer(kernel_y, kernel_x)
            x_c = next(x_pix)
            y_c = next(y_pix)
            x0 = x_c - kernel_radius
            if x0 < 0:
                x0 = 0
            x1 = x_c + kernel_radius
            if x1 > x_size:
                x1 = x_size
            y0 = y_c - kernel_radius
            if y0 < 0:
                y0 = 0
            y1 = y_c + kernel_radius
            if y1 > y_size:
                y1 = y_size
            kernel_single[y0:y1, x0:x1] = kernel_single[y0:y1, x0:x1] / 2.0
            x_i = next(kernel_ind_gen)
            y_i = next(kernel_ind_gen)
            taper = next(kernel_ind_gen)
            for c_i, model in enumerate(model_img):
                model[y0:y1, :] += amp[c_i] * kernel_single[y0:y1, :]
                model[:, x0:x1] += amp[c_i] * kernel_single[:, x0:x1]
                if len(y_i) > 0:
                    model[y_i, x_i] += amp[c_i] * kernel_single[y_i, x_i] * taper
    else:
        # If there is only a single set of amplitudes it is more efficient to multiply by amp in 1D
        model_img = np.sum((np.outer(amp * next(kernel_y_gen), next(kernel_x_gen)) for amp in amplitudes))
    return(model_img)


def input_type_check(var):
    """Ensure that the parameters are iterable."""
    if not hasattr(var, '__iter__'):
        var = [var]
    if type(var) != np.ndarray:
        var = np.array(var)
    return(var)
