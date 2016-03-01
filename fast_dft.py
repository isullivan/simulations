import numpy as np


def fast_dft(amp, x_loc, y_loc, x_size=None, y_size=None, no_fft=False, pad_kernel=2):
    """!A fast DFT approximation from floating-point locations to a regular 2D grid
        that computes the sinc-interpolated values in image space, but only for
        pixels above the set threshold for significance.

        It is safest to use the default value for threshold!
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
    x_size_kernel = int(x_size * pad_kernel)
    y_size_kernel = int(y_size * pad_kernel)

    x_pix = input_type_check([int(num) for num in np.floor(x_loc)])
    y_pix = input_type_check([int(num) for num in np.floor(y_loc)])

    dx_arr = x_loc - np.floor(x_loc)
    dy_arr = y_loc - np.floor(y_loc)
    n_src = len(x_loc)

    x0 = int(np.min(x_pix) - round(x_size_kernel / 2.0))
    y0 = int(np.min(y_pix) - round(y_size_kernel / 2.0))

    x_size_full = x_size_kernel * 2
    y_size_full = y_size_kernel * 2
    x_pix -= x0
    y_pix -= y0
    y_low = y_pix - int(round(y_size_kernel / 2.0))
    y_high = y_low + y_size_kernel
    x_low = x_pix - int(round(x_size_kernel / 2.0))
    x_high = x_low + x_size_kernel

    xv0 = np.arange(x_size_kernel, dtype=np.float64) - int(round(x_size_kernel / 2.0))
    yv0 = np.arange(y_size_kernel, dtype=np.float64) - int(round(y_size_kernel / 2.0))
    x_sign = np.power(-1.0, xv0)
    y_sign = np.power(-1.0, yv0)

    def kernel_1d(delta_arr, sign, locs, size):
        for delta in delta_arr:
            if delta == 0:
                kernel = np.zeros(size, dtype=np.float64)
                kernel[size // 2] = 1.0
            else:
                kernel = np.sin(-pi * delta) / (pi * (locs - delta))
                kernel *= sign
            yield np.matrix(kernel)
            # yield kernel
    kernel_x_gen = kernel_1d(dx_arr, x_sign, xv0, x_size_kernel)
    kernel_y_gen = kernel_1d(dy_arr, y_sign, yv0, y_size_kernel)

    if multi_catalog:
        model_img_full = [np.zeros((y_size_full, x_size_full), dtype=np.float64) for _i in range(n_cat)]
        for _i in range(n_src):
            kernel_single = np.outer(next(kernel_x_gen), next(kernel_y_gen))
            for ci in range(n_cat):
                model_img_full[ci][y_low[_i]:y_high[_i], x_low[_i]:x_high[_i]] += amp[_i, ci] * kernel_single
    else:
        model_img_full = np.zeros((y_size_full, x_size_full), dtype=np.float64)
        for _i in range(n_src):
            # If there is only a single set of amplitudes it is more efficient to multiply by amp in 1D
            kernel_single = np.outer(amp[_i] * next(kernel_x_gen), next(kernel_y_gen))
            model_img_full[y_low[_i]:y_high[_i], x_low[_i]:x_high[_i]] += kernel_single

    def alias_image(model_img_full):
        x_low_img = int(np.max((x0, 0)))
        y_low_img = int(np.max((y0, 0)))
        x_low_full = int(np.max((-x0, 0)))
        y_low_full = int(np.max((-y0, 0)))

        x_high_img = int(np.min((x0 + x_size_full, x_size)))
        y_high_img = int(np.min((y0 + y_size_full, y_size)))
        x_high_full = x_high_img - x_low_img + x_low_full
        y_high_full = y_high_img - y_low_img + y_low_full

        model_img = np.zeros((y_size, x_size), dtype=np.float64)
        model_img[y_low_img:y_high_img, x_low_img:x_high_img] = \
            model_img_full[y_low_full:y_high_full, x_low_full:x_high_full]

        if x_low_full > 0:
            full_view = model_img_full[y_low_full: y_high_full, 0: x_low_full]
            img_view = model_img[y_low_img: y_high_img, x_size - x_low_full: x_size]
            img_view += full_view
            # img_view = np.where(np.abs(img_view) > np.abs(full_view), img_view, full_view)

        if y_low_full > 0:
            full_view = model_img_full[0: y_low_full, x_low_full: x_high_full]
            img_view = model_img[y_size - y_low_full: y_size, x_low_img: x_high_img]
            img_view += full_view
            # img_view = np.where(np.abs(img_view) > np.abs(full_view), img_view, full_view)

        if x_high_full < x_size_full:
            full_view = model_img_full[y_low_full: y_high_full, x_high_full: x_size_full]
            img_view = model_img[y_low_img: y_high_img, 0: x_size_full - x_high_full]
            img_view += full_view
            # img_view = np.where(np.abs(img_view) > np.abs(full_view), img_view, full_view)

        if y_high_full < y_size_full:
            full_view = model_img_full[y_high_full: y_size_full, x_low_full: x_high_full]
            img_view = model_img[0: y_size_full - y_high_full, x_low_img: x_high_img]
            img_view += full_view
            # img_view = np.where(np.abs(img_view) > np.abs(full_view), img_view, full_view)
        return(model_img)

    if multi_catalog:
        model_img = []
        for ci in range(n_cat):
            model_img.append(alias_image(model_img_full[ci]))
    else:
        model_img = alias_image(model_img_full)
    return(model_img)


def input_type_check(var):
    """Ensure that the parameters are iterable."""
    if not hasattr(var, '__iter__'):
        var = [var]
    if type(var) != np.ndarray:
        var = np.array(var)
    return(var)
