from __future__ import division
import numpy as np


def fast_dft(amplitudes, x_loc, y_loc, x_size=None, y_size=None, no_fft=False, threshold=None):
    """!A fast DFT approximation from floating-point locations to a regular 2D grid
        that computes the sinc-interpolated values in image space.
    """
    pi = np.pi

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

    # 1/2 value of kernel_test along either axis and one bin to either side at the edge of the image.
    threshold_use = 1.0 / (((2.0 * pi) ** 2.0) * np.max((x_size, y_size)))
    if threshold is not None:
        if threshold < threshold_use:
            threshold_use = threshold

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
            yield kernel  # np.matrix(kernel)
    kernel_x_gen = kernel_1d(x_loc, x_size)
    kernel_y_gen = kernel_1d(y_loc, y_size)

    def kernel_inds(x_loc, y_loc, threshold=threshold_use):
        for src_i in range(len(x_loc)):
            threshold_x = threshold * np.clip((np.abs(-y_loc[src_i] + np.arange(y_size))), 1.0, None)
            x_min = (int(np.clip(x_loc[src_i] - 1.0 / (pi * t_x) - 1.0, 0, None)) for t_x in threshold_x)
            x_max = (int(np.clip(x_loc[src_i] + 1.0 / (pi * t_x) + 1.0, None, x_size)) for t_x in threshold_x)
            ind_gen = (np.arange(x0, x1) + y_i * x_size for y_i, x0, x1 in zip(range(y_size), x_min, x_max))
            inds = np.array([inds for ind_list in ind_gen for inds in ind_list])
            yield inds

    kernel_test = kernel_inds([x_size // 2], [y_size // 2])
    fill_factor = len(next(kernel_test)) / (x_size * y_size)
    full_switch = False
    if fill_factor > 0.9:
        full_switch = True
    if threshold_use <= 0:
        full_switch = True

    if multi_catalog:
        # model_img_arr = np.zeros((x_size * y_size, n_cat))
        model_img_arr = [np.zeros(x_size * y_size) for c_i in range(n_cat)]
        amp_arr = [amplitudes[_i, :] for _i in range(len(x_loc))]
        if full_switch:
            for amp in amp_arr:
                model_img_arr += np.outer(np.outer(next(kernel_y_gen), next(kernel_x_gen)), amp)
        else:
            print("Threshold used:", threshold_use)
            print("Fraction of pixels used:", 100 * fill_factor)
            kernel_ind_gen = kernel_inds(x_loc, y_loc)
            for amp in amp_arr:
                kernel_x = next(kernel_x_gen)
                kernel_y = next(kernel_y_gen)
                inds = next(kernel_ind_gen)
                # y_i, x_i = np.unravel_index(inds, (y_size, x_size))
                # y_i, x_i = np.where(np.abs(kernel_single) >= threshold_use / pi)
                # inds = x_i + y_i * x_size
                kernel_single = np.ravel(np.outer(kernel_y, kernel_x))
                kernel_vals = kernel_single[inds]
                kernel_gen = (kernel_vals * amp_val for amp_val in amp)
                # model_img_arr[inds, :] += np.outer(kernel_x[x_i] * kernel_y[y_i], amp)
                # model_img_arr[:, inds] += np.outer(kernel_single[inds], amp)
                for model in model_img_arr:
                    model[inds] += next(kernel_gen)
                # model_img_arr[inds, :] += np.outer(kernel_single[inds], amp)

        # model_img = [np.reshape(model_img_arr[:, ci], (y_size, x_size)) for ci in range(n_cat)]
        model_img = [np.reshape(model, (y_size, x_size)) for model in model_img_arr]
    else:
        # If there is only a single set of amplitudes it is more efficient to multiply by amp in 1D
        model_img = np.sum((np.outer(amp * next(kernel_y_gen), next(kernel_x_gen)) for amp in amplitudes))
        """
        model_img = np.zeros((y_size, x_size), dtype=np.float64)

        for _i in range(n_src):
            # If there is only a single set of amplitudes it is more efficient to multiply by amp in 1D
            kernel_single = np.outer(amp[_i] * next(kernel_y_gen), next(kernel_x_gen))
            # kernel_single = amp[_i] * np.outer(next(kernel_y_gen), next(kernel_x_gen))
            model_img += kernel_single
        """
    return(model_img)


def input_type_check(var):
    """Ensure that the parameters are iterable."""
    if not hasattr(var, '__iter__'):
        var = [var]
    if type(var) != np.ndarray:
        var = np.array(var)
    return(var)
