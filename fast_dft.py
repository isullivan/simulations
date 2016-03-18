from __future__ import division
import numpy as np


def fast_dft(amplitudes, x_loc, y_loc, x_size=None, y_size=None, no_fft=True,
             threshold=None, kernel_radius=None, **kwargs):
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

    # value of kernel along either axis.
    threshold_use = 1.0 / (2 * pi * np.max((x_size, y_size)))
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
            yield kernel
    kernel_x_gen = kernel_1d(x_loc, x_size)
    kernel_y_gen = kernel_1d(y_loc, y_size)

    if multi_catalog:
        """
        def x_min_fn(xv, threshold_x):
            return np.round(np.clip(xv - 1.0 / (pi * threshold_x) - 1.0, 0, None))

        def x_max_fn(xv, threshold_x):
            return np.round(np.clip(xv + 1.0 / (pi * threshold_x) + 1.0, None, x_size))

        def kernel_inds(x_loc, y_loc, threshold=threshold_use, kernel_radius=None):
            for src_i in range(len(x_loc)):
                threshold_x = threshold * np.clip((np.abs(-y_loc[src_i] + np.arange(y_size))), 1.0, None)
                x_min = x_min_fn(x_loc[src_i], threshold_x)
                x_max = x_max_fn(x_loc[src_i], threshold_x)
                x_min = np.array([int(x0) for x0 in x_min])
                x_max = np.array([int(x1) for x1 in x_max])
                x_pix = np.round(x_loc[src_i])
                y_pix = np.round(y_loc[src_i])
                # ind_gen = (np.arange(x0, x1, dtype=int) + y_i * x_size
                #            for y_i, x0, x1 in zip(range(y_size), x_min, x_max))
                ind_gen = (np.arange(x0, x_pix - kernel_radius, dtype=int) + y_i * x_size
                           for y_i, x0 in enumerate(x_min)
                           if (x_pix - x0 > kernel_radius) & (np.abs(y_pix - y_i) > kernel_radius))
                inds_min = [inds for ind_list in ind_gen for inds in ind_list]
                ind_gen = (np.arange(x_pix + kernel_radius, x1, dtype=int) + y_i * x_size
                           for y_i, x1 in enumerate(x_max)
                           if (x1 - x_pix > kernel_radius) & (np.abs(y_pix - y_i) > kernel_radius))
                inds_max = [inds for ind_list in ind_gen for inds in ind_list]
                inds = np.asarray(inds_min + inds_max)
                x_i = inds % x_size
                y_i = inds // x_size
                yield x_i
                yield y_i

        def kernel_edge_inds(x_loc, y_loc, threshold=threshold_use):
            y_pix = np.round(y_loc)
            for src_i in range(len(x_loc)):
                threshold_x = threshold * np.clip((np.abs(-y_loc[src_i] + np.arange(y_size))), 1.0, None)
                x_min = x_min_fn(x_loc[src_i], threshold_x)
                # Need the extra -1 here since we're not using np.arange()
                x_max = x_max_fn(x_loc[src_i], threshold_x) - 1
                # x_pix = np.round(x_loc[src_i])
                y_pix = np.round(y_loc[src_i])
                inds_min = [int(x0 + y_i * x_size) for y_i, x0 in enumerate(x_min) if x0 > 1]
                inds_max = [int(x1 + y_i * x_size) for y_i, x1 in enumerate(x_max) if x1 < x_size - 1]
                y_i = int(y_pix[src_i])
                yield np.asarray(inds_min[1:y_i])
                yield np.asarray(inds_min[y_i:-1])
                yield np.asarray(inds_max[1:y_i])
                yield np.asarray(inds_max[y_i:-1])
        kernel_test = kernel_inds([x_size // 2], [y_size // 2], kernel_radius=kernel_radius)
        fill_factor = len(next(kernel_test)) / (x_size * y_size)
        full_switch = False
        if fill_factor > 0.3:
            full_switch = True
        if threshold_use <= 0:
            full_switch = True
        """

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
        if False:  # full_switch
            model_img = [np.zeros((y_size, x_size)) for c_i in range(n_cat)]
            for amp in amp_arr:
                kernel = np.outer(next(kernel_y_gen), next(kernel_x_gen))
                for c_i, model in enumerate(model_img):
                    model += kernel * amp[c_i]
        else:
            model_img = [np.zeros((y_size, x_size)) for c_i in range(n_cat)]
            x_pix = (int(np.round(xv)) for xv in x_loc)
            y_pix = (int(np.round(yv)) for yv in y_loc)
            # print("Threshold used:", threshold_use)
            # print("Fraction of pixels used:", 100 * fill_factor)
            # kernel_ind_gen = kernel_inds(x_loc, y_loc, kernel_radius=kernel_radius)
            kernel_ind_gen = kernel_circle_inds(x_loc, y_loc, kernel_radius=kernel_radius)
            # edge_ind_gen = kernel_edge_inds(x_loc, y_loc)
            # edge_bleed = 2.0
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

            # model_img_arr = [np.zeros(y_size * x_size) for c_i in range(n_cat)]
            """
            
            for amp in amp_arr:
                kernel_x = next(kernel_x_gen)
                kernel_y = next(kernel_y_gen)
                # x_i = next(kernel_ind_gen)
                # y_i = next(kernel_ind_gen)
                # inds = x_i + y_i * x_size
                inds = next(kernel_ind_gen)
                # x_i = inds % x_size
                # y_i = inds // x_size
                # y_i, x_i = np.unravel_index(inds, (y_size, x_size))
                # y_i, x_i = np.where(np.abs(kernel_single) >= threshold_use / pi)
                # inds = x_i + y_i * x_size
                kernel_single = np.ravel(np.outer(kernel_y, kernel_x))
                # l: left, r: right, t: top, b: bottom
                bl_edge_inds = next(edge_ind_gen)
                tl_edge_inds = next(edge_ind_gen)
                br_edge_inds = next(edge_ind_gen)
                tr_edge_inds = next(edge_ind_gen)
                if len(bl_edge_inds) > 0:
                    kernel_single[bl_edge_inds] += (kernel_single[bl_edge_inds - 1]) / edge_bleed
                if len(tl_edge_inds) > 0:
                    kernel_single[tl_edge_inds] += kernel_single[tl_edge_inds - 1] / edge_bleed
                if len(br_edge_inds) > 0:
                    kernel_single[br_edge_inds] += kernel_single[br_edge_inds + 1] / edge_bleed
                if len(tr_edge_inds) > 0:
                    kernel_single[tr_edge_inds] += kernel_single[tr_edge_inds + 1] / edge_bleed
                kernel_vals = kernel_single[inds]
                # model_img_arr[inds, :] += np.outer(kernel_x[x_i] * kernel_y[y_i], amp)
                # model_img_arr[:, inds] += np.outer(kernel_single[inds], amp)
                # kernel_vals = kernel_x[x_i] * kernel_y[y_i]
                for c_i, model in enumerate(model_img_arr):
                    model[inds] += kernel_vals * amp[c_i]
                # model_img_arr[inds, :] += np.outer(kernel_single[inds], amp)

            model_img = [np.reshape(model, (y_size, x_size)) for model in model_img_arr]
            """

        # model_img = [np.reshape(model_img_arr[:, ci], (y_size, x_size)) for ci in range(n_cat)]
        # model_img = [np.reshape(model, (y_size, x_size)) for model in model_img_arr]
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
