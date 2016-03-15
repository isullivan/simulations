from __future__ import division
import numpy as np


def fast_dft(amp, x_loc, y_loc, x_size=None, y_size=None, no_fft=False, threshold=None):
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

    # 1/2 value of kernel_test along either axis and one bin to either side at the edge of the image.
    if threshold is None:
        threshold = 1.0 / (((2.0 * pi) ** 2.0) * np.max((x_size, y_size)))
    print("Threshold used:", threshold)

    kernel_x_size = x_size * 2
    kernel_y_size = y_size * 2

    # pre-compute the indices of the convolution kernel.
    class KernelObj:
        kernel_test_maxval = 100.0 / threshold

        def __init__(self, x_size, y_size):
            xvals, yvals = np.meshgrid(np.arange(x_size), np.arange(y_size), indexing='xy')
            self.xvals = xvals  # - x_size // 2
            self.yvals = yvals  # - y_size // 2
            self.array = np.zeros((y_size, x_size), dtype=np.float64)

        def profile(self, x_offset=0, y_offset=0, kernel_min=1, kernel_max=kernel_test_maxval):
            x_profile = np.clip(abs(pi * (self.xvals - x_offset)), kernel_min, kernel_max)
            y_profile = np.clip(abs(pi * (self.yvals - y_offset)), kernel_min, kernel_max)
            array = 1.0 / (x_profile * y_profile)
            # We want to be sure to capture one pixel beyond our given threshold
            for axis in [0, 1]:
                array += (np.roll(array, 1, axis=axis) + np.roll(array, -1, axis=axis)) / 2.0
            self.array = array

        def threshold_inds(self, threshold):
            return(np.where(self.array >= threshold))

    KernelTest = KernelObj(np.ceil(kernel_x_size), np.ceil(kernel_y_size))
    # for x_off in [0, x_size * 2, -x_size * 2]:
    #    for y_off in [0, y_size * 2, -y_size * 2]:
    #        KernelTest.profile(x_offset=x_off, y_offset=y_off)
    KernelTest.profile(x_offset=x_size, y_offset=y_size)
    xv_k_i, yv_k_i = KernelTest.threshold_inds(threshold)
    xv_k_i -= kernel_x_size // 2
    yv_k_i -= kernel_y_size // 2
    fill_factor = len(xv_k_i) / (x_size * y_size)

    print("Fraction of pixels used:", 100 * fill_factor)

    n_src = len(x_loc)
    x_pix = [int(x) for x in x_loc]
    y_pix = [int(y) for y in y_loc]

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
            yield kernel  # np.matrix(kernel)
    kernel_x_gen = kernel_1d(x_loc, x_size)
    kernel_y_gen = kernel_1d(y_loc, y_size)

    if multi_catalog:
        def kernel_inds(x_inds, y_inds, x_pix, y_pix, size):
            for _i in range(len(x_pix)):
                x_inds_use = x_inds + x_pix[_i]
                y_inds_use = y_inds + y_pix[_i]
                i1 = np.where((x_inds_use >= 0) & (y_inds_use >= 0)
                              & (x_inds_use < size) & (y_inds_use < size))
                i1 = i1[0]
                # y_inds_use = y_inds[i1] - y_pix - size // 2
                # i2 = np.where(y_inds_use >= 0 and y_inds_use < size)
                # i2 = i2[0]
                # i_use = [x_ind >= 0 and x_ind < size and y_ind >= 0 and y_ind < size
                # for x_ind,y_ind in zip(x_inds_use, y_inds_use)]

                # yield y_inds_use[i1] + size * x_inds_use[i1]
                yield x_inds_use[i1]
                yield y_inds_use[i1]

        model_img = [np.zeros((y_size, x_size), dtype=np.float64) for _i in range(n_cat)]
        model_img_arr = np.zeros((x_size * y_size, n_cat))
        kernel_inds_gen = kernel_inds(xv_k_i, yv_k_i, x_pix, y_pix, x_size)
        amp_arr = [amp[_i, :] for _i in range(n_src)]
        # model_img1 = np.sum((np.outer(np.outer(next(kernel_y_gen), next(kernel_x_gen)), amp_val)) for amp_val in amp_arr)
        # model_img = [np.reshape(model_img1[:, ci], (y_size, x_size)) for ci in range(n_cat)]
        if False:
            for amp_vals in amp_arr:
                model_img_arr += np.outer(np.outer(next(kernel_y_gen), next(kernel_x_gen)), amp_vals)
        elif False:
            for amp_vals in amp_arr:
                # inds = next(kernel_inds_gen)
                # model_img_arr = [np.zeros((y_size * x_size), dtype=np.float64) for _i in range(n_cat)]
                x_i = next(kernel_inds_gen)
                y_i = next(kernel_inds_gen)
                inds = x_i + y_i * x_size
                """
                # print("X: ", np.min(x_inds), np.max(x_inds), len(x_inds))
                # print("Y: ", np.min(y_inds), np.max(y_inds), len(y_inds))
                kernel_x = np.array(next(kernel_x_gen))
                kernel_y = np.array(next(kernel_y_gen))
                kernel_vals = kernel_x[0][x_inds] * kernel_y[0][y_inds]
                model_gen = (amp1 * kernel_vals for amp1 in amp[_i, :])
                for model in model_img:
                    model[y_inds, x_inds] += next(model_gen)
                """
                kernel_x = next(kernel_x_gen)
                kernel_y = next(kernel_y_gen)
                # kernel_single = np.outer(kernel_y, kernel_x)

                # y_i, x_i = np.where(np.abs(kernel_single) >= threshold)
                # inds = x_i + y_i * x_size
                kernel_single2 = kernel_x[x_i] * kernel_y[y_i]
                # kernel_single2 /= np.sum(kernel_single2)
                """
                if _i == 20:
                    print(len(kernel_single2))
                    print(np.min(y_i), np.max(y_i))
                    print(np.sum(kernel_single2))
                    print(np.max(np.abs(kernel_single2)))
                    temp = np.outer(kernel_y, kernel_x)
                    print(np.max(np.abs(temp)))
                    print(np.sum(temp))
                    temp_i = np.where(np.abs(temp) >= threshold)
                    print(np.sum(temp[temp_i[0], temp_i[1]]))
                    print(len(temp_i[0]))
                    print(np.min(temp_i[0]), np.max(temp_i[0]))
                """
                model_img_arr[inds, :] += np.outer(kernel_single2, amp_vals)
                """
                model_gen = (amp[_i, ci] * kernel_single for ci in range(n_cat))
                for model in model_img:
                    model += next(model_gen)
                """
        else:
            for amp_vals in amp_arr:
                kernel_x = next(kernel_x_gen)
                kernel_y = next(kernel_y_gen)
                kernel_single = np.outer(kernel_y, kernel_x)

                y_i, x_i = np.where(np.abs(kernel_single) >= threshold / 1000.0)
                inds = x_i + y_i * x_size
                kernel_single2 = kernel_single[y_i, x_i]
                model_img_arr[inds, :] += np.outer(kernel_single2, amp_vals)

        model_img = [np.reshape(model_img_arr[:, ci], (y_size, x_size)) for ci in range(n_cat)]
    else:
        # If there is only a single set of amplitudes it is more efficient to multiply by amp in 1D
        model_img = np.sum((np.outer(amp_val * next(kernel_y_gen), next(kernel_x_gen)) for amp_val in amp))
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
