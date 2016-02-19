import numpy as np


def fast_dft(amp, x_loc, y_loc, x_size=None, y_size=None, threshold=None, no_fft=False):
    """!A fast DFT approximation from floating-point locations to a regular 2D grid
        that computes the sinc-interpolated values in image space, but only for
        pixels above the set threshold for significance.

        It is safest to use the default value for threshold!
    """
    pi = np.pi

    # ensure that the parameters are iterable
    def input_type_check(var):
        if not hasattr(var, '__iter__'):
            var = [var]
        if type(var) != np.ndarray:
            var = np.array(var)
        return(var)

    amp = input_type_check(amp)
    x_loc = input_type_check(x_loc)
    y_loc = input_type_check(y_loc)

    if y_size is None:
        y_size = x_size
    x_size_kernel = int(x_size * 2)
    y_size_kernel = int(y_size * 2)
    if threshold is None:
        threshold = 1.0 / (((2.0 * pi) ** 2.0) * np.max((x_size, y_size)))
    # print("Threshold used:", threshold)

    # NOTE: appear to need to use 'ij' (matrix) indexing here, which results in a transposed array, then
    #       take transpose of final image at the end. This is confusing and clearly a bug, but it works
    xv_test, yv_test = np.meshgrid(np.arange(x_size_kernel), np.arange(y_size_kernel), indexing='ij')
    xv_test -= x_size_kernel // 2
    yv_test -= y_size_kernel // 2

    kernel_test_maxval = 1.0 / threshold

    def kernel_profile(x, y, x_offset=0, y_offset=0, kernel_min=1, kernel_max=kernel_test_maxval):
        x_profile = np.clip(abs(np.pi * (xv_test - x_offset)), kernel_min, kernel_max)
        y_profile = np.clip(abs(np.pi * (yv_test - y_offset)), kernel_min, kernel_max)
        return(1.0 / (x_profile * y_profile))

    kernel_test = (kernel_profile(xv_test, yv_test, x_offset=0, y_offset=0)
                   + kernel_profile(xv_test, yv_test, x_offset=x_size, y_offset=0)
                   + kernel_profile(xv_test, yv_test, x_offset=-x_size, y_offset=0)
                   + kernel_profile(xv_test, yv_test, x_offset=0, y_offset=y_size)
                   + kernel_profile(xv_test, yv_test, x_offset=0, y_offset=-y_size)
                   )
    xv_k, yv_k = np.where(kernel_test >= threshold)
    # print("xv_k:", min(xv_k), max(xv_k), len(xv_k))
    # print("yv_k:", min(yv_k), max(yv_k), len(yv_k))
    print("Fraction of pixels used:", 100 * len(xv_k) / (x_size_kernel * y_size_kernel))
    xv_k -= int(round(x_size_kernel / 2.0))
    yv_k -= int(round(y_size_kernel / 2.0))

    x_pix = input_type_check([int(num) for num in np.floor(x_loc)])
    y_pix = input_type_check([int(num) for num in np.floor(y_loc)])
    # print("x_pix:", x_pix)
    # print("y_pix:", y_pix)

    dx_arr = x_loc - x_pix
    dy_arr = y_loc - y_pix
    # print("dx_arr:", dx_arr)
    # print("dy_arr: ", dy_arr)

    x0 = int(np.min(x_pix) + np.min(xv_k))
    y0 = int(np.min(y_pix) + np.min(yv_k))

    x_size_full = int(np.max(x_pix) + np.max(xv_k) - x0 + 1)
    y_size_full = int(np.max(y_pix) + np.max(yv_k) - y0 + 1)
    x_pix -= x0
    y_pix -= y0
    # print('x0:', x0)
    # print('y0:', y0)
    # print('x_size_full:', x_size_full)
    # print('y_size_full:', y_size_full)

    model_img_full = np.zeros((x_size_full, y_size_full), dtype=np.float64)
    xv0 = np.arange(x_size_kernel, dtype=np.float64) - int(round(x_size_kernel / 2.0))
    yv0 = np.arange(y_size_kernel, dtype=np.float64) - int(round(y_size_kernel / 2.0))
    x_sign = np.power(-1.0, xv0)
    y_sign = np.power(-1.0, yv0)

    # pre-compute the indices of the convolution kernel
    xv_k_i = xv_k + int(round(x_size_kernel / 2.0))
    yv_k_i = yv_k + int(round(y_size_kernel / 2.0))
    sin_x = np.sin(-pi * dx_arr)
    sin_y = np.sin(-pi * dy_arr)

    for _i in range(len(amp)):
        if dx_arr[_i] == 0:
            kernel_x = np.zeros(x_size_kernel, dtype=np.float64)
            kernel_x[x_size_kernel // 2] = 1.0
        else:
            kernel_x = (sin_x[_i] / (pi * (xv0 - dx_arr[_i]))
                        + sin_x[_i] / (pi * (x_size_kernel + xv0 - dx_arr[_i] + x0))
                        + sin_x[_i] / (pi * (-x_size_kernel + xv0 - dx_arr[_i] - x0))
                        )
            kernel_x *= x_sign
            # kernel_x /= np.sum(kernel_x)
        if dy_arr[_i] == 0:
            kernel_y = np.zeros(y_size_kernel, dtype=np.float64)
            kernel_y[y_size_kernel // 2] = 1.0
        else:
            kernel_y = (sin_y[_i] / (pi * (yv0 - dy_arr[_i]))
                        + sin_y[_i] / (pi * (y_size_kernel + yv0 - dy_arr[_i] + y0))
                        + sin_y[_i] / (pi * (-y_size_kernel + yv0 - dy_arr[_i] - y0))
                        )
            kernel_y *= y_sign
            # kernel_y /= np.sum(kernel_y)
        kernel_single = kernel_x[xv_k_i] * kernel_y[yv_k_i]
        x_inds = x_pix[_i] + xv_k
        y_inds = y_pix[_i] + yv_k
        model_img_full[x_inds, y_inds] += amp[_i] * kernel_single

    x_low_img = int(np.max((x0, 0)))
    y_low_img = int(np.max((y0, 0)))
    x_high_img = int(np.min((x0 + x_size_full, x_size)))
    y_high_img = int(np.min((y0 + y_size_full, y_size)))
    x_low_full = int(np.max((-x0, 0)))
    y_low_full = int(np.max((-y0, 0)))
    x_high_full = x_high_img - x_low_img + x_low_full
    y_high_full = y_high_img - y_low_img + y_low_full

    model_img = np.zeros((x_size, y_size), dtype=np.float64)
    model_img[x_low_img:x_high_img, y_low_img:y_high_img] = \
        model_img_full[x_low_full:x_high_full, y_low_full:y_high_full]

    """
    if x_low_full > 0:
        model_img[x_size - x_low_full: x_size, y_low_img: y_high_img] = \
            model_img_full[0: x_low_full, y_low_full: y_high_full]
    """
    """
    if y_low_full > 0:
        model_img[x_low_img: x_high_img, y_size - y_low_full: y_size] = \
            model_img_full[x_low_full: x_high_full, 0: y_low_full]
    """
    """
    if x_high_full < x_size:
        model_img[0: x_size_full - x_high_full - 1, y_low_img: y_high_img] = \
            model_img_full[x_high_full + 1: x_size_full, y_low_full: y_high_full]
    """
    """
    if y_high_full < y_size:
        model_img[x_low_img: x_high_img, 0: y_size_full - y_high_full - 1] = \
            model_img_full[x_low_full: x_high_full, y_high_full + 1: y_size_full]
    """
    # IMPORTANT: See note near the beginning explaining why we have to take the transpose on output
    return(model_img.T)


def run(exit=False):
    amp_vec = np.array(3.)
    x_loc = np.array(37.2342)
    y_loc = np.array(93.3423636)
    img = fast_dft(amp_vec, x_loc, y_loc, x_size=128, y_size=128)
    return(img)

if __name__ == "__main__":
    run(True)
