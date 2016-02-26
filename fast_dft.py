import numpy as np


def fast_dft(amp, x_loc, y_loc, x_size=None, y_size=None, threshold=None, 
             no_fft=False, pad_kernel=2):
    """!A fast DFT approximation from floating-point locations to a regular 2D grid
        that computes the sinc-interpolated values in image space, but only for
        pixels above the set threshold for significance.

        It is safest to use the default value for threshold!
    """
    pi = np.pi

    amp = input_type_check(amp)
    x_loc = input_type_check(x_loc)
    y_loc = input_type_check(y_loc)

    if y_size is None:
        y_size = x_size
    x_size_kernel = int(x_size * pad_kernel)
    y_size_kernel = int(y_size * pad_kernel)

    # 1/2 value of kernel_test along either axis and one bin to either side at the edge of the image.
    threshold_max = 1.0 / (((2.0 * pi) ** 2.0) * np.max((x_size_kernel, y_size_kernel)))
    if threshold is None:
        threshold_use = threshold_max
    else:
        threshold_use = np.min([threshold, threshold_max])
    print("Threshold used:", threshold_use)

    # pre-compute the indices of the convolution kernel.
    class KernelObj:
        kernel_test_maxval = 100.0 / threshold_use

        def __init__(self, x_size, y_size):
            xvals, yvals = np.meshgrid(np.arange(x_size), np.arange(y_size), indexing='xy')
            self.xvals = xvals - x_size // 2
            self.yvals = yvals - y_size // 2
            self.array = np.zeros((y_size, x_size), dtype=np.float64)

        def profile(self, x_offset=0, y_offset=0, kernel_min=1, kernel_max=kernel_test_maxval):
            x_profile = np.clip(abs(pi * (self.xvals - x_offset)), kernel_min, kernel_max)
            y_profile = np.clip(abs(pi * (self.yvals - y_offset)), kernel_min, kernel_max)
            array = 1.0 / (x_profile * y_profile)
            # We want to be sure to capture one pixel beyond our given threshold
            for axis in [0, 1]:
                array += (np.roll(array, 1, axis=axis) + np.roll(array, -1, axis=axis)) / 2.0
            self.array += array

        def threshold_inds(self, threshold):
            return(np.where(self.array >= threshold))

    KernelTest = KernelObj(x_size_kernel, y_size_kernel)
    for x_off in [0, x_size_kernel, -x_size_kernel]:
        for y_off in [0, y_size_kernel, -y_size_kernel]:
            KernelTest.profile(x_offset=x_off, y_offset=y_off)
    xv_k_i, yv_k_i = KernelTest.threshold_inds(threshold_use)

    print("Fraction of pixels used:", 100 * len(xv_k_i) / (x_size_kernel * y_size_kernel))
    xv_k = xv_k_i - int(round(x_size_kernel / 2.0))
    yv_k = yv_k_i - int(round(y_size_kernel / 2.0))

    x_pix = input_type_check([int(num) for num in np.floor(x_loc)])
    y_pix = input_type_check([int(num) for num in np.floor(y_loc)])

    dx_arr = x_loc - np.floor(x_loc)
    dy_arr = y_loc - np.floor(y_loc)
    n_src = len(x_loc)

    x0 = int(np.min(x_pix) + np.min(xv_k))
    y0 = int(np.min(y_pix) + np.min(yv_k))

    x_size_full = x_size_kernel * 2
    y_size_full = y_size_kernel * 2
    x_pix -= x0
    y_pix -= y0
    """
    print('x0:', x0)
    print('y0:', y0)
    print('x_size_full:', x_size_full)
    print('y_size_full:', y_size_full)
    """

    model_img_full = np.zeros((x_size_full, y_size_full), dtype=np.float64)
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
            yield kernel
    kernel_x_gen = kernel_1d(dx_arr, x_sign, xv0, x_size_full)
    kernel_y_gen = kernel_1d(dy_arr, y_sign, yv0, y_size_full)
    
    """
    def kernel_1d(delta=None, sign_arr=None, locs=None, size=None):
        if delta == 0:
            kernel = np.zeros(size, dtype=np.float64)
            kernel[size // 2] = 1.0
        else:
            # kernel = np.sin(-pi * delta) * (1.0 / (pi * (locs - delta))
            #                  + 1.0 / (pi * (size + locs - delta + x0))
            #                  + 1.0 / (pi * (-size + locs - delta - x0))
            #                  )
            kernel = np.sin(-pi * delta) / (pi * (locs - delta))
            kernel *= sign_arr
        return(kernel)
    """
    for _i in range(n_src):
        kernel_x = next(kernel_x_gen)
        kernel_y = next(kernel_y_gen)
        kernel_x = kernel_x[xv_k_i]
        kernel_y = kernel_y[yv_k_i]
        kernel_single = kernel_x * kernel_y
        x_inds = x_pix[_i] + xv_k
        y_inds = y_pix[_i] + yv_k
        model_img_full[y_inds, x_inds] += amp[_i] * kernel_single

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
        # print("Aliasing x low")
        full_view = model_img_full[y_low_full: y_high_full, 0: x_low_full]
        img_view = model_img[y_low_img: y_high_img, x_size - x_low_full: x_size]
        img_view += full_view
        # img_view = np.where(np.abs(img_view) > np.abs(full_view), img_view, full_view)
    
    if y_low_full > 0:
        # print("Aliasing y low")
        full_view = model_img_full[0: y_low_full, x_low_full: x_high_full]
        img_view = model_img[y_size - y_low_full: y_size, x_low_img: x_high_img]
        img_view += full_view
        # img_view = np.where(np.abs(img_view) > np.abs(full_view), img_view, full_view)
    
    if x_high_full < x_size_full:
        # print("Aliasing x high")
        full_view = model_img_full[y_low_full: y_high_full, x_high_full: x_size_full]
        img_view = model_img[y_low_img: y_high_img, 0: x_size_full - x_high_full]
        img_view += full_view
        # img_view = np.where(np.abs(img_view) > np.abs(full_view), img_view, full_view)

    if y_high_full < y_size_full:
        # print("Aliasing y high")
        full_view = model_img_full[y_high_full: y_size_full, x_low_full: x_high_full]
        img_view = model_img[0: y_size_full - y_high_full, x_low_img: x_high_img]
        img_view += full_view
        # img_view = np.where(np.abs(img_view) > np.abs(full_view), img_view, full_view)
    
    return(model_img)


def input_type_check(var):
    """Ensure that the parameters are iterable."""
    if not hasattr(var, '__iter__'):
        var = [var]
    if type(var) != np.ndarray:
        var = np.array(var)
    return(var)


def run(exit=False):
    amp_vec = np.array(3.)
    x_loc = np.array(37.2342)
    y_loc = np.array(93.3423636)
    img = fast_dft(amp_vec, x_loc, y_loc, x_size=128, y_size=128)
    return(img)

if __name__ == "__main__":
    run(True)
