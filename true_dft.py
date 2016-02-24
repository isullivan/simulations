import numpy as np


def true_dft(amp, x_loc, y_loc, x_size=None, y_size=None, threshold=None, no_fft=False):
    if y_size is None:
        y_size = x_size
    no_fft = False
    pi = np.pi
    x_use = x_loc # - x_size / 2.0
    y_use = y_loc # - y_size / 2.0
    x_use *= (2.0 * pi / x_size)
    y_use *= (2.0 * pi / x_size)
    xvals, yvals = np.meshgrid(np.arange(x_size), np.arange(y_size), indexing='xy')

    xvals -= x_size / 2
    yvals -= y_size / 2
    x_use = np.matrix(x_use)
    y_use = np.matrix(y_use)
    xvals = np.matrix(xvals.ravel())
    yvals = np.matrix(yvals.ravel())
    amp_use = np.matrix(np.array(amp))

    phase = xvals.T * x_use + yvals.T * y_use
    cos_term = np.cos(-phase)
    sin_term = np.sin(-phase)
    source_uv_real_vals = cos_term * amp_use.T
    source_uv_im_vals = sin_term * amp_use.T
    fft_vals = 1j * source_uv_im_vals
    fft_vals += source_uv_real_vals
    fft_vals = np.fft.fftshift(fft_vals.reshape(y_size, x_size))

    if no_fft:
        return(fft_vals)
    else:
        return(np.real(np.fft.ifft2(fft_vals)))
