from __future__ import division
import numpy as np


def grid_star(amplitudes, x_loc, y_loc, x_size=None, y_size=None, psf=None, pixel_scale=None,
              offset=None, photons_per_adu=1e4):

    amplitudes = input_type_check(amplitudes)
    x_loc = input_type_check(x_loc)
    y_loc = input_type_check(y_loc)
    if y_size is None:
        y_size = x_size
    model_image = np.zeros((x_size, y_size))

    if offset is None:
        offset_gen = ((x - x_size // 2, y - y_size // 2) for x, y in zip(x_loc, y_loc))
    else:
        offset_gen = ((x - x_size // 2 + offset[0], y - y_size // 2 + offset[1])
                      for x, y in zip(x_loc, y_loc))

    for amp in amplitudes:
        psf_src = psf.drawImage(scale=pixel_scale, method='fft', offset=next(offset_gen),
                                use_true_center=True, nx=x_size, ny=y_size)
        model_image += amp * psf_src.array
    return(model_image)


def input_type_check(var):
    """Ensure that the parameters are iterable."""
    if not hasattr(var, '__iter__'):
        var = [var]
    if type(var) != np.ndarray:
        var = np.array(var)
    return(var)
