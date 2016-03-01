"""This uses a fast and accurate DFT approximation to grid data using a supplied PSF as a kernel."""
from numpy.fft import fft2, ifft2, fftshift
import math
from numpy import real, zeros
# import lsst.afw.table as afwTable
# import lsst.afw.geom as afwGeom
from fast_dft import fast_dft
from true_dft import true_dft


def cat_image(catalog=None, bbox=None, psf=None, threshold=None, name=None,
              return_fft=False, pixel_scale=None, use_true=False, pad_image=2, n_cat=1):
    """
    Take a source catalog, bounding box, and psf and construct a simulated image.
    @param[in] catalog source catalog with schema
        three fields are used:
        - {name}_flux
        - {name}_fluxSigma
        - {name}_flag
    @param[in] name name of flux field to use from source catalog
    """
    if name is None:
        # If no name is supplied, find the first entry in the schema in the format *_flux
        schema = catalog.getSchema()
        schema_entry = schema.extract("*_flux", ordered='true')
        fluxName = schema_entry.iterkeys().next()
    else:
        fluxName = name + '_flux'
    if pixel_scale is None:
        # I think most PSF classes have a getFWHM method. The math converts to a sigma for a gaussian.
        fwhm_to_sigma = 1.0 / (2.0 * math.sqrt(2. * math.log(2)))
        pixel_scale = psf.getFWHM() * fwhm_to_sigma

    fluxKey = schema.find(fluxName).key
    x0, y0 = bbox.getBegin()
    # if catalog.isContiguous()
    flux = catalog[fluxKey]
    flux_shape = list(flux.shape)
    flux_shape.append(n_cat)
    flux_use = zeros(flux_shape)
    for ci in range(n_cat):
        flux_use[:, ci] = flux[:] / n_cat
    xv = catalog.getX() - x0
    yv = catalog.getY() - y0
    x_size, y_size = bbox.getDimensions()

    if pad_image > 1:
        x_size_use = x_size * pad_image
        y_size_use = y_size * pad_image
        pad_kernel = 1
    else:
        x_size_use = x_size * pad_image
        y_size_use = y_size * pad_image
        pad_kernel = 2
    x0 = (x_size_use - x_size) // 2
    x1 = x0 + x_size
    y0 = (y_size_use - y_size) // 2
    y1 = y0 + y_size
    xv += x0
    yv += y0

    psf_image = psf.drawImage(scale=pixel_scale, method='no_pixel',
                              nx=x_size_use, ny=y_size_use, offset=[0, 0], use_true_center=False)

    if use_true:
        source_image = true_dft(flux, xv, yv, x_size=x_size_use, y_size=y_size_use,
                                no_fft=True, threshold=threshold)
    else:
        source_image_use = fast_dft(flux_use, xv, yv, x_size=x_size_use, y_size=y_size_use,
                                    no_fft=True, pad_kernel=pad_kernel)
    # This is not how I want to implement DCR sims. It is for timing tests
    source_image = source_image_use[0]
    for _i in range(n_cat - 1):
        source_image += source_image_use[_i + 1]
    # return(source_image)
    convol = fft2(source_image) * fft2(psf_image.array)
    # fft_filter = outer(hanning(y_size_use), hanning(x_size_use))
    # convol *= fftshift(fft_filter)
    return_image = real(fftshift(ifft2(convol)))
    return(return_image[y0:y1, x0:x1])
