"""This uses a fast and accurate DFT approximation to grid data using a supplied PSF as a kernel."""
from numpy.fft import fft2, ifft2, ifftshift
import math
from numpy import real
# import lsst.afw.table as afwTable
# import lsst.afw.geom as afwGeom
from fast_dft import fast_dft


def cat_image(catalog=None, bbox=None, psf=None, threshold=None, name=None,
              return_fft=False, pixel_scale=None):
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
    xv = catalog.getX() - x0
    yv = catalog.getY() - y0
    flux = catalog[fluxKey]
    x_size, y_size = bbox.getDimensions()

    source_image = fast_dft(flux, xv, yv, x_size=x_size, y_size=y_size, no_fft=True, threshold=threshold)

    return(source_image)
    psf_image = psf.drawImage(scale=pixel_scale, method='no_pixel',
                              nx=x_size, ny=y_size, offset=[0.5, 0.5], use_true_center=True)
    convol = fft2(source_image) * fft2(ifftshift(psf_image.array))
    if return_fft:
        return(convol)
    else:
        return(real(ifft2(convol)))
