"""Function to generate simulated catalogs with reproduceable source spectra to feed into fast_dft."""
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
# import lsst.afw.image as afwImage
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import math
from scipy import constants
import galsim
from fast_dft import fast_dft
bbox_init = afwGeom.Box2I(afwGeom.PointI(0, 0), afwGeom.ExtentI(512, 512))


def cat_image(catalog=None, bbox=bbox_init, name=None, psf=None, pixel_scale=None, pad_image=1.1,
              sky_noise=0.0, instrument_noise=0.0, **kwargs):
    """Wrapper that takes a catalog of stars and simulates an image."""
    if catalog is None:
        catalog = cat_sim(bbox=bbox, name=name, **kwargs)
    x_size, y_size = bbox.getDimensions()
    x0, y0 = bbox.getBegin()
    if name is None:
        # If no name is supplied, find the first entry in the schema in the format *_flux
        schema = catalog.getSchema()
        schema_entry = schema.extract("*_flux", ordered='true')
        fluxName = schema_entry.iterkeys().next()
    else:
        fluxName = name + '_flux'

    if psf is None:
        psf = galsim.Kolmogorov(fwhm=3)
    if pixel_scale is None:
        # I think most PSF classes have a getFWHM method. The math converts to a sigma for a gaussian.
        fwhm_to_sigma = 1.0 / (2.0 * math.sqrt(2. * math.log(2)))
        pixel_scale = psf.getFWHM() * fwhm_to_sigma

    fluxKey = schema.find(fluxName).key
    temperatureKey = schema.find("temperature").key
    x0, y0 = bbox.getBegin()
    # if catalog.isContiguous()
    flux = catalog[fluxKey]
    temperatures = catalog[temperatureKey]
    xv = catalog.getX() - x0
    yv = catalog.getY() - y0

    if pad_image > 1:
        x_size_use = int(x_size * pad_image)
        y_size_use = int(y_size * pad_image)
    else:
        x_size_use = int(x_size * pad_image)
        y_size_use = int(y_size * pad_image)
    x0 = (x_size_use - x_size) // 2
    x1 = x0 + x_size
    y0 = (y_size_use - y_size) // 2
    y1 = y0 + y_size
    xv += x0
    yv += y0

    psf_image = psf.drawImage(scale=pixel_scale, method='no_pixel',
                              nx=x_size_use, ny=y_size_use, offset=[0, 0], use_true_center=False)

    source_image = fast_dft(flux, xv, yv, x_size=x_size_use, y_size=y_size_use, no_fft=True)
    if sky_noise > 0:
        source_image += np.random.normal(scale=sky_noise, size=(y_size_use, x_size_use))

    """
    # This is not how I want to implement DCR sims. It is for timing tests
    source_image = source_image_use[0]
    for _i in range(n_cat - 1):
        source_image += source_image_use[_i + 1]
    """
    # return(source_image)
    convol = fft2(source_image) * fft2(psf_image.array)
    # fft_filter = outer(hanning(y_size_use), hanning(x_size_use))
    # convol *= fftshift(fft_filter)
    return_image = np.real(fftshift(ifft2(convol)))
    if instrument_noise > 0:
        return_image += np.random.normal(scale=instrument_noise, size=(y_size_use, x_size_use))
    return(return_image[y0:y1, x0:x1])


def cat_sim(bbox=None, seed=None, n_star=None, n_galaxy=None, name=None):
    """Wrapper function that generates a semi-realistic catalog of stars."""
    schema = afwTable.SourceTable.makeMinimalSchema()
    if name is None:
        name = "sim"
    fluxName = name + "_flux"
    flagName = name + "_flag"
    fluxSigmaName = name + "_fluxSigma"
    schema.addField(fluxName, type="D")
    schema.addField(fluxSigmaName, type="D")
    schema.addField(flagName, type="D")
    schema.addField(name + "_Centroid_x", type="D")
    schema.addField(name + "_Centroid_y", type="D")
    schema.addField("temperature", type="D")
    schema.getAliasMap().set('slot_Centroid', name + '_Centroid')

    x_size, y_size = bbox.getDimensions()
    x0, y0 = bbox.getBegin()
    temperature, luminosity = stellar_distribution(seed=seed, n_star=n_star)
    rand_gen = np.random
    if seed is not None:
        rand_gen.seed(seed + 1)  # ensure that we use a different seed than stellar_distribution.
    x = rand_gen.uniform(x0, x0 + x_size, n_star)
    y = rand_gen.uniform(y0, y0 + y_size, n_star)
    flux = luminosity * np.abs(np.random.normal(scale=1e4, size=n_star) / 100.)

    catalog = afwTable.SourceCatalog(schema)
    fluxKey = schema.find(fluxName).key
    flagKey = schema.find(flagName).key
    fluxSigmaKey = schema.find(fluxSigmaName).key
    temperatureKey = schema.find("temperature").key
    centroidKey = afwTable.Point2DKey(schema["slot_Centroid"])
    for _i in range(n_star):
        source_test_centroid = afwGeom.Point2D(x[_i], y[_i])
        source = catalog.addNew()
        source.set(fluxKey, flux[_i])
        source.set(centroidKey, source_test_centroid)
        source.set(fluxSigmaKey, 0.)
        source.set(temperatureKey, temperature[_i])
        source.set(flagKey, False)
    return(catalog.copy(True))  # Return a copy to make sure it is contiguous in memory.


def random_star(seed=None, temperature=5600, flux=1.0, Band=None):
    """Generate a blackbody radiation spectrum at a given temperature over a range of wavelengths.
        The output is normalized to sum to the given flux.
        If a seed is supplied, noise can be added to the final spectrum before normalization.
    """
    h = constants.Planck
    kb = constants.Boltzmann
    c = constants.speed_of_light

    prefactor = 2.0 * (kb * temperature)**4. / (h**3 * c**2)

    def radiance_expansion(x, nterms):
        for n in range(1, nterms + 1):
            poly_term = x**3 / n + 3 * x**2 / n**2 + 6 * x / n**3 + 6 / n**4
            exp_term = np.exp(-n * x)
            yield(poly_term * exp_term)

    def integral(generator):
        """Simple wrapper to make the math more apparent."""
        return(np.sum(var for var in generator))

    def radiance_calc(temperature, wavelength_start, wavelength_end, nterms=3):
        nu1 = c / (wavelength_start / 1E9)
        nu2 = c / (wavelength_end / 1E9)
        x1 = h * nu1 / (kb * temperature)
        x2 = h * nu2 / (kb * temperature)
        radiance1 = radiance_expansion(x1, nterms)
        radiance2 = radiance_expansion(x2, nterms)
        radiance_integral1 = prefactor * integral(radiance1)
        radiance_integral2 = prefactor * integral(radiance2)
        return(radiance_integral1 - radiance_integral2)

    normalization = flux / radiance_calc(temperature, Band.start, Band.end)
    for wave_start, wave_end in wavelength_iterator(Band):
        yield(normalization * radiance_calc(temperature, wave_start, wave_end))


class BandDefine:
    """Define the wavelength range and resolution for a given ugrizy band."""

    def __init__(self, band_name='g', step=10):
        band_dict = {'u': (324.0, 395.0), 'g': (405.0, 552.0), 'r': (552.0, 691.0),
                     'i': (818.0, 921.0), 'z': (922.0, 997.0), 'y': (975.0, 1075.0)}
        band_range = band_dict[band_name]
        self.start = band_range[0]
        self.end = band_range[1]
        self.step = step


def wavelength_iterator(band_obj):
    """Define iterator to ensure that loops over wavelength are consistent."""
    wave_start = band_obj.start
    while wave_start < band_obj.end:
        wave_end = wave_start + band_obj.step
        if wave_end > band_obj.end:
            wave_end = band_obj.end
        yield((wave_start, wave_end))
        wave_start = wave_end


def stellar_distribution(seed=None, n_star=None):
    """Function that attempts to return a realistic distribution of temperatures and luminosity scales."""
    star_prob = [0., 76.45, 12.1, 7.6, 3, 0.6, 0.13, 3E-5]
    star_prob = np.cumsum(star_prob)
    luminosity_scale = [(0.01, 0.08), (0.08, 0.6), (0.6, 1.5), (1.5, 5.0), (5.0, 25.0), (25.0, 30000.0),
                        (30000.0, 50000.0)]  # hotter stars are brighter on average.
    temperature_range = [(2400, 3700), (3700, 5200), (5200, 6000), (6000, 7500), (7500, 10000),
                         (10000, 30000), (30000, 50000)]
    star_type = {'M': 0, 'K': 1, 'G': 2, 'F': 3, 'A': 4, 'B': 5, 'O': 6}
    brightest_star = 'A'
    max_prob = np.max(star_prob[0:star_type[brightest_star] + 1])
    rand_gen = np.random
    if seed is not None:
        rand_gen.seed(seed)
    star_sort = rand_gen.uniform(0, max_prob, n_star)
    temperature = []
    luminosity = []
    for _i in range(star_type[brightest_star]):
        inds = np.where((star_sort < star_prob[_i + 1]) * (star_sort > star_prob[_i]))
        inds = inds[0]
        # if not inds.any():
        #    continue
        for ind in inds:
            temp_use = rand_gen.uniform(temperature_range[_i][0], temperature_range[_i][1])
            lum_use = rand_gen.uniform(luminosity_scale[_i][0], luminosity_scale[_i][1])
            temperature.append(temp_use)
            luminosity.append(lum_use)
    return((temperature, luminosity))
