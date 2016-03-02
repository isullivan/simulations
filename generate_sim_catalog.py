"""Function to generate simulated catalogs with reproduceable source spectra to feed into fast_dft."""
# import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
# import lsst.afw.image as afwImage
import numpy as np
from scipy import constants
bbox_init = afwGeom.Box2I(afwGeom.PointI(0, 0), afwGeom.ExtentI(512, 512))


def cat_sim(bbox=bbox_init, seed=None, n_star=None, n_galaxy=None, pixel_scale=None):

    return(catalog)


def cat_image(catalog=None, bbox=bbox_init, **kwargs):
    pass


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

    def radiance_calc(temperature, wavelength_start, wavelength_end, nterms=3):
        nu1 = c / (wavelength_start / 1E9)
        nu2 = c / (wavelength_end / 1E9)
        x1 = h * nu1 / (kb * temperature)
        x2 = h * nu2 / (kb * temperature)
        radiance1 = radiance_expansion(x1, nterms)
        radiance2 = radiance_expansion(x2, nterms)
        radiance_integral1 = prefactor * np.sum(poly_term for poly_term in radiance1)
        radiance_integral2 = prefactor * np.sum(poly_term for poly_term in radiance2)
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
