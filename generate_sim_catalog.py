"""Function to generate simulated catalogs with reproduceable source spectra to feed into fast_dft."""
import lsst.afw.table as afwTable
import lsst.afw.geom as afwGeom
from lsst.utils import getPackageDir
from lsst.sims.photUtils import Sed, Bandpass  # , PhotometricParameters
from lsst.sims.photUtils import matchStar
# import lsst.afw.image as afwImage
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
# import math
from scipy import constants
import galsim
from calc_refractive_index import diff_refraction
from fast_dft import fast_dft
bbox_init = afwGeom.Box2I(afwGeom.PointI(0, 0), afwGeom.ExtentI(512, 512))
photons_per_adu = 1e4  # used only to approximate the effect of photon shot noise, if photon_noise=True


def cat_image(catalog=None, bbox=bbox_init, name=None, psf=None, pixel_scale=None, pad_image=1.1,
              sky_noise=0.0, instrument_noise=0.0, photon_noise=False, wavelength_step=10,
              dcr_flag=False, band_name='g', sed_list=None, **kwargs):
    """Wrapper that takes a catalog of stars and simulates an image."""
    if catalog is None:
        catalog = cat_sim(bbox=bbox, name=name, **kwargs)
    schema = catalog.getSchema()
    n_star = len(catalog)
    bandpass = load_bandpass(band_name=band_name, **kwargs)
    x_size, y_size = bbox.getDimensions()
    x0, y0 = bbox.getBegin()
    if name is None:
        # If no name is supplied, find the first entry in the schema in the format *_flux
        schema_entry = schema.extract("*_flux", ordered='true')
        fluxName = schema_entry.iterkeys().next()
    else:
        fluxName = name + '_flux'

    if psf is None:
        psf = galsim.Kolmogorov(fwhm=3)
    if pixel_scale is None:
        # I think most PSF classes have a getFWHM method. The math converts to a sigma for a gaussian.
        fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2. * np.log(2)))
        pixel_scale = psf.getFWHM() * fwhm_to_sigma

    if sed_list is None:
        # Load in model SEDs
        matchStarObj = matchStar()
        sed_list = matchStarObj.loadKuruczSEDs()

    fluxKey = schema.find(fluxName).key
    temperatureKey = schema.find("temperature").key
    x0, y0 = bbox.getBegin()
    # if catalog.isContiguous()
    flux = catalog[fluxKey]
    temperatures = catalog[temperatureKey]
    if dcr_flag:
        flux_arr = np.ndarray((n_star, bandpass_nstep(bandpass, wavelength_step=wavelength_step)))
        for _i in range(n_star):
            f_star = flux[_i]
            t_star = temperatures[_i]
            star_spectrum = star_gen(sed_list=sed_list, temperature=t_star, flux=f_star,
                                     bandpass=bandpass, wavelength_step=wavelength_step)
            flux_arr[_i, :] = np.array([f for f in star_spectrum])
    else:
        flux_arr = flux
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

    source_image = fast_dft(flux_arr, xv, yv, x_size=x_size_use, y_size=y_size_use, no_fft=True)

    if dcr_flag:
        convol = np.zeros((y_size_use, x_size_use), dtype='complex64')
        dcr_gen = dcr_generator(bandpass, pixel_scale=pixel_scale, wavelength_step=wavelength_step, **kwargs)
        for _i, offset in enumerate(dcr_gen):
            psf_image = psf.drawImage(scale=pixel_scale, method='no_pixel', offset=offset,
                                      nx=x_size_use, ny=y_size_use, use_true_center=False)
            source_image_use = source_image[_i]
            if sky_noise > 0:
                source_image_use += (np.random.normal(scale=sky_noise, size=(y_size_use, x_size_use))
                                     / np.sqrt(bandpass_nstep(bandpass, wavelength_step=wavelength_step)))
            convol += fft2(source_image_use) * fft2(psf_image.array)
    else:
        psf_image = psf.drawImage(scale=pixel_scale, method='no_pixel', offset=[0, 0],
                                  nx=x_size_use, ny=y_size_use, use_true_center=False)
        if photon_noise:
            base_noise = np.random.normal(scale=1.0, size=(y_size_use, x_size_use))
            base_noise *= np.sqrt(np.abs(source_image) / photons_per_adu)
            source_image += base_noise
        if sky_noise > 0:
            source_image += np.random.normal(scale=sky_noise, size=(y_size_use, x_size_use))
        convol = fft2(source_image) * fft2(psf_image.array)

    # fft_filter = outer(hanning(y_size_use), hanning(x_size_use))
    # convol *= fftshift(fft_filter)
    return_image = np.real(fftshift(ifft2(convol)))
    if instrument_noise > 0:
        return_image += np.random.normal(scale=instrument_noise, size=(y_size_use, x_size_use))
    return(return_image[y0:y1, x0:x1])


def bandpass_nstep(bandpass, wavelength_step=None):
    """Simple function to pre-compute the number of bins to use for a given bandpass."""
    if wavelength_step is None:
        wavelength_step = bandpass.wavelen_step
    return(int(np.ceil((bandpass.wavelen_max - bandpass.wavelen_min) / wavelength_step)))


def dcr_generator(bandpass, pixel_scale=None, elevation=None, azimuth=None,
                  wavelength_step=None, **kwargs):
    """Call the functions that compute Differential Chromatic Refraction."""
    if elevation is None:
        elevation = 50.0
    if azimuth is None:
        azimuth = 0.0
    zenith_angle = 90.0 - elevation
    wavelength_midpoint = bandpass.calc_eff_wavelen()
    for wavelength in wavelength_iterator(bandpass, wavelength_step=wavelength_step, use_midpoint=True):
        # Note that refract_amp can be negative, since it's relative to the midpoint of the band
        refract_amp = diff_refraction(wavelength=wavelength, wavelength_ref=wavelength_midpoint,
                                      zenith_angle=zenith_angle, **kwargs)
        refract_amp *= 3600.0 / pixel_scale  # Refraction initially in degrees, convert to pixels.
        dx = refract_amp * np.sin(np.radians(azimuth))
        dy = refract_amp * np.cos(np.radians(azimuth))
        yield((dx, dy))


def cat_sim(bbox=None, seed=None, n_star=None, n_galaxy=None, name=None, **kwargs):
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
    schema.addField("spectral_id", type="D")
    schema.addField("metallicity", type="D")
    schema.addField("gravity", type="D")
    schema.addField("dust", type="D")
    schema.getAliasMap().set('slot_Centroid', name + '_Centroid')

    x_size, y_size = bbox.getDimensions()
    x0, y0 = bbox.getBegin()
    temperature, luminosity = stellar_distribution(seed=seed, n_star=n_star, **kwargs)
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


def star_gen(sed_list=None, seed=None, temperature=5600, metallicity=0.0, surface_gravity=1.0,
             flux=1.0, bandpass=None, wavelength_step=None):
    """Generate a randomized spectrum at a given temperature over a range of wavelengths."""
    """
        Either use a supplied list of SEDs to be drawn from, or use a blackbody radiation model.
        The output is normalized to sum to the given flux.
        [future] If a seed is supplied, noise can be added to the final spectrum before normalization.
    """
    if sed_list is not None:
        temperature_list = [star.temp for star in sed_list]
        grav_list = [star.logg for star in sed_list]
        metal_list = [star.logZ for star in sed_list]
        temp_weight = (temperature_list - temperature) / temperature
        t_inds = np.where(temp_weight <= np.min(temp_weight))

    else:
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

        normalization = flux / radiance_calc(temperature, bandpass.wavelen_min, bandpass.wavelen_max)
        for wave_start, wave_end in wavelength_iterator(bandpass, wavelength_step=wavelength_step):
            yield(normalization * radiance_calc(temperature, wave_start, wave_end))


def load_bandpass(band_name='g', use_mirror=True, use_lens=True, use_atmos=True,
                  use_filter=True, use_detector=True, **kwargs):
    """Load in Bandpass object."""
    class BandpassMod(Bandpass):
        """Customize a few methods of the Bandpass class from sims_photUtils."""

        def calc_eff_wavelen(self, wavelength_min=None, wavelength_max=None):
            """Calculate effective wavelengths for filters."""
            # This is useful for summary numbers for filters.
            # Calculate effective wavelength of filters.
            if self.phi is None:
                self.sbTophi()
            if wavelength_min is None:
                wavelength_min = np.min(self.wavelen)
            if wavelength_max is None:
                wavelength_max = np.max(self.wavelen)
            w_inds = (self.wavelen >= wavelength_min) & (self.wavelen <= wavelength_max)
            effwavelenphi = (self.wavelen[w_inds] * self.phi[w_inds]).sum() / self.phi[w_inds].sum()
            return effwavelenphi

    """Define the wavelength range and resolution for a given ugrizy band."""
    band_dict = {'u': (324.0, 395.0), 'g': (405.0, 552.0), 'r': (552.0, 691.0),
                 'i': (818.0, 921.0), 'z': (922.0, 997.0), 'y': (975.0, 1075.0)}
    band_range = band_dict[band_name]
    bandpass = BandpassMod(wavelen_min=band_range[0], wavelen_max=band_range[1])
    throughput_dir = getPackageDir('throughputs')
    lens_list = ['baseline/lens1.dat', 'baseline/lens2.dat', 'baseline/lens3.dat']
    mirror_list = ['baseline/m1.dat', 'baseline/m2.dat', 'baseline/m3.dat']
    atmos_list = ['atmos/atmos_11.dat']
    detector_list = ['baseline/detector.dat']
    filter_list = ['baseline/filter_' + band_name + '.dat']
    component_list = []
    if use_mirror:
        component_list += mirror_list
    if use_lens:
        component_list += lens_list
    if use_atmos:
        component_list += atmos_list
    if use_detector:
        component_list += detector_list
    if use_filter:
        component_list += filter_list
    bandpass.readThroughputList(rootDir=throughput_dir, componentList=component_list)
    return(bandpass)


def wavelength_iterator(bandpass, wavelength_step=None, use_midpoint=False):
    """Define iterator to ensure that loops over wavelength are consistent."""
    if wavelength_step is None:
        wavelength_step = bandpass.wavelen_step
    wave_start = bandpass.wavelen_min
    while wave_start < bandpass.wavelen_max:
        wave_end = wave_start + wavelength_step
        if wave_end > bandpass.wavelen_max:
            wave_end = bandpass.wavelen_max
        if use_midpoint:
            yield(bandpass.calc_eff_wavelen(wavelength_min=wave_start, wavelength_max=wave_end))
        else:
            yield((wave_start, wave_end))
        wave_start = wave_end


def stellar_distribution(seed=None, n_star=None, hottest_star='A', coolest_star='M', **kwargs):
    """Function that attempts to return a realistic distribution of temperatures and luminosity scales."""
    star_prob = [76.45, 12.1, 7.6, 3, 0.6, 0.13, 3E-5]
    luminosity_scale = [(0.01, 0.08), (0.08, 0.6), (0.6, 1.5), (1.5, 5.0), (5.0, 25.0), (25.0, 30000.0),
                        (30000.0, 50000.0)]  # hotter stars are brighter on average.
    temperature_range = [(2400, 3700), (3700, 5200), (5200, 6000), (6000, 7500), (7500, 10000),
                         (10000, 30000), (30000, 50000)]
    metallicity_range = [(-5.0, 1.0)] * len(star_prob)
    surface_gravity_range = [(0.0, 5.0)] * len(star_prob)
    star_type = {'M': 0, 'K': 1, 'G': 2, 'F': 3, 'A': 4, 'B': 5, 'O': 6}
    s_hot = star_type[hottest_star] + 1
    s_cool = star_type[coolest_star]
    n_star_type = s_hot - s_cool
    star_prob = star_prob[s_cool:s_hot]
    star_prob.insert(0, 0)
    luminosity_scale = luminosity_scale[s_cool:s_hot]
    temperature_range = temperature_range[s_cool:s_hot]
    metallicity_range = metallicity_range[s_cool:s_hot]
    surface_gravity_range = surface_gravity_range[s_cool:s_hot]
    star_prob = np.cumsum(star_prob)
    max_prob = np.max(star_prob)
    rand_gen = np.random
    if seed is not None:
        rand_gen.seed(seed)
    star_sort = rand_gen.uniform(0, max_prob, n_star)
    temperature = []
    luminosity = []
    metallicity = []
    surface_gravity = []
    info_string = 'Number of stars of each type: ' + coolest_star
    for _i in range(n_star_type):
        inds = np.where((star_sort < star_prob[_i + 1]) * (star_sort > star_prob[_i]))
        inds = inds[0]  # np.where returns a tuple of two arrays
        info_string += " " + str(len(inds))
        for ind in inds:
            temp_use = rand_gen.uniform(temperature_range[_i][0], temperature_range[_i][1])
            lum_use = rand_gen.uniform(luminosity_scale[_i][0], luminosity_scale[_i][1])
            metal_use = rand_gen.uniform(metallicity_range[_i][0], metallicity_range[_i][1])
            grav_use = rand_gen.uniform(surface_gravity_range[_i][0], surface_gravity_range[_i][1])
            temperature.append(temp_use)
            luminosity.append(lum_use)
            metallicity.append(metal_use)
            surface_gravity.append(grav_use)
    info_string += " " + hottest_star
    print(info_string)
    return((temperature, luminosity))

"""
def sim_sed_read():
    "Simple function that reads in the filenames in the sims_sed_library and extracts values."
    sed_dir = os.path.join(getPackageDir('sims_sed_library'), 'starSED', 'kurucz')
    list_of_star_sed_names = os.listdir(sed_dir)
    metallicity_list = []
    temperature_list = []
    gravity_list = []
    for name in list_of_star_sed_names:
        log_metallicity = float(name[2:4])
        sign_metallicity = name[1]
        if sign_metallicity == 'm':
            log_metallicity *= -1
        metallicity_list.append(log_metallicity)
        temperature = float(name[19:23])
        temperature_list.append(temperature)
        surface_gravity = float(name[16:18])
        gravity_list.append(surface_gravity)
    return(temperature_list, gravity_list, metallicity_list)
"""
