import numpy as np
import scipy.interpolate as si
from ppxf import ppxf_util as P
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker


def rebin_spectrum(lamdas, flux, errors, pixel_weights, instrumental_resolution=None, skyspecs=None, c=299792.458):
    """
    Take a set of spectra and rebin to have uniform spacing in log lambda, rather than lambda. Note that the spectra can't have any gaps in them- the array **must** be a continous array of values.

    Arguments:
        lamdas (array-like): Array of wavelength values for each element of the spectra. **Must** be continuous, with no gaps!
        flux (array-like): Array of flux values at each wavelength.
        errors (array-like):. Array of standard error values at each wavelength.
        pixel_weights (array-like): Array of weight values at each wavelength.

    Keyword Arguments:
        instrumental_resolution (array_like, optional): Defaults to None. Array of instrumental resolution values at each wavelength. If given,
            convolve the model spectra with a gaussian of this (variable) resolution during the fit.
        skyspecs (array_like, optional):  Defaults to None. Array of sky spectra to subtract during the fit.
        c (float, optional): Defaults to 299792.458. Speed of light **in km/s**.

    Returns:
        (tuple): a tuple containing:
            * galaxy (array_like): The log-rebinned array of flux values
            * noise (array_like): The log-rebinned array of standard error values
            * all_sky (array_like or None): The log-rebinned (two-dimensional) array of sky spectra. If skyspecs is None, this is None
            * log_inst_res (array_like or None): The log-rebinned instrumental resolution. If instrumental_resolution is None, this is None
            * weights (array_like): The log-rebinned array of weights
            * velscale (float): Velocity difference between adjacent pixels
            * goodpixels (array_like): The log-rebinned array of standard error values. TODO: Delete this
            * lam_range_gal (list): A two component vector with the start and stop wavelength values of the spectra.
            * logLam (array_like): The log-rebinned array of wavelength values
    """

    lower = lamdas.min()
    upper = lamdas.max()

    assert (lower >= lamdas.min()) & (upper <= lamdas.max()), 'Lower and upper limits must be within the wavelength ranges of the data'

    lam_range_gal = np.round(np.array([lower, upper]))
    mask = (lamdas >= lower) & (lamdas <= upper)

    flux = flux[mask]
    errors = errors[mask]
    pixel_weights = pixel_weights[mask]

    if skyspecs is not None:
        skyspecs = skyspecs[:, mask[0]]

    if instrumental_resolution is not None:
        instrumental_resolution = instrumental_resolution[mask]

    # Log rebin them
    galaxy, logLam, velscale = P.log_rebin(lam_range_gal, flux)
    noise, _, _ = P.log_rebin(lam_range_gal, errors, velscale=velscale)
    weights, _, _ = P.log_rebin(lam_range_gal, pixel_weights, velscale=velscale)

    all_sky = None
    if skyspecs is not None:
        O2_sky, _, _ = P.log_rebin(lam_range_gal, skyspecs[0, :], velscale=velscale)
        sky, _, _ = P.log_rebin(lam_range_gal, skyspecs[1, :], velscale=velscale)
        OH_sky, _, _ = P.log_rebin(lam_range_gal, skyspecs[2, :], velscale=velscale)
        NaD_sky, _, _ = P.log_rebin(lam_range_gal, skyspecs[3, :], velscale=velscale)

        all_sky = np.array([O2_sky, sky, OH_sky, NaD_sky])

    log_inst_res = None
    if instrumental_resolution is not None:
        log_inst_res, _, _ = P.log_rebin(lam_range_gal, instrumental_resolution, velscale=velscale)

    goodpixels = np.arange(len(galaxy))

    return galaxy, noise, all_sky, log_inst_res, weights, velscale, goodpixels, lam_range_gal, logLam
################################################################################################################################################################


def get_start_vals_and_bounds(theta):

    """
    Given a set of parameter values in lmfit format, get their values and bounds as numpy arrays.
    This is taken from the lmfit code itself.

    Arguments:
        theta: an lmfit Parameters object

    Returns:
        (tuple): a tuple containing:
            * var_arr (array_like): A numpy array of each of the parameter values
            * bounds: (array_like): A 2xN numpy array of (lower_bound, upper_bouns) for each fitting parameter
    """

    variables = [thing for thing in theta if theta[thing].vary]
    ndim = len(variables)
    var_arr = np.zeros(ndim)
    i = 0
    bounds = []
    for par in theta:
        param = theta[par]
        if param.expr is not None:
            param.vary = False
        if param.vary:
            var_arr[i] = param.value
            i += 1
        else:
            # don't want to append bounds if they're not being varied.
            continue

        param.from_internal = lambda val: val
        lb, ub = param.min, param.max
        if lb is None or lb is np.nan:
            lb = -np.inf
        if ub is None or ub is np.nan:
            ub = np.inf
        bounds.append((lb, ub))
    bounds = np.array(bounds)

    return var_arr, bounds


def get_starting_positions_for_walkers(start_values, stds, nwalkers, bounds):

    """
    Given a set of N starting values and sigmas for each value, make a set of random starting positions for M walkers. This is accomplished by
    drawing from an N dimensional gaussian M times.

    Arguments:
        start_values (array): A 1D array of starting values for each parameter.
        stds: (array): A 1D array of standard deviations, corresponding to the spread around the start value for each parameter
        nwakers: (int): The number of walkers we'll use.
        bounds (array):
    Returns:
        (array): An (N_walkers x N_parameters) array of starting positions.
    """

    ball = np.random.randn(start_values.shape[0], nwalkers) * stds[:, None] + start_values[:, None]

    ball = check_walker_starting_positions(ball, bounds)

    return ball.T


def check_walker_starting_positions(p0, bounds):

    """
    Sometimes our random starting positions for the walkers scatter outside our prior ranges. This function checks which walkers are oustide
    the bounds limits and replaces them with the upper limut of the prior- so, for example, an age of 14.1 Gys becomes 14 Gyrs.

    Arguments:
        p0 (array): A 2D array of shape (N_parameters, N_walkers) of starting positions
        bounds (array): A 2D array of shape (N_parameters, 2). Each row of the array is (lower, higher) bounds for the repsective parameter.

    Returns:
        (array): An (N_parameters x N_walkers) array of starting positions.
    """

    walkers_too_low = np.where(p0 < bounds[:, 0, None])
    walkers_too_high = np.where(p0 > bounds[:, 1, None])

    for parameter, walker_number in zip(*walkers_too_low):
        p0[parameter, walker_number] = bounds[parameter, 0] + 10 * np.finfo(float).eps

    for parameter, walker_number in zip(*walkers_too_high):
        p0[parameter, walker_number] = bounds[parameter, 1] - 10 * np.finfo(float).eps

    assert np.all((p0 > bounds[:, 0, None]) & (p0 < bounds[:, 1, None])), 'Some walkers are starting in bad places of parameter space!'

    return p0

def make_mask(lamdas, wavelengths):

    """
    Make a boolean mask which is False between each pair of wavelengths and True outside them.
    This is useful for masking skylines in our spectra

    Arguments:
        lamdas (array): An array of wavelength values
        wavelengths (list): A 2 component vector of low lambda and high lambda values we want to mask between
    Returns:
        (boolean array): A boolean of array of True outside the pair of wavelengths and False between them.
    """

    mask = np.ones_like(lamdas, dtype=bool)
    for pair in wavelengths:
        low, high = pair
        mask = mask & (lamdas < low) | (lamdas > high)

    return mask


def contiguous_zeros(a):
    """
    Find the beginning and end of each run of 0s in an array. This is useful for finding the chunks of the
    weights which we've masked as zeros.
    This is taken from https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array

    For example:
        a = [1, 2, 3, 0, 0, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 9, 8, 7, 0, 10, 11]

    would return:
        array([[ 3,  9], [12, 16], [19, 20]])
    Arguments:
        a (Boolean array): A Boolean array
    Returns:
        (array): A two dimensional array consisting of (start, stop) indices of each run of zeros

    """
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def plot_fit(theta, parameters, fig=None, axs=None, color='b'):

    """
    Given a set of fit values, plot the data, the model and the errors nicely.

    Arguments:
        theta (lmfit Parameters object): The values of the fit
        parameters (dict): A dictionary containing the flux, noise, etc. TODO: Clean this up!
        fig (matplotlib figure): Default is None. A figure object. If None, create a new one.
        axs (matplotlib axis): Default is None. An axis object. If None, create a new one.
        color (string): A matplotlib color string, for the colour of the model line in the plot.

    Returns:
        (tuple): A tuple containing:
            * likelihood (float): the likelihood for the model
            * fig, axs (tuple): A tuple of the matplotlib figure and axes objects
    """

    likelihood, [lams, temps, errors, specs, skies, emission_lines, polys, w] = lnlike(theta, parameters, ret_specs=True)

    # galaxy, noise, all_sky, weights, velscale, goodpixels, vsyst, interp_funct, correction_interps, logLams, logLam_gal, fit_wavelengths=parameters
    # Unpack the parameters we need
    fit_wavelengths = parameters['fit_wavelengths']
    c_light = parameters['c_light']
    vel = theta['Vel'].value

    if fig is None or axs is None:

        N_cols = np.ceil(len(fit_wavelengths) / 2).astype(int)

        fig = plt.figure(figsize=(18, 8.5))
        axs = np.empty((2, len(fit_wavelengths)), dtype='object')
        outer_grid = gridspec.GridSpec(2, N_cols)

        for i in range(len(fit_wavelengths)):

            inner_grid = gridspec.GridSpecFromSubplotSpec(2, 2, width_ratios=[1, 1], height_ratios=[2, 1], subplot_spec=outer_grid[i // N_cols, i % N_cols], hspace=0.05)
            axs[0, i] = fig.add_subplot(inner_grid[0, :2])
            axs[1, i] = fig.add_subplot(inner_grid[1, :2], sharex=axs[0, i])
            plt.setp(axs[0, i].get_xticklabels(), visible=False)

    fit_ranges = fit_wavelengths * (np.exp(vel / c_light))

    for i, fit_range in enumerate(fit_ranges):

        gmask = np.where((lams > fit_range[0]) & (lams < fit_range[1]))

        g = specs[gmask] / np.nanmedian(specs[gmask])
        t = temps[gmask] / np.nanmedian(specs[gmask])
        n = errors[gmask] / np.nanmedian(specs[gmask])
        sky = skies[gmask] / np.nanmedian(specs[gmask])
        # p = polys[gmask] / np.nanmedian(specs[gmask])
        em_lines = emission_lines[gmask] / np.nanmedian(specs[gmask])

        x = lams[gmask] / (np.exp(vel / c_light))

        axs[0, i].plot(x, g, c='k', linewidth=1.5)
        ylimits = axs[0, i].get_ylim()
        axs[0, i].plot(x, g + sky, c='k', linewidth=1.5, alpha=0.3)
        axs[0, i].plot(x, t, c=color, linewidth=2.0, zorder=10)
        axs[0, i].plot(x, t + em_lines, c='r', linewidth=2.0, zorder=9)
        axs[0, i].fill_between(x, g - n, g + n, facecolor='k', alpha=0.3)

        axs[0, i].set_ylim(ylimits)
        axs[1, i].plot(x, 100 * (g - t - em_lines) / (t), c='k', linewidth=1.5)
        axs[1, i].fill_between(x, 100 * n / g, -100.0 * n / (g), facecolor='k', alpha=0.5)
        axs[1, i].axhline(0.0, linestyle='dashed', c='k')

        axs[0, i].set_xlim([x.min(), x.max()])
        axs[1, i].set_ylim([-5, 5])

        axs[1, i].set_xlabel(r'Rest Wavelength (\AA)', fontsize=14)
        axs[0, i].set_ylabel(r'Flux (Arbitrary Units)', fontsize=12)
        axs[1, i].set_ylabel(r'Residuals (\%)', fontsize=12)

        # Avoid the overlapping labels
        axs[0, i].yaxis.set_major_locator(ticker.MaxNLocator(prune='lower'))
        axs[0, i].yaxis.set_major_locator(ticker.MaxNLocator(prune='lower'))
        axs[1, i].yaxis.set_major_locator(ticker.MultipleLocator(2))

        # Plot gray shaded regions on the areas that we mask from the chi-squared
        zero_weights = contiguous_zeros(w)

        for pair in zero_weights:

            low = pair[0]
            high = pair[1]
            axs[0, i].axvspan(lams[low] / (np.exp(vel / c_light)), lams[high] / (np.exp(vel / c_light)), alpha=0.5, color='grey')
            axs[1, i].axvspan(lams[low] / (np.exp(vel / c_light)), lams[high] / (np.exp(vel / c_light)), alpha=0.5, color='grey')

        # axs[i, 0].axvspan(Ha_lam[0][0], Ha_lam[0][1], alpha=0.5, color='grey')
        # axs[i, 1].axvspan(Ha_lam[0][0], Ha_lam[0][1], alpha=0.5, color='grey')

    fig.tight_layout()

    return likelihood, (fig, axs)

def get_best_fit_template(theta, parameters, convolve=True):

    """
    Given a set of values for each of the parameters in the fit, make a model which we can compare to the data

    Arguments:
        theta (lmfit Parameters): An lmfit Parameters object
        parameters (dict): A dictionary containing... TODO: Rename this and tidy up!
        convolve (bool): Default is True. Convolve the model with the Line of sight velocity distribution? Or leave unconvolved
    Returns:
        (tuple): A tuple containing:
            * loglams (array): log-rebinned wavelength array
            * template (array): the model array to be compared to the data
            * base_template (array): the model array but *without* including variation in any chemical abundances. This model only includes
                variation in stellar age, metallicity and the two IMF parameters.
    """

    # Unpack the parameters
    galaxy = parameters['log_galaxy']
    velscale = parameters['velscale']
    vsyst = parameters['dv']
    interp_funct = parameters['linear_interp']
    correction_interps = parameters['correction_interps']
    logLams = parameters['log_lam_template']
    logLam_gal = parameters['log_lam']

    general_interp, na_interp, carbon_interp, positive_only_interp, T_interp = correction_interps

    # Values of theta
    vel = theta['Vel'].value
    sigma = theta['sigma'].value
    Na_abundance = theta['Na'].value
    Carbon_abundance = theta['C'].value,

    general_abundances = np.array([theta['Ca'].value, theta['Fe'].value, theta['N'].value, theta['Ti'].value, theta['Mg'].value, theta['Si'].value, theta['Ba'].value])  

    positive_abundances = np.array([theta['as_Fe'].value, theta['Cr'].value,theta['Mn'].value,theta['Ni'].value,theta['Co'].value, theta['Eu'].value,theta['Sr'].value, theta['K'].value,theta['V'].value,theta['Cu'].value])

    positive_abundance_indices = np.where(positive_abundances > 0.0)[0]
    general_abundance_indices = np.where(general_abundances != 0.0)[0]

    age = theta['age'].value
    Z = theta['Z'].value
    imf_x1, imf_x2 = theta['imf_x1'].value, theta['imf_x2'].value
    # O2_scale=theta['O2_Scale'].value
    # sky_scale=theta['sky_Scale'].value
    # OH_scale=theta['OH_Scale'].value
    # NaD_sky_scale=theta['NaD_sky_scale'].value

    # Make the base template (age, Z, IMF)
    base_template = _make_model(age, Z, imf_x1, imf_x2, interp_funct, logLams)

    # Make the correction for elements which vary >0.0

    if positive_abundance_indices.size > 0:
        positive_only_correction = _get_correction(positive_only_interp, logLams, positive_abundance_indices, positive_abundances, age, Z)
    else:
        positive_only_correction = np.zeros_like(base_template)

    # Response function for general element corrections
    if general_abundance_indices.size > 0:
        general_correction = _get_correction(general_interp, logLams, general_abundance_indices, general_abundances, age, Z)
    else:
        general_correction = np.zeros_like(base_template)

    # Have to treat Na differently
    if Na_abundance != 0.0:
        na_correction = na_interp((Na_abundance, age, Z, logLams))
    else:
        na_correction = np.zeros_like(base_template)

    # And Carbon differently
    if Carbon_abundance != 0.0:
        carbon_correction = carbon_interp((Carbon_abundance, age, Z, logLams))
    else:
        carbon_correction = np.zeros_like(base_template)

    # Add things together- see Appendix of Vaughan+ 2018
    template = np.exp(np.log(base_template) + general_correction + positive_only_correction + na_correction + carbon_correction)

    if convolve:
        template = P.convolve_gauss_hermite(template, velscale=velscale, start=[vel, sigma], npix=len(galaxy), vsyst=vsyst).squeeze()
        logLams = logLam_gal.copy()

    return logLams, template, base_template


def _make_model(age, Z, imf_x1, imf_x2, interp_funct, logLams):

    """
    Make the 'base' model from the model interpolators
    Arguments:
        age (float): Stellar age
        Z (float): Metallicity
        imf_x1 (float): Low-mass IMF slope between 0.08M and 0.5M
        imf_x2 (float): Low-mass IMF slope between 0.5M and 1.0M
        interp_funct (scipy RegularGridInterpolator): Model interpolator
        logLams (array): Log-rebinned wavelength array

    Returns:
        (array): A model at the requried fit parameters
    """

    model = interp_funct((logLams, age, Z, imf_x1, imf_x2))

    return model

def _get_correction(interpolator, logLams, elems, abunds, age, Z):

    """
    Get the response function we apply to our base model

    Arguments:
        interpolator (scipy RegularGridInterpolator): An interpolator for the required correction
        logLams (array): Log-rebinned wavelength array
        elems (array): A vector of integers corresponding to the elements we want to vary
        abunds (array): A vector of abundances for each element we're varying
        age (float): Stellar age
        Z (float): Metallicity

    """

    # The interpolator expects a list of 6 numbers. Meshgrid the two arrays which are of different lengths
    # (the indices and the number of elements to enhance) and then create lists of ages, Zs and IMFs of the correct
    # shapes. Then do the same for the abundances. Stack together and pass to the interpolator object!

    points = np.meshgrid(elems, logLams, indexing='ij')
    flat = np.array([m.flatten() for m in points])
    # flat is now an array of points of shape 2, len(indices)*len(elems)
    # make arrays of the other variables of length len(indices)*len(elems)
    ages = np.ones_like(points[0]) * age
    Zs = np.ones_like(points[0]) * Z

    # Get the correct abundance for each element- luckily we can index the abundance array by the integer element array
    abunds = abunds[points[0]]

    # Stack together
    xi = np.vstack((flat[0, :], abunds.ravel(), ages.ravel(), Zs.ravel(), flat[1, :]))
    # Do the interpolation
    out_array = interpolator(xi.T)
    # reshape everything to be (len(indices), len(elements))
    result = out_array.reshape(*points[0].shape)

    return np.sum(result, axis=0)


def get_linear_best_fit_template(theta, parameters, fit_class, convole=True):

    """
    Get the best-fit template on a *linear* wavelength scale by interpolating the log-lambda template.
    This is required when calculating the luminosity of a template through a filter.

    Arguments:
        theta (lmfit Parameters): An lmfit Parameters object
        parameters (dict): A dictionary containing... TODO: Rename this and tidy up!
        fit_class: TODO

    Returns:
        (tuple): tuple containing:
            * lin_template (array): the model array to be compared to the data, on a linear wavelength scale
            * lin_base_template (array): the model array but *without* including variation in any chemical abundances, on a linear wavelength scale. This model only includes
                variation in stellar age, metallicity and the two IMF parameters.
    """

    logLams, template, base_template = get_best_fit_template(theta, parameters, convolve=convole)

    interp1 = si.interp1d(np.exp(logLams), template, fill_value='extrapolate')
    lin_template = interp1(fit_class.lin_lam)

    interp2 = si.interp1d(np.exp(fit_class.logLam_template), base_template, fill_value='extrapolate')
    lin_base_template = interp2(fit_class.lin_lam)

    return lin_template, lin_base_template


def lnlike(theta, parameters, plot=False, ret_specs=False):

    """
    The log-Likelihood function of the fitting. TODO- expand this.

    Arguments:
        theta (lmfit Parameters): An lmfit Parameters object
        parameters (dict): A dictionary containing... TODO: Rename this and tidy up!
        plot (Boolean): Deprecated. TODO remove
        ret_specs (boolean): Default is False. If True, return a series of spectra made during the fitting process

    Returns:
        (float): The log-likelihood of the fit parameters. If ret_specs is True, also return:
            * likelihood
            * A tuple containing (lam, t, e, s, skies, g_lines, p, w). Add more info here.

    """

    # Unpack the parameters
    galaxy = parameters['log_galaxy']
    noise = parameters['log_noise']
    all_sky = parameters['log_skyspecs']
    weights = parameters['log_weights']
    emission_lines = parameters['emission_lines']
    velscale = parameters['velscale']
    vsyst = parameters['dv']
    logLam_gal = parameters['log_lam']
    fit_wavelengths = parameters['fit_wavelengths']
    c_light = parameters['c_light']

    if all_sky is not None:
        O2_sky, base_sky, OH_sky, NaD_sky = all_sky

    # Values of theta
    vel = theta['Vel'].value

    # Scale the noise by some value f
    ln_f = theta['ln_f'].value

    if all_sky is not None:
        O2_scale = theta['O2_Scale'].value
        sky_scale = theta['sky_Scale'].value
        OH_scale = theta['OH_Scale'].value
        NaD_sky_scale = theta['NaD_sky_scale'].value

    if emission_lines is not None:
        vel_gas = theta['Vel_em'].value
        sig_gas = theta['sig_em'].value
        Ha_flux = np.exp(theta['Ha'].value)
        Hbeta_flux = np.exp(theta['Hb'].value)

        SII_6716 = np.exp(theta['SII_6716'].value)
        SII_6731 = np.exp(theta['SII_6731'].value)
        OIII = np.exp(theta['OIII'].value)
        OI = np.exp(theta['OI'].value)
        NII = np.exp(theta['NII'].value)

    # SINGLE POWER LAW IMF
    # theta['imf_x1'].set(theta['imf_x2'].value)

    _, temp, base_template = get_best_fit_template(theta, parameters, convolve=True)

    # Ranges we fit over- these have to change with redshift
    fit_ranges = fit_wavelengths * (np.exp(vel / c_light))

    chisqs = np.zeros_like(galaxy)

    # Make the emission lines:
    if emission_lines is not None:
        unconvolved_em_lines = Hbeta_flux * emission_lines[:, 0] + Ha_flux * emission_lines[:, 1] + SII_6716 * emission_lines[:, 2] + SII_6731 * emission_lines[:, 3] + OIII * emission_lines[:, 4] + OI * emission_lines[:, 5] + NII * emission_lines[:, 6]
        convolved_em_lines = P.convolve_gauss_hermite(unconvolved_em_lines, velscale=velscale, start=[vel_gas, sig_gas], npix=len(galaxy), vsyst=vsyst).squeeze()
    else:
        convolved_em_lines = np.zeros_like(galaxy)

    lams = []
    specs = []
    temps = []
    gas_lines = []
    residuals = []
    errors = []
    skies = []
    polys = []

    # # ToDo- move this to the templates
    # #Convolve with the instrumental resolution
    # import ipdb; ipdb.set_trace()
    # if instrumental_resolution is not None:
    #     temp=P.gaussian_filter1d(temp, instrumental_resolution/velscale)
    #     #Bit hacky
    #     #Replace the first and last few values of temp (which get set to zero by the convolution)
    #     #to the median value of the whole thing
    #     temp[temp==0]=np.median(temp)

    # Do the fitting
    for i, fit_range in enumerate(fit_ranges):

        # Mask around each fit range

        gmask = np.where((np.exp(logLam_gal) > fit_range[0]) & (np.exp(logLam_gal) < fit_range[1]))

        g = galaxy[gmask]
        n = noise[gmask]
        t = temp[gmask]
        gas = convolved_em_lines[gmask]
        ws = weights[gmask].astype(bool)

        if all_sky is not None:

            # Mask the sky spectra
            O2_s = O2_sky[gmask]
            s = base_sky[gmask]
            OH_s = OH_sky[gmask]
            NaD_s = NaD_sky[gmask]

            # Make our sky model
            sky = O2_scale * O2_s + sky_scale * s + OH_scale * OH_s + NaD_sky_scale * NaD_s
            # sky/=galmedian
        else:
            sky = np.zeros_like(g)

        # Order of the polynomial
        morder = int((fit_range[1] - fit_range[0]) / 100)

        # Fit the polynomials, weighting by the noise and ignoring pixels with 0 weight
        poly_weights = 1.0 / n**2
        poly_weights[~ws] = 0.0
        poly_weights[~np.isfinite(poly_weights)] = 0.0
        poly = _fit_legendre_polys((g - sky - gas) / (t), morder, weights=poly_weights)

        # Scale the noise by some fraction ln_f
        n_corrected = np.sqrt((1 + np.exp(2 * ln_f)) * n**2)

        # Calculate the chi_squared
        chisqs[gmask] = (((g - sky - gas - t * poly) / n_corrected)**2)

        lams.append(np.exp(logLam_gal[gmask]))
        temps.append(poly * t)
        gas_lines.append(gas)
        residuals.append(g - sky - poly * gas - poly * t)
        errors.append(n_corrected)
        specs.append(g - sky)
        skies.append(sky)
        polys.append(poly)

    # We may have a gap in the ranges we want to fit over
    # This ensures that only pixels in the fit ranges contribute to the chisquared
    all_fit_ranges_mask = np.zeros_like(galaxy, dtype=bool)

    for fit_range in fit_ranges:
        this_fit_range_mask = (np.exp(logLam_gal) > fit_range[0]) & (np.exp(logLam_gal) < fit_range[1])
        all_fit_ranges_mask[this_fit_range_mask] = 1

    # Sum the chisquareds, masking out the pixels we don't want
    chisq = np.sum((chisqs * weights)[all_fit_ranges_mask])

    # Count all of the things we're varying
    n_variables = len([thing for thing in theta if theta[thing].vary])
    n_masked_pixels = len(np.where(weights[all_fit_ranges_mask] == 0)[0])
    n_dof = len(chisqs[all_fit_ranges_mask]) - n_masked_pixels - n_variables

    ###########################################################################

    lam = np.concatenate(lams)
    t = np.concatenate(temps)
    e = np.concatenate(errors)
    s = np.concatenate(specs)
    skies = np.concatenate(skies)
    g_lines = np.concatenate(gas_lines)
    p = np.concatenate(polys)
    w = weights[all_fit_ranges_mask].astype(bool)

    # Log likelihood- chisqaured plus sum of errors, which now depend on ln_f
    likelihood = -0.5 * (chisq) - 0.5 * np.sum(np.log(2 * np.pi * e**2) * w)

    if ret_specs:
        return likelihood, [lam, t, e, s, skies, g_lines, p, w]

    return likelihood

def _fit_legendre_polys(ratio, morder, weights=None):

    """
    Fit a legendre polynomial to the ratio of the data to the model, optionally weighting by the noise

    Arguments:
        ratio (array): Ratio of the data to the model spectrum.
        morder (int): Order of the polynomial
        weights (optional, array): Default is None. Weights for each value for the polynomial fitting
    Returns:
        (array): The polynomial value at each wavelength location
    """

    x_vals = np.linspace(-1, 1, len(ratio))
    coeffs = np.polynomial.legendre.legfit(x_vals, ratio, morder, w=weights)

    polynomial = np.polynomial.legendre.legval(x_vals, coeffs)

    return polynomial


def init(func, *args, **kwargs):

    global parameters, logLam
    parameters, logLam, ndim = func(*args, **kwargs)
