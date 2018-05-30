

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np 
import scipy.constants as const

from scipy import special
import scipy.interpolate as si


def rebin_MUSE_spectrum(lamdas, flux, errors, pixel_weights, instrumental_resolution=None, skyspecs=None, c=299792.458):
    """
    Take a set of spectra and rebin to have uniform spacing in log lambda, rather than lambda. Note that the spectra can't have any gaps in them- the array **must** be 
    a continous array of values.

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
 
    lower=lamdas.min()
    upper=lamdas.max()

    assert (lower>=lamdas.min()) & (upper<=lamdas.max()), 'Lower and upper limits must be within the wavelength ranges of the data'

    

    lam_range_gal=np.array([lower, upper])
    mask=np.where((lamdas>lower) & (lamdas<upper))

    flux=flux[mask]
    errors=errors[mask]
    pixel_weights=pixel_weights[mask]

    if skyspecs is not None:
        skyspecs=skyspecs[:, mask[0]]

    if instrumental_resolution is not None:
        instrumental_resolution=instrumental_resolution[mask]
    

    #Log rebin them
    galaxy, logLam, velscale = log_rebin(lam_range_gal, flux)
    noise, _, _=log_rebin(lam_range_gal, errors, velscale=velscale) 
    weights, _, _=log_rebin(lam_range_gal, pixel_weights, velscale=velscale)
    #import pdb; pdb.set_trace()

    all_sky=None
    if skyspecs is not None:
        O2_sky, _, _=log_rebin(lam_range_gal, skyspecs[0, :], velscale=velscale)
        sky, _, _=log_rebin(lam_range_gal, skyspecs[1, :], velscale=velscale) 
        OH_sky, _, _=log_rebin(lam_range_gal, skyspecs[2, :], velscale=velscale)
        NaD_sky, _, _=log_rebin(lam_range_gal, skyspecs[3, :], velscale=velscale)

        all_sky=np.array([O2_sky, sky, OH_sky, NaD_sky])

    log_inst_res=None
    if instrumental_resolution is not None:
        log_inst_res, _, _=log_rebin(lam_range_gal, instrumental_resolution, velscale=velscale)


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

    variables=[thing for thing in theta if theta[thing].vary]
    ndim=len(variables)
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

def get_starting_poitions_for_walkers(start_values, stds, nwalkers):

    """
    Given a set of N starting values and sigmas for each value, make a set of random starting positions for M walkers. This is accomplished by 
    drawing from an N dimensional gaussian M times.

    Arguments:
        start_values (array): A 1D array of starting values for each parameter.
        stds: (array): A 1D array of standard deviations, corresponding to the spread around the start value for each parameter
        nwakers: (int): The number of walkers we'll use.
    Returns:
        (array): An (N_parameters x N_walkers) array of starting positions.
    """

    ball=np.random.randn(start_values.shape[0], nwalkers)*stds[:, None] + start_values[:, None]

    return ball

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

    mask=np.ones_like(lamdas, dtype=bool)
    for pair in wavelengths:
        low, high=pair
        mask = mask & (lamdas<low) | (lamdas>high)

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
            * chisq (float): the value of Chi-squared for the model
            * chisq_per_dof (float): Chi-squared per degree of freedom
            * fig, axs (tuple): A tuple of the matplotlib figure and axes objects
    """

    chisq, chisq_per_dof, [lams, temps, errors, specs, skies, emission_lines, polys, w]=lnlike(theta, parameters, ret_specs=True)

    print('Chisq per DoF is {}'.format(chisq_per_dof))

    #galaxy, noise, all_sky, weights, velscale, goodpixels, vsyst, interp_funct, correction_interps, logLams, logLam_gal, fit_wavelengths=parameters
    #Unpack the parameters we need
    fit_wavelengths=parameters['fit_wavelengths']
    c_light=parameters['c_light']
    vel = theta['Vel'].value

    import matplotlib.pyplot as plt 
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as ticker   



    


    
    if fig is None or axs is None:

        N_cols=np.ceil(len(fit_wavelengths)/2).astype(int)


        fig=plt.figure(figsize=(18, 8.5))
        axs=np.empty((2, len(fit_wavelengths)), dtype='object')
        outer_grid=gridspec.GridSpec(2, N_cols)

        for i in range(len(fit_wavelengths)):
            
            inner_grid = gridspec.GridSpecFromSubplotSpec(2, 2, width_ratios=[1, 1], height_ratios=[2, 1], subplot_spec=outer_grid[i//N_cols, i%N_cols], hspace=0.05)
            axs[0, i] = fig.add_subplot(inner_grid[0, :2])
            axs[1, i] = fig.add_subplot(inner_grid[1, :2], sharex=axs[0, i])
            plt.setp(axs[0, i].get_xticklabels(), visible=False)


    fit_ranges=fit_wavelengths*(np.exp(vel/c_light))



    for i, fit_range in enumerate(fit_ranges):

        gmask=np.where((lams>fit_range[0]) & (lams<fit_range[1]))
        
        
        g=specs[gmask]/np.nanmedian(specs[gmask])
        t=temps[gmask]/np.nanmedian(specs[gmask])
        n=errors[gmask]/np.nanmedian(specs[gmask])
        sky=skies[gmask]/np.nanmedian(specs[gmask])
        p=polys[gmask]/np.nanmedian(specs[gmask])
        em_lines=emission_lines[gmask]/np.nanmedian(specs[gmask])

        x=lams[gmask]/(np.exp(vel/c_light))

        axs[0, i].plot(x, g, c='k', linewidth=1.5)
        ylimits=axs[0, i].get_ylim()
        axs[0, i].plot(x, g+sky, c='k', linewidth=1.5, alpha=0.3)
        axs[0, i].plot(x, t, c=color, linewidth=2.0, zorder=10)
        axs[0, i].plot(x, t+em_lines, c='r', linewidth=2.0, zorder=9)
        #axs[0, i].plot(x, p, c='g', alpha=0.5)
        axs[0, i].fill_between(x, g-n, g+n, facecolor='k', alpha=0.3)

        axs[0, i].set_ylim(ylimits)
        axs[1, i].plot(x, 100*(g-t-em_lines)/(t), c='k', linewidth=1.5)
        axs[1, i].fill_between(x, 100*n/g, -100.0*n/(g), facecolor='k', alpha=0.5)
        axs[1, i].axhline(0.0, linestyle='dashed', c='k')

            
        axs[0, i].set_xlim([x.min(), x.max()])
        axs[1, i].set_ylim([-5, 5])

        axs[1, i].set_xlabel(r'Rest Wavelength (\AA)', fontsize=14)
        axs[0, i].set_ylabel(r'Flux (Arbitrary Units)', fontsize=12)
        axs[1, i].set_ylabel(r'Residuals (\%)', fontsize=12)

        #Avoid the overlapping labels
        axs[0, i].yaxis.set_major_locator(ticker.MaxNLocator(prune='lower'))
        axs[0, i].yaxis.set_major_locator(ticker.MaxNLocator(prune='lower'))
        axs[1, i].yaxis.set_major_locator(ticker.MultipleLocator(2))

        #Plot gray shaded regions on the areas that we mask from the chi-squared
        zero_weights=contiguous_zeros(w)
        #import pdb; pdb.set_trace()
        
        for pair in zero_weights:
            
            low=pair[0]
            high=pair[1]
            axs[0, i].axvspan(lams[low]/(np.exp(vel/c_light)), lams[high]/(np.exp(vel/c_light)), alpha=0.5, color='grey')
            axs[1, i].axvspan(lams[low]/(np.exp(vel/c_light)), lams[high]/(np.exp(vel/c_light)), alpha=0.5, color='grey')

        # axs[i, 0].axvspan(Ha_lam[0][0], Ha_lam[0][1], alpha=0.5, color='grey')
        # axs[i, 1].axvspan(Ha_lam[0][0], Ha_lam[0][1], alpha=0.5, color='grey')


    fig.tight_layout()

    return chisq, chisq_per_dof, (fig, axs)




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

    c_light=const.c/1000.0

    #Unpack the parameters
    #galaxy, noise, all_sky, weights, velscale, goodpixels, vsyst, interp_funct, correction_interps, logLams, logLam_gal, fit_wavelengths=parameters
    #Unpack the parameters
    galaxy=parameters['log_galaxy']
   # noise=parameters['log_noise']
   # all_sky=parameters['log_skyspecs']
    #weights=parameters['log_weights']
    velscale=parameters['velscale']
    #goodpixels=parameters['goodpixels']
    vsyst=parameters['dv']
    interp_funct=parameters['linear_interp']
    correction_interps=parameters['correction_interps']
    logLams=parameters['log_lam_template']
    logLam_gal=parameters['log_lam']
    #fit_wavelengths=parameters['fit_wavelengths']
    #c_light=parameters['c_light']
    #O2_sky, base_sky, OH_sky=all_sky
    general_interp, na_interp, positive_only_interp, T_interp=correction_interps


    #Values of theta
    vel =theta['Vel'].value 
    sigma=theta['sigma'].value
    Na_abundance=theta['Na'].value
    general_abundances=np.array([theta['Ca'].value, theta['Fe'].value, theta['C'].value, theta['N'].value, theta['Ti'].value, theta['Mg'].value, theta['Si'].value, theta['Ba'].value])  
    positive_abundances=np.array([theta['as_Fe'].value, theta['Cr'].value,theta['Mn'].value,theta['Ni'].value,theta['Co'].value, theta['Eu'].value,theta['Sr'].value, theta['K'].value,theta['V'].value,theta['Cu'].value])
    age=theta['age'].value
    Z=theta['Z'].value
    imf_x1, imf_x2=theta['imf_x1'].value, theta['imf_x2'].value
    #O2_scale=theta['O2_Scale'].value
    #sky_scale=theta['sky_Scale'].value
    #OH_scale=theta['OH_Scale'].value
    #NaD_sky_scale=theta['NaD_sky_scale'].value

    #Make the base template (age, Z, IMF)
    base_template=_make_model(age, Z, imf_x1, imf_x2, interp_funct, logLams)

    #Make the correction for elements which vary >0.0
    positive_only_correction=_get_correction(positive_only_interp, logLams, np.arange(len(positive_abundances)), positive_abundances, age, Z)

    #Response function for general element corrections
    general_correction=_get_correction(general_interp, logLams, np.arange(len(general_abundances)), general_abundances, age, Z)   

    #Have to treat Na differently
    na_correction=na_interp((Na_abundance, age, Z, logLams))

    #Add things together- see Appendix of Vaughan+ 2018
    template=np.exp(np.log(base_template)+general_correction+positive_only_correction+na_correction)


    if convolve:
        template=convolve_template_with_losvd(template, vel, sigma, velscale=velscale, vsyst=vsyst)[:len(galaxy)]
        logLams=logLam_gal.copy()

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

    model=interp_funct((logLams, age, Z, imf_x1, imf_x2))

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

    #The interpolator expects a list of 6 numbers. Meshgrid the two arrays which are of different lengths
    # (the indices and the number of elements to enhance) and then create lists of ages, Zs and IMFs of the correct
    # shapes. Then do the same for the abundances. Stack together and pass to the interpolator object!

    points = np.meshgrid(elems, logLams, indexing='ij')
    flat = np.array([m.flatten() for m in points])
    #flat is now an array of points of shape 2, len(indices)*len(elems)
    #make arrays of the other variables of length len(indices)*len(elems)
    ages=np.ones_like(points[0])*age
    Zs=np.ones_like(points[0])*Z
    

    #Get the correct abundance for each element- luckily we can index the abundance array by the integer element array
    abunds=abunds[points[0]]

    # import pdb; pdb.set_trace()

    #Stack together
    xi=np.vstack((flat[0, :], abunds.ravel(), ages.ravel(), Zs.ravel(), flat[1, :]))
    #Do the interpolation
    out_array = interpolator(xi.T)
    #reshape everything to be (len(indices), len(elements))
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


    logLams, template, base_template=get_best_fit_template(theta, parameters, convolve=convole)

    interp1=si.interp1d(np.exp(logLams), template, fill_value='extrapolate')
    lin_template=interp1(fit_class.lin_lam)

    interp2=si.interp1d(np.exp(fit_class.logLam_template), base_template, fill_value='extrapolate')
    lin_base_template=interp2(fit_class.lin_lam)

    return lin_template, lin_base_template



def lnlike(theta, parameters, plot=False, ret_specs=False):

    """
    The log-Likelihood function of the fitting
    TODO: Add more here!
    Arguments:
        theta (lmfit Parameters): An lmfit Parameters object
        parameters (dict): A dictionary containing... TODO: Rename this and tidy up!
        plot (Boolean): Deprecated. TODO remove
        ret_specs (boolean): Default is False. If True, return a series of spectra made during the fitting process
    Returns:
        * likelihood (float): The log-likelihood of the fit parameters
        if ret_specs is True, also return:
        * TODO List all these
        * likelihood, -1.0*likelihood/n_dof, [lam, t, e, s, skies, g_lines, p, w]  

    """ 


    

    #Unpack the parameters
    galaxy=parameters['log_galaxy']
    noise=parameters['log_noise']
    all_sky=parameters['log_skyspecs']
    weights=parameters['log_weights']
    emission_lines=parameters['emission_lines']
    velscale=parameters['velscale']
    #goodpixels=parameters['goodpixels']
    vsyst=parameters['dv']
    #interp_funct=parameters['linear_interp']
    correction_interps=parameters['correction_interps']
    #logLams=parameters['log_lam_template']
    logLam_gal=parameters['log_lam']
    fit_wavelengths=parameters['fit_wavelengths']
    c_light=parameters['c_light']
    instrumental_resolution=parameters['instrumental_resolution']




    #all_sky, weights, velscale, goodpixels, vsyst, interp_funct, correction_interps, logLams, logLam_gal, fit_wavelengths=parameters

    if all_sky is not None:
        O2_sky, base_sky, OH_sky, NaD_sky=all_sky

    general_interp, na_interp, positive_only_interp, T_interp=correction_interps



    #Values of theta
    vel =theta['Vel'].value 
    # sigma=theta['sigma'].value
    # Na_abundance=theta['Na'].value
    # general_abundances=np.array([theta['Ca'].value, theta['Fe'].value, theta['C'].value, theta['N'].value, theta['Ti'].value, theta['Mg'].value, theta['Si'].value, theta['Ba'].value])  
    # positive_abundances=np.array([theta['as_Fe'].value, theta['Cr'].value,theta['Mn'].value,theta['Ni'].value,theta['Co'].value, theta['Eu'].value,theta['Sr'].value, theta['K'].value,theta['V'].value,theta['Cu'].value])
    # age=theta['age'].value
    # Z=theta['Z'].value
    # imf_x1, imf_x2=theta['imf_x1'].value, theta['imf_x2'].value

    #Scale the noise by some value f
    ln_f=theta['ln_f'].value


    if all_sky is not None: 
        O2_scale=theta['O2_Scale'].value
        sky_scale=theta['sky_Scale'].value
        OH_scale=theta['OH_Scale'].value
        NaD_sky_scale=theta['NaD_sky_scale'].value

    if emission_lines is not None:
        vel_gas=theta['Vel_em'].value
        sig_gas=theta['sig_em'].value
        Ha_flux=np.exp(theta['Ha'].value)
        Hbeta_flux=np.exp(theta['Hb'].value)
        
        SII_6716=np.exp(theta['SII_6716'].value)
        SII_6731=np.exp(theta['SII_6731'].value)
        OIII=np.exp(theta['OIII'].value)
        OI=np.exp(theta['OI'].value)
        NII=np.exp(theta['NII'].value)



    #SINGLE POWER LAW IMF
    #theta['imf_x1'].set(theta['imf_x2'].value)


    _, temp, base_template=get_best_fit_template(theta, parameters, convolve=True)

    
    #Ranges we fit over- these have to change with redshift
    fit_ranges=fit_wavelengths*(np.exp(vel/c_light))

    #Ranges of pixels to mask. These also have to change with z
    #mask_ranges=masked_wavelengths*(np.exp(vel/c_light))

    #make the array to mask out things from the Chi-squared
    # pixel_mask=np.ones_like(galaxy, dtype=bool)
    # for array in mask_ranges:
    #     m=make_mask(logLam_gal, array)
    #     pixel_mask= m & pixel_mask
   
    chisqs=np.zeros_like(galaxy)

    #Median both the galaxy and noise. Added in October 2017!
    #Changed back in Jan 2018!
    #galmedian=np.median(galaxy)
    #t_median=np.median(temp)





    #Make the emission lines:
    if emission_lines is not None:
        unconvolved_em_lines=Hbeta_flux*emission_lines[:, 0] + Ha_flux*emission_lines[:, 1] + SII_6716*emission_lines[:, 2] + SII_6731*emission_lines[:, 3] + OIII * emission_lines[:, 4] + OI*emission_lines[:, 5] + NII*emission_lines[:, 6]
        convolved_em_lines=convolve_template_with_losvd(unconvolved_em_lines, vel_gas, sig_gas, velscale=velscale, vsyst=vsyst)[:len(galaxy)]
    else:
        convolved_em_lines=np.zeros_like(galaxy)


    #If we want to return the spectra, make empty lists to append to
    #if ret_specs:
    lams=[]
    specs=[]
    temps=[]
    gas_lines=[]
    residuals=[]
    errors=[]
    skies=[]
    polys=[]


    #Convolve with the instrumental resolution
    if instrumental_resolution is not None:
        temp=gaussian_filter1d(temp, instrumental_resolution/velscale)
        #Bit hacky
        #Replace the first and last few values of temp (which get set to zero by the convolution)
        #to the median value of the whole thing
        temp[temp==0]=np.median(temp)




    #Do the fitting
    for i, fit_range in enumerate(fit_ranges):

        #Mask around each fit range
        
        gmask=np.where((np.exp(logLam_gal)>fit_range[0]) & (np.exp(logLam_gal)<fit_range[1]))
        
        g=galaxy[gmask]
        n=noise[gmask]
        t=temp[gmask]
        gas=convolved_em_lines[gmask]


        # galmedian=np.median(g)
        # tempmedian=np.median(t)

        # g/=galmedian
        # n/=galmedian
        # t/=tempmedian
        
        if all_sky is not None:
            #Mask the sky spectra
            O2_s=O2_sky[gmask]
            s=base_sky[gmask]
            OH_s=OH_sky[gmask]
            NaD_s=NaD_sky[gmask]

            #Make our sky model
            sky=O2_scale*O2_s+sky_scale*s+OH_scale*OH_s+NaD_sky_scale*NaD_s
            #sky/=galmedian
        else:
            sky=np.zeros_like(g)



        #subtract the sky
        g_corrected=(g -sky)
        

        #Order of the polynomial
        morder=int((fit_range[1]-fit_range[0])/100)
        
        
        #Fit the polynomials, weighting by the noise
        #poly=C.fit_continuum(gmask[0], g/t, n**2, clip=[2, 10.0, 10.0], order=morder)
        poly=_fit_legendre_polys(g_corrected/t, morder, weights=1.0/n**2)

        #Scale the noise by some fraction ln_f
        n_corrected=np.sqrt((1+np.exp(2*ln_f))*n**2)         

        #Calculate the chi_squared
        chisqs[gmask]=(((g_corrected + poly*gas -t*poly)/n_corrected)**2)

        
        lams.append(np.exp(logLam_gal[gmask]))
        temps.append(poly*t)
        gas_lines.append(poly*gas)
        residuals.append(g_corrected + poly*gas-poly*t)
        errors.append(n_corrected)
        specs.append(g_corrected)
        skies.append(sky)
        polys.append(poly)



    #We may have a gap in the ranges we want to fit over
    #This ensures that only pixels in the fit ranges contribute to the chisquared
    all_fit_ranges_mask=np.zeros_like(galaxy, dtype=bool)

    for fit_range in fit_ranges:
        this_fit_range_mask=(np.exp(logLam_gal)>fit_range[0]) & (np.exp(logLam_gal)<fit_range[1])
        all_fit_ranges_mask[this_fit_range_mask]=1


    #Sum the chisquareds, masking out the pixels we don't want
    chisq=np.sum((chisqs*weights)[all_fit_ranges_mask])

    #Count all of the things we're varying
    n_variables=len([thing for thing in theta if theta[thing].vary])

    
    n_masked_pixels=len(np.where(weights[all_fit_ranges_mask]==0)[0])
    n_dof=len(chisqs[all_fit_ranges_mask]) - n_masked_pixels - n_variables  
            
    ###########################################################################

       
    lam=np.concatenate(lams)
    t=np.concatenate(temps)
    e=np.concatenate(errors)
    s=np.concatenate(specs)
    skies=np.concatenate(skies)
    g_lines=np.concatenate(gas_lines)
    p=np.concatenate(polys)
    w=weights[all_fit_ranges_mask].astype(bool)


    #Log likelihood- chisqaured plus sum of errors, which now depend on ln_f
    likelihood=-0.5*(chisq + np.sum(np.log(e**2)*w))


    if ret_specs:
        return likelihood, -1.0*likelihood/n_dof, [lam, t, e, s, skies, g_lines, p, w]  

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


    x_vals=np.linspace(-1, 1, len(ratio))
    coeffs=np.polynomial.legendre.legfit(x_vals, ratio, morder, w=weights)

    polynomial=np.polynomial.legendre.legval(x_vals, coeffs)

    return polynomial


def init(func, *args, **kwargs):
    
    global parameters, logLam
    parameters, logLam, ndim=func(*args, **kwargs)






def convolve_template_with_losvd(template, vel=0.0, sigma=0.0, velscale=None, vsyst=0.0):

    """
    **From Michele Cappellari's PPXF code**

    Given a template, convolve it with a line-of-sight velocity distribution. 
    
    """

    t_rfft, npad=_templates_rfft(template)
    losvd_rfft=_losvd_rfft(vel, sigma, npad, velscale, npad, vsyst=vsyst)

    convolved_t=np.fft.irfft(t_rfft*losvd_rfft)

    return convolved_t


def _losvd_rfft(vel, sig, pad, velscale, npad, vsyst=0.0):

    """
    **From Michele Cappellari's PPXF code**
    """


    nl = npad//2 + 1

    vel=(vel+vsyst)/velscale
    sig/=velscale



    a, b = np.array([vel, 0.0])/sig

    w = np.linspace(0, np.pi*sig, nl)
    #analytic FFT of LOSVD
    losvd_rfft = np.exp(1j*a*w - 0.5*(1 + b**2)*w**2)

    return np.conj(losvd_rfft)

    

def _templates_rfft(templates):
    """
    **From Michele Cappellari's PPXF code**
    Pre-compute the FFT (of real input) of all templates

    """
    npad = 2**int(np.ceil(np.log2(templates.shape[0])))
    templates_rfft = np.fft.rfft(templates, n=npad, axis=0)

    return templates_rfft, npad


def emline(logLam_temp, line_wave, FWHM_gal):
    """
    **From Michele Cappellari's PPXF code**
    Instrumental Gaussian line spread function integrated within the
    pixels borders. The function is normalized in such a way that

        integ.sum() = 1

    For sigma=FWHM_gal/2.355 larger than one pixels, this function
    quickly converges to the normalized Gaussian function:

        gauss = dLogLam * np.exp(-0.5*(x/xsig)**2) / (np.sqrt(2*np.pi)*xsig)

    :param logLam_temp: np.log(wavelength) in Angstrom
    :param line_wave: lines wavelength in Angstrom
    :param FWHM_gal: FWHM in Angstrom. This can be a scalar or the
        name of a function wich returns the FWHM for given wavelength.
    :return: LSF computed for every logLam_temp

    """
    if callable(FWHM_gal):
        FWHM_gal = FWHM_gal(line_wave)

    # Compute pixels borders for Gaussian integration
    logLamBorders = (logLam_temp[1:] + logLam_temp[:-1])/2
    xsig = FWHM_gal/2.355/line_wave    # sigma in x=logLambda units

    # Perform pixel integration
    x = logLamBorders[:, None] - np.log(line_wave)
    integ = 0.5*np.diff(special.erf(x/(np.sqrt(2)*xsig)), axis=0)

    return np.pad(integ, ((1,), (0,)), 'constant')

###############################################################################
# NAME:
#   EMISSION_LINES
#
# MODIFICATION HISTORY:
#   V1.0.0: Michele Cappellari, Oxford, 7 January 2014
#   V1.1.0: Fixes [OIII] and [NII] doublets to the theoretical flux ratio.
#       Returns line names together with emission lines templates.
#       MC, Oxford, 3 August 2014
#   V1.1.1: Only returns lines included within the estimated fitted wavelength range.
#       This avoids identically zero gas templates being included in the PPXF fit
#       which can cause numerical instabilities in the solution of the system.
#       MC, Oxford, 3 September 2014
#   V1.2.0: Perform integration over the pixels of the Gaussian line spread function
#       using the new function emline(). Thanks to Eric Emsellem for the suggestion.
#       MC, Oxford, 10 August 2016
#   V1.2.1: Allow FWHM_gal to be a function of wavelength. MC, Oxford, 16 August 2016

def emission_lines(logLam_temp, lamRange_gal, FWHM_gal, quiet=True):
    """
    **From Michele Cappellari's PPXF code**
    Generates an array of Gaussian emission lines to be used as gas templates in PPXF.
    These templates represent the instrumental line spread function (LSF) at the
    set of wavelengths of each emission line.

    Additional lines can be easily added by editing the code of this procedure,
    which is meant as a template to be modified by the users where needed.

    For accuracy the Gaussians are integrated over the pixels boundaries.
    This integration is only useful for quite unresolved Gaussians but one should
    keep in mind that, if the LSF is not well resolved, the input spectrum is not
    properly sampled and one is wasting useful information from the spectrograph!

    The [OI], [OIII] and [NII] doublets are fixed at theoretical flux ratio~3.

    :param logLam_temp: is the natural log of the wavelength of the templates in
        Angstrom. logLam_temp should be the same as that of the stellar templates.
    :param lamRange_gal: is the estimated rest-frame fitted wavelength range
        Typically lamRange_gal = np.array([np.min(wave), np.max(wave)])/(1 + z),
        where wave is the observed wavelength of the fitted galaxy pixels and
        z is an initial rough estimate of the galaxy redshift.
    :param FWHM_gal: is the instrumantal FWHM of the galaxy spectrum under study
        in Angstrom. One can pass either a scalar or the name "func" of a function
        func(wave) which returns the FWHM for a given vector of input wavelengths.
    :return: emission_lines, line_names, line_wave

    """
    # Balmer Series:      Hdelta   Hgamma    Hbeta   Halpha
    line_wave = np.array([4101.76, 4340.47, 4861.33, 6562.80])  # air wavelengths
    line_names = np.array(['Hdelta', 'Hgamma', 'Hbeta', 'Halpha'])
    emission_lines = emline(logLam_temp, line_wave, FWHM_gal)

    #                 -----[OII]-----    -----[SII]-----
    lines = np.array([3726.03, 3728.82, 6716.47, 6730.85])  # air wavelengths
    names = np.array(['[OII]3726', '[OII]3729', '[SII]6716', '[SII]6731'])
    gauss = emline(logLam_temp, lines, FWHM_gal)
    emission_lines = np.append(emission_lines, gauss, 1)
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, lines)

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #                 -----[OIII]-----
    lines = np.array([4958.92, 5006.84])    # air wavelengths
    doublet = 0.33*emline(logLam_temp, lines[0], FWHM_gal) + emline(logLam_temp, lines[1], FWHM_gal)
    emission_lines = np.append(emission_lines, doublet, 1)
    line_names = np.append(line_names, '[OIII]5007d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #                  -----[OI]-----
    lines = np.array([6300.30, 6363.67])    # air wavelengths
    doublet = emline(logLam_temp, lines[0], FWHM_gal) + 0.33*emline(logLam_temp, lines[1], FWHM_gal)
    emission_lines = np.append(emission_lines, doublet, 1)
    line_names = np.append(line_names, '[OI]6300d') # single template for this doublet
    line_wave = np.append(line_wave, lines[0])

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #                 -----[NII]-----
    lines = np.array([6548.03, 6583.41])    # air wavelengths
    doublet = 0.33*emline(logLam_temp, lines[0], FWHM_gal) + emline(logLam_temp, lines[1], FWHM_gal)
    emission_lines = np.append(emission_lines, doublet, 1)
    line_names = np.append(line_names, '[NII]6583d') # single template for this doublet
    line_wave = np.append(line_wave, lines[1])

    # Only include lines falling within the estimated fitted wavelength range.
    # This is important to avoid instabilities in the pPXF system solution
    #
    w = (line_wave > lamRange_gal[0]) & (line_wave < lamRange_gal[1])
    emission_lines = emission_lines[:, w]
    line_names = line_names[w]
    line_wave = line_wave[w]

    if not quiet:
        print('Emission lines included in gas templates:')
        print(line_names)

    return emission_lines, line_names, line_wave

###############################################################################


###############################################################################
#
# NAME:
#   LOG_REBIN
#
# MODIFICATION HISTORY:
#   V1.0.0: Using interpolation. Michele Cappellari, Leiden, 22 October 2001
#   V2.0.0: Analytic flux conservation. MC, Potsdam, 15 June 2003
#   V2.1.0: Allow a velocity scale to be specified by the user.
#       MC, Leiden, 2 August 2003
#   V2.2.0: Output the optional logarithmically spaced wavelength at the
#       geometric mean of the wavelength at the border of each pixel.
#       Thanks to Jesus Falcon-Barroso. MC, Leiden, 5 November 2003
#   V2.2.1: Verify that lamRange[0] < lamRange[1].
#       MC, Vicenza, 29 December 2004
#   V2.2.2: Modified the documentation after feedback from James Price.
#       MC, Oxford, 21 October 2010
#   V2.3.0: By default now preserve the shape of the spectrum, not the
#       total flux. This seems what most users expect from the procedure.
#       Set the keyword /FLUX to preserve flux like in previous version.
#       MC, Oxford, 30 November 2011
#   V3.0.0: Translated from IDL into Python. MC, Santiago, 23 November 2013
#   V3.1.0: Fully vectorized log_rebin. Typical speed up by two orders of magnitude.
#       MC, Oxford, 4 March 2014
#   V3.1.1: Updated documentation. MC, Oxford, 16 August 2016

def log_rebin(lamRange, spec, oversample=1, velscale=None, flux=False):
    """
    **From Michele Cappellari's PPXF code**
    Logarithmically rebin a spectrum, while rigorously conserving the flux.
    Basically the photons in the spectrum are simply redistributed according
    to a new grid of pixels, with non-uniform size in the spectral direction.
    
    When the flux keyword is set, this program performs an exact integration 
    of the original spectrum, assumed to be a step function within the 
    linearly-spaced pixels, onto the new logarithmically-spaced pixels. 
    The output was tested to agree with the analytic solution.

    :param lamRange: two elements vector containing the central wavelength
        of the first and last pixels in the spectrum, which is assumed
        to have constant wavelength scale! E.g. from the values in the
        standard FITS keywords: LAMRANGE = CRVAL1 + [0, CDELT1*(NAXIS1 - 1)].
        It must be LAMRANGE[0] < LAMRANGE[1].
    :param spec: input spectrum.
    :param oversample: can be used, not to loose spectral resolution,
        especally for extended wavelength ranges and to avoid aliasing.
        Default: OVERSAMPLE=1 ==> Same number of output pixels as input.
    :param velscale: velocity scale in km/s per pixels. If this variable is
        not defined, then it will contain in output the velocity scale.
        If this variable is defined by the user it will be used
        to set the output number of pixels and wavelength scale.
    :param flux: (boolean) True to preserve total flux. In this case the
        log rebinning changes the pixels flux in proportion to their
        dLam so the following command will show large differences
        beween the spectral shape before and after LOG_REBIN:
           plt.plot(exp(logLam), specNew)  # Plot log-rebinned spectrum
           plt.plot(np.linspace(lamRange[0], lamRange[1], spec.size), spec)
        By defaul, when this is False, the above two lines produce
        two spectra that almost perfectly overlap each other.
    :return: [specNew, logLam, velscale] where logLam is the natural
        logarithm of the wavelength and velscale is in km/s.

    """
    lamRange = np.asarray(lamRange)
    assert len(lamRange) == 2, 'lamRange must contain two elements'
    assert lamRange[0] < lamRange[1], 'It must be lamRange[0] < lamRange[1]'
    assert spec.ndim == 1, 'input spectrum must be a vector'
    n = spec.shape[0]
    m = int(n*oversample)

    dLam = np.diff(lamRange)/(n - 1.)        # Assume constant dLam
    lim = lamRange/dLam + [-0.5, 0.5]        # All in units of dLam
    borders = np.linspace(*lim, num=n+1)     # Linearly
    logLim = np.log(lim)

    c = 299792.458                           # Speed of light in km/s
    if velscale is None:                     # Velocity scale is set by user
        velscale = np.diff(logLim)/m*c       # Only for output
    else:
        logScale = velscale/c
        m = int(np.diff(logLim)/logScale)    # Number of output pixels
        logLim[1] = logLim[0] + m*logScale

    newBorders = np.exp(np.linspace(*logLim, num=m+1)) # Logarithmically
    k = (newBorders - lim[0]).clip(0, n-1).astype(int)

    specNew = np.add.reduceat(spec, k)[:-1]  # Do analytic integral
    specNew *= np.diff(k) > 0    # fix for design flaw of reduceat()
    specNew += np.diff((newBorders - borders[k])*spec[k])

    if not flux:
        specNew /= np.diff(newBorders)

    # Output log(wavelength): log of geometric mean
    logLam = np.log(np.sqrt(newBorders[1:]*newBorders[:-1])*dLam)

    return specNew, logLam, velscale

###############################################################################
# NAME:
#   GAUSSIAN_FILTER1D
#
# MODIFICATION HISTORY:
#   V1.0.0: Written as a replacement for the Scipy routine with the same name,
#       to be used with variable sigma per pixel. MC, Oxford, 10 October 2015

def gaussian_filter1d(spec, sig):
    """
    **From Michele Cappellari's PPXF code**
    Convolve a spectrum by a Gaussian with different sigma for every pixel.
    If all sigma are the same this routine produces the same output as
    scipy.ndimage.gaussian_filter1d, except for the border treatment.
    Here the first/last p pixels are filled with zeros.
    When creating a template library for SDSS data, this implementation
    is 60x faster than a naive for loop over pixels.

    :param spec: vector with the spectrum to convolve
    :param sig: vector of sigma values (in pixels) for every pixel
    :return: spec convolved with a Gaussian with dispersion sig

    """
    sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
    p = int(np.ceil(np.max(3*sig)))
    m = 2*p + 1  # kernel size
    x2 = np.linspace(-p, p, m)**2

    n = spec.size
    a = np.zeros((m, n))
    for j in range(m):   # Loop over the small size of the kernel
        a[j, p:-p] = spec[j:n-m+j+1]

    gau = np.exp(-x2[:, None]/(2*sig**2))
    gau /= np.sum(gau, 0)[None, :]  # Normalize kernel

    conv_spectrum = np.sum(a*gau, 0)

    return conv_spectrum
