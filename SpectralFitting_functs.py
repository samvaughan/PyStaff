

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np 
import scipy.constants as const
import ppxf_util as util

from stellarpops.tools import fspTools as FT

from scipy import special
import scipy.interpolate as si


def rebin_MUSE_spectrum(lamdas, flux, errors, pixel_weights, skyspecs=None, c=299792.458):
    ##############################################################################################################################################################


    #flux_median=np.median(flux)

    # flux/=flux_median
    # errors/=flux_median
 
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

    #GoodPixels from pPXF
    

    #Log rebin them
    galaxy, logLam, velscale = util.log_rebin(lam_range_gal, flux)
    noise, _, _=util.log_rebin(lam_range_gal, errors, velscale=velscale) 
    weights, _, _=util.log_rebin(lam_range_gal, pixel_weights, velscale=velscale)
    #import pdb; pdb.set_trace()

    all_sky=None
    if skyspecs is not None:
        O2_sky, _, _=util.log_rebin(lam_range_gal, skyspecs[0, :], velscale=velscale)
        sky, _, _=util.log_rebin(lam_range_gal, skyspecs[1, :], velscale=velscale) 
        OH_sky, _, _=util.log_rebin(lam_range_gal, skyspecs[2, :], velscale=velscale)
        NaD_sky, _, _=util.log_rebin(lam_range_gal, skyspecs[3, :], velscale=velscale)

        all_sky=np.array([O2_sky, sky, OH_sky, NaD_sky])

        

    goodpixels = np.arange(len(galaxy))     

    return galaxy, noise, all_sky, weights, velscale, goodpixels, lam_range_gal, logLam
################################################################################################################################################################


def get_start_vals_and_bounds(theta):

    """
    Given a set of parameter values in lmfit format, get their values and bounds as numpy arrays.
    This is taken from the lmfit code itself
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
    Given a set of starting values and spreads for each value, make a set of random starting positions for each walker
    """

    ball=np.random.randn(start_values.shape[0], nwalkers)*stds[:, None] + start_values[:, None]

    return ball

def make_mask(lamdas, wavelengths):
    mask=np.ones_like(lamdas, dtype=bool)
    for pair in wavelengths:
        low, high=pair
        #import pdb; pdb.set_trace()
        mask = mask & (lamdas<low) | (lamdas>high)

    return mask

def contiguous_zeros(a):
    # Find the beginning and end of each run of 0s in the weights 
    # from https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def plot_fit(theta, parameters, fig=None, axs=None, color='b'):

    #c_light=const.c/1000.0

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
    base_template=FT.make_model_CvD(age, Z, imf_x1, imf_x2, interp_funct, logLams)

    #Make the correction for elements which vary >0.0
    positive_only_correction=FT.get_correction(positive_only_interp, logLams, np.arange(len(positive_abundances)), positive_abundances, age, Z)

    #Response function for general element corrections
    general_correction=FT.get_correction(general_interp, logLams, np.arange(len(general_abundances)), general_abundances, age, Z)   

    #Have to treat Na differently
    na_correction=na_interp((Na_abundance, age, Z, logLams))

    #Add things together- see Appendix of Vaughan+ 2018
    template=np.exp(np.log(base_template)+general_correction+positive_only_correction+na_correction)


    if convolve:
        template=FT.convolve_template_with_losvd(template, vel, sigma, velscale=velscale, vsyst=vsyst)[:len(galaxy)]
        logLams=logLam_gal.copy()

    return logLams, template, base_template

def get_linear_best_fit_template(theta, parameters, fit_class, convole=True):


    logLams, template, base_template=get_best_fit_template(theta, parameters, convolve=convole)

    interp1=si.interp1d(np.exp(logLams), template, fill_value='extrapolate')
    lin_template=interp1(fit_class.lin_lam)

    interp2=si.interp1d(np.exp(fit_class.logLam_template), base_template, fill_value='extrapolate')
    lin_base_template=interp2(fit_class.lin_lam)

    return lin_template, lin_base_template



def lnlike(theta, parameters, plot=False, ret_specs=False):

    """
    (log) Likelihood function
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

    temp_medianed=temp#/t_median
    galaxy_medianed=galaxy#/galmedian
    noise_medianed=noise#/galmedian

    #overall_poly=FT.fit_legendre_polys(galaxy_medianed/temp_medianed, 20, weights=1.0/noise_medianed**2)
    #temp_medianed=temp_medianed*overall_poly

    #Make the emission lines:
    if emission_lines is not None:

        unconvolved_em_lines=Hbeta_flux*emission_lines[:, 0] + Ha_flux*emission_lines[:, 1] + SII_6716*emission_lines[:, 2] + SII_6731*emission_lines[:, 3] + OIII * emission_lines[:, 4] + OI*emission_lines[:, 5] + NII*emission_lines[:, 6]
        convolved_em_lines=FT.convolve_template_with_losvd(unconvolved_em_lines, vel_gas, sig_gas, velscale=velscale, vsyst=vsyst)[:len(galaxy)]
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



    #Do the fitting
    for i, fit_range in enumerate(fit_ranges):

        #Mask around each fit range
        
        gmask=np.where((np.exp(logLam_gal)>fit_range[0]) & (np.exp(logLam_gal)<fit_range[1]))
        
        g=galaxy_medianed[gmask]
        n=noise_medianed[gmask]
        t=temp_medianed[gmask]
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
        poly=FT.fit_legendre_polys(g_corrected/t, morder, weights=1.0/n**2)

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


def init(func, *args, **kwargs):
    
    global parameters, logLam
    parameters, logLam, ndim=func(*args, **kwargs)



def make_mask(lamdas, wavelengths):
    mask=np.ones_like(lamdas, dtype=bool)
    for pair in wavelengths:
        low, high=pair
        #import pdb; pdb.set_trace()
        mask = mask & (lamdas<low) | (lamdas>high)

    return mask


###############################################################################
#FUNCTIONS TAKEN FROM MICHELE CAPPELLARI'S pPXF CODE
#
# Copyright (C) 2001-2016, Michele Cappellari
# E-mail: michele.cappellari_at_physics.ox.ac.uk
#
# Updated versions of the software are available from my web page
# http://purl.org/cappellari/software
#
# If you have found this software useful for your research,
# I would appreciate an acknowledgment to the use of the
# "Penalized Pixel-Fitting method by Cappellari & Emsellem (2004)"
#  and/or "by Cappellari (2017)" if you use the new features.
#
# This software is provided as is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
################################################################################

def emline(logLam_temp, line_wave, FWHM_gal):
    """
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
