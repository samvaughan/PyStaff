#! /usr/bin/env python
from __future__ import print_function

##############################################################################
import numpy as np
import emcee
import os, sys
import scipy.interpolate as si 
import lmfit as LM

from pystaff.SpectralFitting import SpectralFit
from pystaff import SpectralFitting_functs as SF


#Likelihood function here. We could put it in the SpectraFitting class, but when 
#working with MPI on a cluster that would mean we'd need to pickle the fit_settings
#dictionary, which massively slows things down
def lnprob(T, theta, var_names, bounds, ret_specs=False):

    #Log prob function. T is an array of values

    
    assert len(T)==len(var_names), 'Error! The number of variables and walker position shapes are different'

    #Prior information comes from the parameter bounds now
    if np.any(T > bounds[:, 1]) or np.any(T < bounds[:, 0]):
        return -np.inf


    #make theta from the emcee walker positions
    for name, val in zip(var_names, T):
        theta[name].value = val

    if ret_specs==False:
        ll=SF.lnlike(theta, fit.fit_settings)
        return ll
    else:
        return SF.lnlike(theta, fit.fit_settings, ret_specs=True)


#Can select either Kroupa or Salpeter to use with the SSP models
element_imf='kroupa'


####################################################
#Read in the data
datafile = 'data/example_spectrum.txt'
lamdas, flux, errors, instrumental_resolution_values = np.genfromtxt(datafile, unpack=True)

# The instrumental resolution can be included if it's known. 
# This must be a function which gives the value of the instrumental resolution in km/s at any wavelength value
# We use this to convolve the templates with __before__ starting the fitting. 
# It's okay to only have measured instrumental resolution values near where your data are- but the function must return some value for the instrumental resolution at any wavelength
# Otherwise leave it as None

def instrumental_resolution_function(wavelength):

    import scipy.interpolate as si 

    instrumental_resolution_interpolator = si.interp1d(lamdas, instrumental_resolution_values, bounds_error=False, fill_value=(instrumental_resolution_values[0], instrumental_resolution_values[-1]))

    return instrumental_resolution_interpolator(wavelength)

instrumental_resolution=instrumental_resolution_function

# Sky Spectra
# Give a list of 1D sky spectra to be scaled and subtracted during the fit
# Otherwise leave sky as None
skyspecs=None
# ######################################################


# ######################################################
# Mask out regions that we don't want to fit, e.g. around telluric residuals, particularly nasty skylines, etc
# THESE SHOULD BE OBSERVED WAVELENGTHS
# A few examples of areas I often avoid due to skylines or telluric residuals
telluric_lam_1=np.array([[6862, 6952]])
telluric_lam_2=np.array([[7586, 7694]])
skylines=np.array([[8819, 8834], [8878.0, 8893], [8911, 8925], [8948, 8961]])

masked_wavelengths=np.vstack([telluric_lam_1, telluric_lam_2, skylines]).reshape(-1, 1, 2)
string_masked_wavelengths=["{} to {}".format(pair[0][0], pair[0][1]) for pair in masked_wavelengths]

#Make a mask of pixels we don't want
pixel_mask=np.ones_like(flux, dtype=bool)
for array in masked_wavelengths:   
    m=SF.make_mask(lamdas, array)
    pixel_mask= m & pixel_mask

#Now switch the weights of these pixels to 0
pixel_weights=np.ones_like(flux)
pixel_weights[~pixel_mask]=0.0


#Wavelengths we'll fit between.
#Split into 4 to make the multiplicative polynomials faster
fit_wavelengths=np.array([[4750, 5600], [5600, 6800], [6800, 8000], [8000,  9200]])
string_fit_wavelengths=["{} to {}".format(pair[0], pair[1]) for pair in fit_wavelengths]

#FWHM.
#This should be the FWHM in pixels of the instrument used to observe the spectrum.
FWHM_gal=3.0

#Now set up the spectral fitting class
print('Setting up the fit')
# These should be the location of the folder containing all the templates
# The code will use 'glob' to search for all templates matching the correct filenames. 
base_template_location = '~/Science/stellarpops/CvD2/vcj_twopartimf/vcj_ssp_v8'
varelem_template_location = '~/Science/stellarpops/CvD2/atlas_rfn_v3'

fit=SpectralFit(lamdas, flux, errors, pixel_weights, fit_wavelengths, FWHM_gal, instrumental_resolution=instrumental_resolution, skyspecs=skyspecs, element_imf=element_imf, base_template_location=base_template_location, varelem_template_location=varelem_template_location)
fit.set_up_fit()


#Here are the available fit parameters
#They can easily be switched off by changing vary to False
#The min and max values act as flat priors
theta=LM.Parameters()
#LOSVD parameters
theta.add('Vel', value=1600, min=-1000.0, max=10000.0)
theta.add('sigma', value=330.0, min=10.0, max=500.0)

#Abundance of Na. Treat this separately, since it can vary up to +1.0 dex
theta.add('Na', value=0.5, min=-0.45, max=1.0, vary=True)

# Abundance of Carbon. Treat this separately, since its templates are at +/- 0.15 dex rather than +/- 0.3
theta.add('C', value=0.0, min=-0.2, max=0.2, vary=True)

#Abundance of elements which can vary positively and negatively
theta.add('Ca', value=0.0,  min=-0.45, max=0.45, vary=True)
theta.add('Fe', value=0.0, min=-0.45, max=0.45, vary=True)
theta.add('N', value=0.0, min=-0.45, max=0.45, vary=True)
theta.add('Ti', value=0.0, min=-0.45, max=0.45, vary=True)
theta.add('Mg', value=0.0, min=-0.45, max=0.45, vary=True)
theta.add('Si', value=0.0, min=-0.45, max=0.45, vary=True)
theta.add('Ba', value=0.0, min=-0.45, max=0.45, vary=True)

#Abundance of elements which can only vary above 0.0
theta.add('as_Fe', value=0.0, min=0.0, max=0.45, vary=True)
theta.add('Cr', value=0.0, min=0.0, max=0.45, vary=False)
theta.add('Mn', value=0.0, min=0.0, max=0.45, vary=False)
theta.add('Ni', value=0.0, min=0.0, max=0.45, vary=False)
theta.add('Co', value=0.0, min=0.0, max=0.45, vary=False)
theta.add('Eu', value=0.0, min=0.0, max=0.45, vary=False)
theta.add('Sr', value=0.0, min=0.0, max=0.45, vary=False)
theta.add('K', value=0.0, min=0.0, max=0.45, vary=False)
theta.add('V', value=0.0, min=0.0, max=0.45, vary=False)
theta.add('Cu', value=0.0, min=0.0, max=0.45, vary=False)

#Emission line kinematics
#Each line is fixed to the same velocity and sigma
theta.add('Vel_em', value=1600, min=0.0, max=10000)
theta.add('sig_em', value=200.0, min=10.0, max=500.0)

#Emission line strengths
#These are log flux- they get exponentiated in the likelihood function
theta.add('Ha', value=1.0, min=-10.0, max=10.0)
theta.add('Hb', value=0.3, min=-10.0, max=10.0)
theta.add('NII', value=0.5, min=-10.0, max=10.0)
theta.add('SII_6716', value=0.5, min=-10.0, max=10.0)
theta.add('SII_6731', value=0.5, min=-10.0, max=10.0)
theta.add('OIII', value=-2.0, min=-10.0, max=10.0)
theta.add('OI', value=-2.0, min=-10.0, max=10.0)

#Base population parameters
#Age, Metallicity, and the two IMF slopes
theta.add('age', value=13.0, min=1.0, max=14.0)
theta.add('Z', value=0.0, min=-1.0, max=0.2)
theta.add('imf_x1', value=2.35, min=0.5, max=3.5)
theta.add('imf_x2', value=2.35, min=0.5, max=3.5) 

#Strengths of skylines 
theta.add('O2_Scale', value=0.0, min=-100000000, max=100000000, vary=False) 
theta.add('sky_Scale', value=0.0, min=-100000000, max=100000000, vary=False) 
theta.add('OH_Scale', value=0.0, min=-100000000, max=100000000, vary=False) 
theta.add('NaD_sky_scale', value=0.0, min=-100000000, max=100000000, vary=False)

#Option to rescale the error bars up or down
theta.add('ln_f', value=0.0, min=-5.0, max=5.0, vary=True)

#Select the parameters we're varying, ignore the fixed ones
variables=[thing for thing in theta if theta[thing].vary]
ndim=len(variables)
#Vice versa, plus add in the fixed value
fixed=[ "{}={},".format(thing, theta[thing].value) for thing in theta if not theta[thing].vary]


#Optionally plot the fit with our initial guesses
SF.plot_fit(theta, fit.fit_settings)




###################################################################################################
#Set up the initial positions of the walkers as a ball with a different standard deviation in each dimension
nwalkers=70
nsteps=100


#Get the spread of the starting positions
stds=[]
n_general=9
n_positive=1
n_emission_lines=7

#Add in all these standard deviations
#Kinematic parameters
stds.extend([100.0, 50.0])
#General parameters
stds.extend([0.1]*n_general)
#Positive parameters
stds.extend([0.1]*n_positive)
#Emission lines
stds.extend([100.0, 50.0])
stds.extend([1.0]*n_emission_lines)
#Age
stds.extend([1.0])
#Z, imf1, imf2
stds.extend([0.1, 0.1, 0.1])
#Sky
#stds.extend([100.0,  100.0,  100.0, 100.0])
#ln_f
stds.extend([0.5])

stds=np.array(stds)

assert len(stds)==len(variables), "You must have the same number of dimensions for the Gaussian ball as variables!"

#Now get the starting values for each parameter, as well as the prior bounds
start_values, bounds=SF.get_start_vals_and_bounds(theta)
p0=SF.get_starting_positions_for_walkers(start_values, stds, nwalkers, bounds)
###################################################################################################


# ###################################################################################################
#Do the sampling
#This may take a while!

print("Running the fitting with {} walkers for {} steps".format(nwalkers, nsteps))
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[theta, variables, bounds], pool=None)
result = sampler.run_mcmc(p0, nsteps, progress=True)


# ####################################################################################################

# #get rid of the burn-in
# burnin=np.array(nsteps-5000).clip(0)
# samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
# print("\tDone")

# #Get the 16th, 50th and 84th percentiles of the marginalised posteriors for each parameter 
# best_results = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))))
# #If your posterior surface isn't a nice symmetric Gaussian, then the vector of median values for each parameter (as we're doing here)
# #could very well correspond to an unlikely area of parameter space and you'll need to do something different to this!


# for v, r in zip(variables, best_results):
#     print("{}: {:.3f} +{:.2f}/-{:.2f}".format(v, r[0], r[1], r[2]))

# #Make a set of parameters with the results
# results_theta=LM.Parameters()
# for v, r in zip(variables, best_results):
#     print(v, r)
#     results_theta.add('{}'.format(v), value=r[0], vary=False)
# #and include the things we kept fixed originally too:
# [results_theta.add('{}'.format(thing), value=theta[thing].value, vary=False) for thing in theta if not theta[thing].vary]

# #... and plot
# SF.plot_fit(results_theta, fit.fit_settings)

# ####################################################################################################

# #It's always a good idea to inspect the traces
# #Can also make corner plots, if you have corner available:
# #import corner
# #corner.corner(samples, labels=variables)
# #plt.savefig('corner_plot.pdf')
# #And you should inspect the residuals around the best fit as a function of wavelength


# ###################################################################################################