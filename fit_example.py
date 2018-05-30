#! /usr/bin/env python

##############################################################################
import numpy as np
import emcee3
import os
import sys
import scipy.interpolate as si 
from astropy.io import fits
from astropy.wcs import WCS
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
M_datafile='/path/to/spectrum.fits'

M_data=fits.open(os.path.expanduser(M_datafile))
M_header=M_data[1].header

M_WCS=WCS(M_header)

lambdas=M_WCS.wcs_pix2world(np.arange(M_header['NAXIS1'])[:,np.newaxis], 0).flatten()*(10**10) #In angstroms

flux=M_data[1].data
errors=M_data[2].data

# ######################################################
# Sky Spectra
# Give a list of 1D sky spectra to be scaled and subtracted during the fit
# Otherwise leave sky as None
sky=None
# ######################################################

# ######################################################
# Instrumental Resolution
# The instrumental resolution can be included if it's known. We need a value of sigma_inst in km/s for every pixel
# Otherwise leave it as None
instrumental_resolution=None
# ######################################################



# Mask out regions that we don't want to fit, e.g. around telluric residuals, particularly nasty skylines, etc
# THESE SHOULD BE OBSERVED WAVELENGTHS
telluric_lam_1=np.array([[6862, 6952]])
telluric_lam_2=np.array([[7586, 7694]])
skylines=np.array([[8819, 8834], [8878.0, 8893], [8911, 8925], [8948, 8961]])

masked_wavelengths=np.vstack([telluric_lam_1, telluric_lam_2, skylines]).reshape(-1, 1, 2)
string_masked_wavelengths=["{} to {}".format(pair[0][0], pair[0][1]) for pair in masked_wavelengths]

#Mask pixels we don't want
pixel_mask=np.ones_like(flux, dtype=bool)
for array in masked_wavelengths:   
    m=SF.make_mask(lamdas, array)
    pixel_mask= m & pixel_mask

#Now switch the weights of these pixels to 0
pixel_weights=np.ones_like(flux)
pixel_weights[~pixel_mask]=0.0


#Wavelengths we'll fit between.
#Split into 4 to make the multiplicative polynomials faster
fit_wavelengths=np.array([[4600, 5600], [5600, 6800], [6800, 8000], [8000,  9000], [9700, 10500]])
string_fit_wavelengths=["{} to {}".format(pair[0], pair[1]) for pair in fit_wavelengths]

#FWHM. Should make a way to measure this!
FWHM_gal=2.5

#Now set up the spectral fitting class
print 'Setting up the fit'
fit=SpectralFit(lamdas, flux, errors, pixel_weights, fit_wavelengths, FWHM_gal, instrumental_resolution=instrumental_resolution, skyspecs=skyspecs, element_imf=element_imf)
fit.set_up_fit()

# # #Set up the parameters
# with MPIPool() as pool:
#     if not pool.is_master():
#         pool.wait()
#         sys.exit(0)



#Here are the available fit parameters
#They can easily be switched off by changing vary to False
#The min and max values act as flat priors
theta=LM.Parameters()
#LOSVD parameters
theta.add('Vel', value=1800.91, min=-1000.0, max=10000.0)
theta.add('sigma', value=330.0, min=10.0, max=500.0)

#Abundance of Na. Treat this separately, since it can vary up to +1.0 dex
theta.add('Na', value=0.0, min=-0.45, max=1.0, vary=True)

#Abundance of elements which can vary positively and negatively
theta.add('Ca', value=0.0,  min=-0.45, max=0.45, vary=True)
theta.add('Fe', value=0.0, min=-0.45, max=0.45, vary=True)
theta.add('C', value=0.0, min=-0.45, max=0.45, vary=True)
theta.add('N', value=0.0, min=-0.45, max=0.45, vary=True)
theta.add('Ti', value=0.0, min=-0.45, max=0.45, vary=True)
theta.add('Mg', value=0.0, min=-0.45, max=0.45, vary=True)
theta.add('Si', value=0.0, min=-0.45, max=0.45, vary=True)
theta.add('Ba', value=0.0, min=-0.45, max=0.45, vary=True)

#Abundance of elements which can only vary aboce 0.0
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
theta.add('Vel_em', value=1800.91, min=0.0, max=10000)
theta.add('sig_em', value=200.0, min=10.0, max=500.0)

#Emission line strengths
#These are log flux- they get exponentiated in the likelihood function
theta.add('Ha', value=-1.5, min=-10.0, max=10.0)
theta.add('Hb', value=-2.0, min=-10.0, max=10.0)
theta.add('NII', value=-1.0, min=-10.0, max=10.0)
theta.add('SII_6716', value=-2.0, min=-10.0, max=10.0)
theta.add('SII_6731', value=-2.0, min=-10.0, max=10.0)
theta.add('OIII', value=-2.0, min=-10.0, max=10.0)
theta.add('OI', value=-2.0, min=-10.0, max=10.0)

#Base population parameters
#Age, Metallicity, and the two IMF slopes
theta.add('age', value=10.0, min=1.0, max=14.0)
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
#Set up the fit
#Set up the initial positions of the walkers as a ball with a different standard deviation in each dimension

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

#Not get the starting values for each parameter, as well as the prior bounds
start_values, bounds=SF.get_start_vals_and_bounds(theta)
p0=SF.get_starting_poitions_for_walkers(start_values, stds, nwalkers)
#Check everything is within the bounds
#Make sure the positive parameters stay positive
p0[2+n_general:2+n_general+n_positive, :]=np.abs(p0[2+n_general:2+n_general+n_positive, :])
#This checks to see if any rows of the array have values which are too high, and replaces them with the upper bound value
#Add the machine epsilon to deal with cases where we end up with, for example, one walker set to be -0.20000000000000001 instead of -0.2
p0[p0<bounds[:, 0, None]]=bounds[np.any(p0<bounds[:, 0, None], axis=1), 0]+10*np.finfo(np.float64).eps
#And the same for any values which are too low
p0[p0>bounds[:, 1, None]]=bounds[np.any(p0>bounds[:, 1, None], axis=1), 1]-10*np.finfo(np.float64).eps

assert np.all((p0>bounds[:, 0, None])&(p0<bounds[:, 1, None])), 'Some walkers are starting in bad places of parameter space!'
p0=p0.T
###################################################################################################


###################################################################################################
#Do the sampling
#This may take a while!
nwalkers=200
nsteps=30000

# fname='attempt1_centre.h5'
# backend=emcee3.backends.HDFBackend(fname)

# sampler = emcee3.EnsembleSampler(nwalkers, ndim, lnprob, args=[theta, variables, bounds], backend=backend, pool=None)
# result = sampler.run_mcmc(p0, nsteps, progress=True)


###################################################################################################

#Once we're done, plot the results
chisq, chisq_per_dof, (fig, ax)=SF.plot_fit(theta, SF.plot_fit(theta, fit.fit_settings))

#Get the autocorrelation time and discard the burnin
tau = backend.get_autocorr_time(tol=0)
burnin = int(2*np.max(tau))
print("\tDiscarding burn-in: {} steps".format(burnin))
samples = backend.get_chain(discard=burnin, flat=False)
print("\tDone")
n_params=samples.shape[-1]

best_results = np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0))))

for v, r in zip(variables, best_results):
    print "{}: {:.3f} +{:.2f}/-{:.2f}".format(v, r[0], r[1], r[2])

###################################################################################################

#It's always a good idea to inspect the traces
#Can also make corner plots, if you have corner available:
#import corner
#corner.corner(samples, labels=variables)
#plt.savefig('corner_plot.pdf')
#And you should check the residuals around the best fit as a function of wavelength

###################################################################################################