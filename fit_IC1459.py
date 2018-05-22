#! /usr/bin/env python

##############################################################################
import numpy as np


import emcee3
import os
import argparse

import scipy.interpolate as si 
from astropy.io import fits
from astropy.wcs import WCS

import lmfit_SPV as LMSPV
import SpectralFitting
import SpectralFitting_functs as SF

#from emcee.utils import MPIPool
from schwimmbad import MPIPool


print 'Starting!'


#Likelihood function here: saves pickling the parameters dictionary
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
        ll=SF.lnlike(theta, parameters)
        return ll
    else:
        return SF.lnlike(theta, parameters, ret_specs=True)


j=0
element_imf='kroupa'
imf_form='two-part'

####################################################
#Read in the data
K_datafile='~/z/Data/Laura_IC1459/IC1459_KMOS_MED_CORE_R1.fits'
M_datafile='~/z/Data/Laura_IC1459/IC1459_MUSE_MED_CORE_R1.fits'

K_data=fits.open(os.path.expanduser(K_datafile))
M_data=fits.open(os.path.expanduser(M_datafile))

K_header=K_data[1].header
M_header=M_data[1].header

K_WCS=WCS(K_header)
M_WCS=WCS(M_header)


M_lam=M_WCS.wcs_pix2world(np.arange(M_header['NAXIS1'])[:,np.newaxis], 0).flatten()*(10**10) #In angstroms
K_lam=K_WCS.wcs_pix2world(np.arange(K_header['NAXIS1'])[:,np.newaxis], 0).flatten()*(10**10)-4.0 #In angstroms, with correction for global offset between MUSE and KMOS (?)

M_spec=M_data[1].data
M_err=M_data[2].data
M_sky=M_data[3].data


K_spec=K_data[1].data
K_err=K_data[2].data


#Interpolate the KMOS data to be on the MUSE wavelength grid
new_K_lam=K_lam[0]+1.25*np.arange((K_lam[-1]-K_lam[0])/1.25)

interp_spec=si.interp1d(K_lam, K_spec, kind='cubic')
new_K_spec=interp_spec(new_K_lam)

interp_errors=si.interp1d(K_lam, K_err, kind='cubic')
new_K_err=interp_errors(new_K_lam)



flux=np.concatenate((M_spec, new_K_spec[new_K_lam>M_lam[-1]]))
errors=np.concatenate((M_err, new_K_err[new_K_lam>M_lam[-1]]))
lamdas=M_WCS.wcs_pix2world(np.arange(len(flux))[:,np.newaxis], 0).flatten()*(10**10) #In angstroms


######################################################
#Instrumental Resolution

#From my MUSE work on NGC 1399
#Get widths of isolated skylines
centres, dVs=np.genfromtxt('MUSE_sigma_inst.txt', unpack=True)
#CHANGE THE VALUES FROM SIGMA TO FWHM
#I think that they should be FWHM anyway, and I got it wrong in the NGC 1399 paper (as did Mentz et. al 2016 and Davor Krajnovic 2015!!!)
poly=np.polyfit(centres, dVs*np.sqrt(8*np.log(2)), 3)
f=np.poly1d(poly)
inst_res_MUSE=f(M_lam)

#Do the same for KMOS
#Here we're fitting a straight line because it gives the most 'conservative' fit
#Also, the ESO website only gives three values for the instrumental resolution, at the start, 
#stop and middle wavelengths, so it seems the spectral resolution doesn't vary too much
centres, dVs=np.genfromtxt('KMOS_sigma_inst.txt', unpack=True)
poly=np.polyfit(centres, dVs, 1)
f=np.poly1d(poly)
inst_res_KMOS=f(new_K_lam)

instrumental_resolution=np.concatenate((inst_res_MUSE, inst_res_KMOS[new_K_lam>M_lam[-1]]))
######################################################

#Mask out regions around telluric residuals and H-alpha.
#THESE SHOULD BE OBSERVED WAVELENGTHS
telluric_lam_1=np.array([[6862, 6952]])
telluric_lam_2=np.array([[7586, 7694]])
#Ha_lam=np.array([[6569, 6638]])

skylines=np.array([[8819, 8834], [8878.0, 8893], [8911, 8925], [8948, 8961]])
#skylines=np.array([[7735, 7745], [8404.0, 8414], [8782, 8791], [8841, 8852], [8873, 8887], [8735.0, 8744.0]])*(1+0.0043)
#Other skylines which we can probably subtract okay: [5546, 5556], [5857, 5875.0], [6268, 6276], [6332, 6340]

#masked_wavelengths=np.vstack([telluric_lam_1, telluric_lam_2, Ha_lam, skylines]).reshape(-1, 1, 2)
masked_wavelengths=np.vstack([telluric_lam_1, telluric_lam_2, skylines]).reshape(-1, 1, 2)
string_masked_wavelengths=["{} to {}".format(pair[0][0], pair[0][1]) for pair in masked_wavelengths]

#Mask pixels we don't want
pixel_mask=np.ones_like(flux, dtype=bool)
for array in masked_wavelengths:   
    m=SF.make_mask(lamdas, array)
    pixel_mask= m & pixel_mask

pixel_weights=np.ones_like(flux)
pixel_weights[~pixel_mask]=0.0


#Wavelengths we'll fit between.
#Split into 4 to make the multiplicative polynomials faster
fit_wavelengths=np.array([[4600, 5600], [5600, 6800], [6800, 8000], [8000,  9000], [9700, 10500]])
string_fit_wavelengths=["{} to {}".format(pair[0], pair[1]) for pair in fit_wavelengths]

#FWHM. Should make a way to measure this!
FWHM_gal=2.5

print 'Done all the preliminaries'

# #Set up the parameters
with MPIPool() as pool:
    if not pool.is_master():


        print 'Setting up the fit'
        fit=SpectralFitting.SpectralFit(lamdas, flux, errors, pixel_weights, fit_wavelengths, FWHM_gal, instrumental_resolution=instrumental_resolution, skyspecs=None, element_imf=element_imf)
        fit.set_up_fit()
        parameters=fit.fit_settings

        theta=LMSPV.Parameters()
        theta.add('Vel', value=1800.91, min=-1000.0, max=10000.0)
        theta.add('sigma', value=330.0, min=10.0, max=500.0)

        theta.add('Na', value=0.0, min=-0.45, max=1.0, vary=True)

        theta.add('Ca', value=0.0,  min=-0.45, max=0.45, vary=True)
        theta.add('Fe', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('C', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('N', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('Ti', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('Mg', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('Si', value=0.0, min=-0.45, max=0.45, vary=True)
        theta.add('Ba', value=0.0, min=-0.45, max=0.45, vary=True)


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


        theta.add('Vel_em', value=1800.91, min=0.0, max=10000)
        theta.add('sig_em', value=200.0, min=10.0, max=500.0)

        #These are log flux- they get exponentiated in the likelihood function
        theta.add('Ha', value=-1.5, min=-10.0, max=10.0)
        theta.add('Hb', value=-2.0, min=-10.0, max=10.0)
        theta.add('NII', value=-1.0, min=-10.0, max=10.0)
        theta.add('SII_6716', value=-2.0, min=-10.0, max=10.0)
        theta.add('SII_6731', value=-2.0, min=-10.0, max=10.0)
        theta.add('OIII', value=-2.0, min=-10.0, max=10.0)
        theta.add('OI', value=-2.0, min=-10.0, max=10.0)

        theta.add('age', value=10.0, min=1.0, max=14.0)
        theta.add('Z', value=0.0, min=-1.0, max=0.2)
        theta.add('imf_x1', value=2.35, min=0.5, max=3.5)
        theta.add('imf_x2', value=2.35, min=0.5, max=3.5) 

        theta.add('O2_Scale', value=-335.70307655300002, min=-100000000, max=100000000, vary=False) 
        theta.add('sky_Scale', value=988.58729658499999, min=-100000000, max=100000000, vary=False) 
        theta.add('OH_Scale', value=-132.30995387499999, min=-100000000, max=100000000, vary=False) 
        theta.add('NaD_sky_scale', value=-335.70307655300002, min=-100000000, max=100000000, vary=False)

        theta.add('ln_f', value=0.0, min=-5.0, max=5.0, vary=True)

        #Select the parameters we're varying, ignore the fixed ones
        variables=[thing for thing in theta if theta[thing].vary]
        ndim=len(variables)
        #Vice versa, plus add in the fixed value
        fixed=[ "{}={},".format(thing, theta[thing].value) for thing in theta if not theta[thing].vary]
        nwalkers=200
        nsteps=30000



        #Get the spread of the starting positions
        stds=[]
        n_general=9
        n_positive=1
        n_emission_lines=7

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




        start_values, bounds=SF.get_start_vals_and_bounds(theta)
        p0=SF.get_starting_poitions_for_walkers(start_values, stds, nwalkers)
        #CHeck everything is within the bounds
        #Make sure the positive parameters stay positive
        p0[2+n_general:2+n_general+n_positive, :]=np.abs(p0[2+n_general:2+n_general+n_positive, :])
        #This checks to see if any rows of the array have values which are too high, and replaces them with the upper bound value
        #Add the machine epsilon to deal with cases where we end up with, for example, one walker set to be -0.20000000000000001 instead of -0.2
        p0[p0<bounds[:, 0, None]]=bounds[np.any(p0<bounds[:, 0, None], axis=1), 0]+10*np.finfo(np.float64).eps
        #And the same for any values which are too low
        p0[p0>bounds[:, 1, None]]=bounds[np.any(p0>bounds[:, 1, None], axis=1), 1]-10*np.finfo(np.float64).eps

        assert np.all((p0>bounds[:, 0, None])&(p0<bounds[:, 1, None])), 'Some walkers are starting in bad places of parameter space!'


        fname='attempt1_centre.h5'
        backend=emcee3.backends.HDFBackend(fname)

        sampler = emcee3.EnsembleSampler(nwalkers, ndim, lnprob, args=[theta, variables, bounds], backend=backend, pool=None)
        result=sampler.run_mcmc(p0.T, nsteps, progress=True)

        #SF.lnlike(theta, fit.fit_settings)