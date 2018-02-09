import numpy as np

#from stellarpops.tools import fspTools as FT

import os
import argparse


from astropy.io import fits

import sys
sys.path.append('/home/vaughan/Science/SpectralAnalysis/SpectralFitting/NGC1399')
import MUSE_functs as MF


import lmfit_SPV as LMSPV


#from emcee.utils import MPIPool
from schwimmbad import MPIPool

import SpectralFitting
import SpectralFitting_functs as SF


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
        ll=MF.lnlike(theta, sf.fit_settings)
        return ll
    else:
        return MF.lnlike(theta, sf.fit_settings, ret_specs=True)


j=0
element_imf='kroupa'
imf_form='two-part'
# extend_chain=args.extend
# #Check our chain exists
# if extend_chain is not None:
#     assert os.path.isfile(extend_chain), "The chain at {} doesn't exist!".format(extend_chain)
#     print('Extending the chain at {}'.format(args.extend))


####################################################
#Read in the data
datafile='~/z/Data/Laura_IC1459/muse_core.fits'
#No error file yet
varfile='~/z/Data/Laura_IC1459/muse_core.fits'
skyfile=None

data=fits.open(os.path.expanduser(datafile))
vardata=fits.open(os.path.expanduser(varfile))


h1=data[0].header

lamdas = h1['CRVAL1'] + (np.arange(h1['NAXIS1']) - h1['CRPIX1'])*h1['CDELT1']

flux=data[0].data
errors=0.1*np.sqrt(vardata[0].data)


data.close()
vardata.close()
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
    m=MF.make_mask(lamdas, array)
    pixel_mask= m & pixel_mask

pixel_weights=np.ones_like(flux)
pixel_weights[~pixel_mask]=0.0


#Wavelengths we'll fit between.
#Split into 4 to make the multiplicative polynomials faster
fit_wavelengths=np.array([[4600, 5600], [5600, 6800], [6800, 8000], [8000,  9000]])
string_fit_wavelengths=["{} to {}".format(pair[0], pair[1]) for pair in fit_wavelengths]

#FWHM. Should make a way to measure this!
FWHM_gal=2.5

# #Set up the parameters
fit=SpectralFitting.SpectralFit(lamdas, flux, errors, pixel_weights, fit_wavelengths, FWHM_gal, skyspecs=None, element_imf=element_imf)
fit.set_up_fit()






theta=LMSPV.Parameters()
theta.add('Vel', value=1.27341185e+03, min=-1000.0, max=10000.0)
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


theta.add('Vel_em', value=1.27341185e+03, min=0.0, max=10000)
theta.add('sig_em', value=200.0, min=10.0, max=500.0)

#These are log flux- they get exponentiated in the likelihood function
theta.add('Ha', value=0.0, min=-10.0, max=10.0)
theta.add('Hb', value=-2.0, min=-10.0, max=10.0)
theta.add('NII', value=-2.0, min=-10.0, max=10.0)
theta.add('SII_6716', value=-2.0, min=-10.0, max=10.0)
theta.add('SII_6731', value=-2.0, min=-10.0, max=10.0)
theta.add('OIII', value=-2.0, min=-10.0, max=10.0)
theta.add('OI', value=-2.0, min=-10.0, max=10.0)

theta.add('age', value=10.0, min=1.0, max=14.0)
theta.add('Z', value=0.0, min=-1.0, max=0.2)
theta.add('imf_x1', value=2.35, min=0.5, max=3.5)
theta.add('imf_x2', value=2.35, min=0.5, max=3.5) 

theta.add('O2_Scale', value=-335.70307655300002, min=-100000000, max=100000000, vary=True) 
theta.add('sky_Scale', value=988.58729658499999, min=-100000000, max=100000000, vary=True) 
theta.add('OH_Scale', value=-132.30995387499999, min=-100000000, max=100000000, vary=True) 
theta.add('NaD_sky_scale', value=-335.70307655300002, min=-100000000, max=100000000, vary=True)

theta.add('ln_f', value=2.0, min=-5.0, max=5.0, vary=True)

#Select the parameters we're varying, ignore the fixed ones
variables=[thing for thing in theta if theta[thing].vary]
ndim=len(variables)
#Vice versa, plus add in the fixed value
fixed=[ "{}={},".format(thing, theta[thing].value) for thing in theta if not theta[thing].vary]
nwalkers=200
nsteps=30000



SF.lnlike(theta, fit.fit_settings)