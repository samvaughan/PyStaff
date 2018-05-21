import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits
import scipy.constants as const

from stellarpops.tools import miscTools as misc


#Load the data- a sky spectrum taken from one of the KCLASH cubes
lamdas, skyspec=np.genfromtxt('/Data/KCLASH/Data/Sci/skyspec.txt', unpack=True)
lamdas*=10**4 #in angstroms
#Get the list of isolated MUSE skylines
lines=misc.get_KMOS_skylines()

sigmas=np.empty(lines.shape[0])
FWHMS=np.empty(lines.shape[0])

plt.plot(lamdas, skyspec)
for i, line in enumerate(lines):
    sigmas[i]=misc.fit_gaussian_to_skyline(lamdas, skyspec, line, plot=True)
    FWHMS[i]=sigmas[i]*np.sqrt(8*np.log(2))

centres=np.mean(lines, axis=1)

#delta V/C = delta lam/lambda
dVs=FWHMS/centres*(const.c/1000.0)

np.savetxt('KMOS_sigma_inst.txt', np.column_stack([centres, dVs]))