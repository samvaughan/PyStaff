import os
import sys
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import SpectralFitting 
import SpectralFitting_functs
import numpy as np

class SpectralFittingTestCase(unittest.TestCase):
    """Tests for `SpectralFitting.py`."""


    def test_length_checking(self):

        """Do we raise an error for inputs of different lengths?"""
        #Dummy values
        lamdas=np.ones(1000)
        flux=np.ones(1000)
        noise=np.ones(1001)
        pixel_weights=np.ones(1000)
        fit_wavelengths=[[0, 100], [200, 300], [300, 400], [400, 500]]
        FWHM_gal=2.0

        with self.assertRaises(ValueError):
            fit=SpectralFitting.SpectralFit(lamdas, flux, noise, pixel_weights, fit_wavelengths, FWHM_gal, skyspecs=None, element_imf='kroupa', instrumental_resolution=None)

    def test_shape_checking(self):

        """Do we raise an error for inputs of different shapes?"""
        #Dummy values
        lamdas=np.ones(1000)
        flux=np.ones(1000)
        noise=np.ones(1000).reshape(-1, 1)
        pixel_weights=np.ones(1000)
        fit_wavelengths=[[0, 100], [200, 300], [300, 400], [400, 500]]
        FWHM_gal=2.0

        with self.assertRaises(ValueError):
            fit=SpectralFitting.SpectralFit(lamdas, flux, noise, pixel_weights, fit_wavelengths, FWHM_gal, skyspecs=None, element_imf='kroupa', instrumental_resolution=None)

    def test_instrumental_resolution_length_checking(self):

        """Do we raise an error for inputs of different shapes?"""
        #Dummy values
        lamdas=np.ones(1000)
        flux=np.ones(1000)
        noise=np.ones(1000)
        pixel_weights=np.ones(1000)
        fit_wavelengths=[[0, 100], [200, 300], [300, 400], [400, 500]]
        FWHM_gal=2.0

        instrumental_resolution=np.zeros(1001)

        with self.assertRaises(ValueError):
            fit=SpectralFitting.SpectralFit(lamdas, flux, noise, pixel_weights, fit_wavelengths, FWHM_gal, skyspecs=None, element_imf='kroupa', instrumental_resolution=instrumental_resolution)


    def test_check_for_gaps_in_lambda(self):
        """Do we raise an error for a gap in our wavelength array"""
        
        #Dummy values
        #Make a jump in our wavelength array
        lamdas=np.concatenate((np.arange(500), np.arange(500)+1000))
        flux=np.ones(1000)
        noise=np.ones(1000)
        pixel_weights=np.ones(1000)
        fit_wavelengths=[[0, 100], [200, 300], [300, 400], [400, 500]]
        FWHM_gal=2.0

        instrumental_resolution=np.zeros(1000)

        with self.assertRaises(ValueError):
            fit=SpectralFitting.SpectralFit(lamdas, flux, noise, pixel_weights, fit_wavelengths, FWHM_gal, skyspecs=None, element_imf='kroupa', instrumental_resolution=instrumental_resolution)




if __name__ == '__main__':
    unittest.main()