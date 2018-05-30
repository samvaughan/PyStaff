import os
import sys
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pystaff import SpectralFitting 
from pystaff import SpectralFitting_functs as SF
import numpy as np

class SpectralFitting_functs_isolatedfunctions_TestCase(unittest.TestCase):
    """Tests for `SpectralFitting.py`."""


    def test_starting_positions_for_walkers_return_shape(self):

        """Does the `get_starting_poitions_for_walkers` function return the correct shape?"""
        start_values=np.array([1.0, 2.0, 3.0])
        stds=np.array([1.0, 1.0, 10.0])
        nwalkers=100

        ball=SF.get_starting_poitions_for_walkers(start_values, stds, nwalkers)
        self.assertEqual(ball.shape, (len(start_values), nwalkers))

        ball=SF.get_starting_poitions_for_walkers(start_values, stds, nwalkers/2)
        self.assertNotEqual(ball.shape, (len(start_values), nwalkers))

    def test_randomness_of_starting_positions_for_walkers(self):

        """Does the `get_starting_poitions_for_walkers` function return random positions around the correct values?"""
        start_values=np.array([1.0, 2.0, 3.0])
        stds=np.array([1.0, 5.0, 10.0])
        #Big number here to avoid issues with randomness
        nwalkers=10000

        ball=SF.get_starting_poitions_for_walkers(start_values, stds, nwalkers)
        self.assertTrue(len(np.unique(ball))==ball.size)

        #We've set up starting values in order of size
        #so check this
        means=np.mean(ball, axis=1)
        spreads=np.std(ball, axis=1)
        self.assertTrue(np.allclose(np.round(np.abs(means-start_values)), 0))
        self.assertTrue(np.allclose(np.round(np.abs(spreads-stds)), 0))



    def test_make_mask_masks_correct_elements(self):
        """Does the `make_mask` function mask the correct element in a simple case?"""

        fiducial_mask=[ True,  True,  True,  True,  True, False, False, False,  True,  True]

        lamdas=np.arange(10)
        wavelengths=[[5, 7]]
        mask=SF.make_mask(lamdas, wavelengths)
        self.assertTrue(np.array_equal(mask, fiducial_mask))

        #And with two sets of wavelengths
        fiducial_mask=[ False,  False,  True,  True,  True, False, False, False,  True,  True]

        lamdas=np.arange(10)
        wavelengths=[[0, 1], [5, 7]]
        mask=SF.make_mask(lamdas, wavelengths)
        self.assertTrue(np.array_equal(mask, fiducial_mask))



    def test_contiguous_zeros_gives_correct_results(self):
        """Does the `contiguous_zeros` function give the correct results in a simple case?"""

        fiducial_array = np.array([[ 3,  9], [12, 16], [19, 20]])
        test_case = [1, 2, 3, 0, 0, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 9, 8, 7, 0, 10, 11]

        result = SF.contiguous_zeros(test_case)

        self.assertTrue(np.array_equal(result, fiducial_array))


    def test_fit_legendre_polynomials_cases(self):
        """Does the `_fit_legendre_polynomials` function give the correct results in simple cases?"""


        x=np.linspace(0.0, 1.0, 100)
        y=x**2
        polynomial=SF._fit_legendre_polys(y, 2, weights=None)
        self.assertTrue(np.allclose(np.abs(polynomial-y), 0.0))

        polynomial=SF._fit_legendre_polys(y, 2, weights=np.ones_like(y))
        self.assertTrue(np.allclose(np.abs(polynomial-y), 0.0))

        polynomial=SF._fit_legendre_polys(y, 1, weights=None)
        self.assertFalse(np.allclose(np.abs(polynomial-y), 0.0))





if __name__ == '__main__':
    unittest.main()







