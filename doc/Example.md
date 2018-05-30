# A simple example

This example is contained in `fit_example.py`

### Packages

You'll need the following packages to run this simple example, most of which should come as standard in the scientific python library. 

* `numpy`, `scipy` and `matplotlib`
* `lmfit` for its `Parameters` object
* A fitting code- we'll use `emcee`

### Load in our data

We first read the datafile and get the relevant wavelength array, flux spectrum and error spectrum. TODO: Include a datafile with this example

```python
M_datafile='/path/to/spectrum.fits'

M_data=fits.open(os.path.expanduser(M_datafile))
M_header=M_data[1].header

M_WCS=WCS(M_header)

lambdas=M_WCS.wcs_pix2world(np.arange(M_header['NAXIS1'])[:,np.newaxis], 0).flatten()*(10**10) #In angstroms

flux=M_data[1].data
errors=M_data[2].data
```

You can optionally load a number of sky spectra here, and an array of values for the instrumental resolution as a function of wavelength. 

### Mask out any sections we want to ignore

We can give a list of two-component arrays with 'start' and 'stop' wavelengths around regions of the spectrum we don't want to fit. These might be areas of residual telluric absorption, particularly bad sky-lines, etc. These should be *observed* wavelengths, so can be read straight from the spectrum.

```python 
telluric_lam_1=np.array([[6862, 6952]])
telluric_lam_2=np.array([[7586, 7694]])
skylines=np.array([[8819, 8834], [8878.0, 8893], [8911, 8925], [8948, 8961]])
```

### Select the regions we want to fit between

During the fit, we have to compare the model and our data and use multiplicative Legendre polynomials to correct for small differences in continuum shape caused by the instrument or poor flux calibration. Ideally we'd fit the entire spectrum at once, but this tends to make finding these polynomials too slow. Here, we compromise by splitting our spectrum into four sections, with each one getting its own set of polynomials. These are then combined in the likelihood function. 

These wavelength are *rest frame* wavelengths. They'll be the values which we plot on the x-axis at the end. 

```python
fit_wavelengths=np.array([[4600, 5600], [5600, 6800], [6800, 8000], [8000,  9000], [9700, 10500]])
```


TODO: Sort out what FWML_gal is 

### Load the class

We can now load our SpectralFit class: 
```python
fit=SpectralFit(lamdas, flux, errors, pixel_weights, fit_wavelengths, FWHM_gal, instrumental_resolution=instrumental_resolution, skyspecs=skyspecs, element_imf=element_imf)
fit.set_up_fit()
```



