# A simple example

This example is contained in `fit_example.py`. It walks you through how to run a fit on an example spectrum located in `data/example_spectrum.txt`. This file has four columns; the wavelength of each value, the fluxes, the error and the instrumental resolution at that wavelength. It's made from the SSP models themselves, plus added Gaussian noise and emission line templates. 

##Setting Up
### Packages

You'll need the following packages to run this simple example, many of which should come as standard in the scientific python library. 

* `numpy`, `scipy` and `matplotlib`
* `lmfit` for its `Parameters` object
* A fitting code- we'll use `emcee`

### The data

We first read the datafile and get the relevant wavelength array, flux spectrum and error spectrum, as well as the instrumental resolution. 

```python
datafile='data/example_spectrum.txt'

lamdas, flux, errors, instrumental_resolution=np.genfromtxt(datafile, unpack=True)
```

You can optionally load a number of sky spectra here too. 

### Sections we want to ignore

We can give a list of two-component arrays with 'start' and 'stop' wavelengths around regions of the spectrum we don't want to fit. These might be areas of residual telluric absorption, particularly bad sky-lines, etc. These should be *observed* wavelengths, so can be read straight from the spectrum.

```python 
telluric_lam_1=np.array([[6862, 6952]])
telluric_lam_2=np.array([[7586, 7694]])
skylines=np.array([[8819, 8834], [8878.0, 8893], [8911, 8925], [8948, 8961]])
```

### Regions we want to fit between

During the fit, we have to compare the model and our data and use multiplicative Legendre polynomials to correct for small differences in continuum shape caused by the instrument or poor flux calibration. Ideally we'd fit the entire spectrum at once, but this tends to make finding these polynomials too slow. Here, we compromise by splitting our spectrum into four sections, with each one getting its own set of polynomials. These are then combined in the likelihood function. 

These wavelength are *rest frame* wavelengths. They'll be the values which we plot on the x-axis at the end. 

```python
fit_wavelengths=np.array([[4750, 5600], [5600, 6800], [6800, 8000], [8000,  9200]])
```

### Load the class

We can now load our SpectralFit class: 
```python
fit=SpectralFit(lamdas, flux, errors, pixel_weights, fit_wavelengths, FWHM_gal, instrumental_resolution=instrumental_resolution, skyspecs=skyspecs, element_imf=element_imf)
fit.set_up_fit()
```

which will read in all of our SSP templates, log-rebin them and get everything read to fit. 


## A simple fitting process with emcee

We now have our list of free parameters in the model. This is the `lmfit` Parameters object, which looks like: 

```python
theta=LM.Parameters()
#LOSVD parameters
theta.add('Vel', value=0.0, min=-1000.0, max=10000.0)
theta.add('sigma', value=300.0, min=10.0, max=500.0)

#Abundance of Na. Treat this separately, since it can vary up to +1.0 dex
theta.add('Na', value=0.5, min=-0.45, max=1.0, vary=True)

#Abundance of elements which can vary positively and negatively
theta.add('Ca', value=0.0,  min=-0.45, max=0.45, vary=True)
theta.add('Fe', value=0.0, min=-0.45, max=0.45, vary=True)
# and so on...
```
Each of these variables can be switched on and off using the `vary` keyword. The `min` and `max` keywords give the range of the (flat) prior. More details can be found at the `lmfit` documentation [here](https://lmfit-py.readthedocs.io/en/latest/parameters.html)

We now have a way to make templates at various model values and compare them directly to our data! Try changing a few values of `theta` and seeing what happens to the model. You can plot the fit at any time by calling `SF.plot_fit(theta, fit.fit_settings)`. 

Once we're happy with the variables we want to fit for, we select the starting positions of our walkers. Here we're just assuming a small ball around the initial parameter positions, but you should check that starting in different (or random) areas of parameter space gives you the same results! 

These functions make this ball, using a different standard deviation in each dimension: 

```python
#Now get the starting values for each parameter, as well as the prior bounds
start_values, bounds=SF.get_start_vals_and_bounds(theta)
p0=SF.get_starting_poitions_for_walkers(start_values, stds, nwalkers)
```

We can now run the fit using `emcee` (or any program of your choice: other MCMC samplers are available!):

```python
print("Running the fitting with {} walkers for {} steps".format(nwalkers, nsteps))
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[theta, variables, bounds], pool=None)
result = sampler.run_mcmc(p0, nsteps, progress=True)
```

The `emcee` documentation can be found [here](http://dfm.io/emcee/current/). 

## Results

Once our fit has finished (which may take a while!), we can get the samples, find various sample statistics of our posterior and plot the fit itself: 

```python
#get rid of the burn-in
burnin=nsteps-5000
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
print("\tDone")

#Get the 16th, 50th and 84th percentiles of the marginalised posteriors for each parameter 
best_results = np.array(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0))))

for v, r in zip(variables, best_results):
    print("{}: {:.3f} +{:.2f}/-{:.2f}".format(v, r[0], r[1], r[2]))
#Make a set of parameters with the results
results_theta=LM.Parameters()
for v, r in zip(variables, best_results):
    results_theta.add('{}'.format(v), value=r, vary=False)
#... and plot
SF.plot_fit(results_theta, fit.fit_settings)
```

It's always a good idea to look at a corner plot of your results (each parameter plotted against the others, as well as a one dimensional marginalised histogram). Finally, inspecting the residuals as a function of wavelength is also very important! A lot of issues with the fitting can be diagnosed this way, as well as getting a feeling for how reliable the results may be. 

As an aside, the v3.0.0dev version of `emcee` has some really nice features- such as incrementally saving your progress to a `h5` file and showing a built in progress bar. I'd highly recommend keeping an eye on it!