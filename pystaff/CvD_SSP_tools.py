import numpy as np
import scipy.constants as const
from astropy.io import fits
import scipy.interpolate as si
import glob
from ppxf import ppxf_util as P
import glob
import os


class Spectrum(object):
    """

    UPDATE THE DOCSTRING! A stripped down version of specTools in order to package this all together nicely
    Original Author: Ryan Houghton (20/4/11)
    Updated: Sam Vaughan (Early 2018)

    Purpose: A spectrum object to aid manipulation of wavelength/frequency and the associated
             flux values

    Inputs:
    This class MUST be initalised with EITHER
       lamspec - a NLAMxNSPEC numpy array with [:,0] being the wavelength array (AA) and [:,1:] being
                 the associated flux values (erg/s/cm**2/AA)
       muspec  - a NLAMxNSPEC numpy array with [:,0] being the frequency array (GHz) and [:,1:] being
                 the associated flux values (GJy = 1e-14 erg/s/cm**2/Hz)

    NOTE: 2xN seems a funny way to order the indicies but in numpy, this IS the efficient
    way to order the memory elements

    Definitions:
    Once you have initalised the class with either lamspec or muspec, you will find available
    the following variables (no matter if you initalised with lamspec or muspec):
       lam  - the wavelength array (AA)
       flam - the associated flux (erg/s/cm**2/AA)
       mu   - the frequency array (GHz)
       fmu  - the associated flux (GJy = 1e-14 erg/s/cm**2/Hz)

    Functions:
       calcABmag  - given a filter (spectrum class), return a magnitude on the AB system
       calcSTmag  - as above but for the ST system
       calcVEGAmag- as above but for the VEGA system; you must also supply a vega *spectrum*

    Notes:
       - You can define more than one flam/fmu for each lam/mu
       - Filters used in the functions should also be *spectrum* classes
       - Filters don't need to be sorted in mu or lam
    """

    global c, pc, d_sun

    def __init__(self, lam=None, lamspec=None, errlamspec=None, age=None, Z=None, wavesyst=None, userdict=None):

        """
        Inputs:
        -------
           lam     - the 1D array of wavelengths (becomes self.lam)
           lamspec - the (1D, 2D or 3D) array of fluxes on the specified lambda grid (becomes self.flam)
           errlamspec - standard deviation for pixels in lamspec (becomes self.eflam)
           age     - the 1D array of ages (Gyr) for the spectra
           Z       - the 1D array of metallicities for the spectra
           wavesyst - you MUST specify if the wavelengths are in the AIR or VAC system.
           userdict- add extra user-defined tags to the spectrum, such as airmass, water vapour etc.

        Notes:
        ------
        There are 3 ways to inialise a spectrum class:
            1. With a 1D lam/mu and a 1D spectrum (single spec mode).
                 - Magnitudes are returned as single values
            2. With a 1D lam/mu and a 2D spectrum array (NSPECxN{LAM/MU}) (multispec mode)
                 - In this case, you may specify AGE where len(age)=NSPEC.
                 - Magnitudes are returned as 1D arrays with NSPEC=len(age) elements
            #3. With a 1D lam/mu and a 3D spectrum array (NZxNAGExN{LAM/MU}) (multispec mode)
            #     - In this case, you can specify AGE and Z.
            #     - Magnitudes will be returned as 2D arrays with NZxNAGE elements
        """

        # start defining spectrum parts
        if lamspec is not None:
            # check that lam has been given
            if lam is None:
                raise Exception("If you give lamspec, you must also give lam")

            # make sure 2d
            flam = np.atleast_2d(lamspec)
            # get array size
            loc = flam.shape

            # check for bigger arrays
            # if len(loc)> 2: raise "lamspec not understood"

            # get sizes
            nlam = loc[1]
            nspec = loc[0]

            self.lam = lam
            self.flam = np.atleast_2d(singleOrList2Array(flam))

            if errlamspec is not None:
                eflam = np.atleast_2d(errlamspec)
                eloc = eflam.shape
                self.eflam = eflam
                # sanity check
                assert np.all(loc == eloc), "Flux and error arrays appear different sizes..."
            else:
                self.eflam = None

        # add age info
        if age is not None:
            self.age = singleOrList2Array(age)

            checkDims(self.age, "Age", self.flam.shape[:-1])
            self.logage = np.log10(self.age)
        else:
            self.age = None

        # add metallicitiy
        if Z is not None:

            self.Z = singleOrList2Array(Z)
            checkDims(self.Z, "Z", self.flam.shape[:-1])
        else:
            self.Z = None

        # add VAC or AIR
        if wavesyst is not None:
            if (wavesyst == "vac" or wavesyst == "VAC" or wavesyst == "Vac"):
                self.wavesyst = "vac"
            elif (wavesyst == "air" or wavesyst == "AIR" or wavesyst == "Air"):
                self.wavesyst = "air"
            else:
                raise ValueError("wavesyst not understood. Should be air or vac.")
        else:
            warn.warn("You failed to specify if the wavelength is defined in AIR or VAC units.")
            self.wavesyst = None

        # add user dictionary for extra info
        if userdict is not None:
            self.__userdict__ = userdict
            keys = list(userdict.keys())
            for key in keys:
                setattr(self, key, singleOrList2Array(userdict[key]))
        else:
            self.__userdict__ = None


def checkDims(var, varname, parentShape):
    assert (np.isscalar(var)) or (np.all(np.equal(np.array(var).shape, parentShape))), varname + " dimensions not understood."


def singleOrList2Array(invar):
    """
    If a single value, leave. If a list, convert to array. If array, leave as array.
    But for size-1 arrays/lits, convert back to scalar.
    """

    if isinstance(invar, list):
        # convert to array unless size-1
        rval = np.squeeze(np.array(invar))
        if rval.size == 1:
            rval = np.asscalar(rval)
    elif isinstance(invar, np.ndarray):
        # leave except if size-1
        rval = np.squeeze(invar)
        if rval.size == 1:
            rval = np.asscalar(rval)
    else:
        # leave
        rval = invar
    # return
    return rval

def load_varelem_CvD16ssps(varelem_template_location, imf='kroupa', verbose=True):

    '''
    Load the CvD16 spectra (response functions) with variable elemental abundances.

    Arguments:
        dirname (string): Base location of the stellar models directory
        folder (string): The folder inside the base directory which contains the response functions
        imf (string): The IMF of the response functions. Choices are 'kroupa' or 'salpeter'
        verbose (bool): Print information to the console
    Returns:
        (dict): A dictionary of spectra. See below for example

    Example:
        The spectra are returned in a dictionary. The dictionary is indexed by a string referring to the name of the respone function,
        e.g. 'Na+'. At each index, we have a Spectrum class which contains the response functions at various ages and metallicities

        >>> spectra=load_varelem_CvD16ssps(dirname='/path/to/dir', folder='folder', imf='kroupa')
        >>> spectra['Na+'] #Gives all RFs with [Na/H]=+0.3

    '''

    varelem_template_location = os.path.expanduser(varelem_template_location)

    if imf in ['kroupa', 'krpa', 'Kroupa', 'Krpa']:
        model_spectra = sorted(glob.glob('{}/atlas_ssp_*.krpa.s100'.format(varelem_template_location)))
        imf_name = 'krpa'
    elif imf in ['Salpeter', 'salpeter', 'salp', 'Salp']:
        model_spectra = sorted(glob.glob('{}/atlas_ssp_*.salp.s100'.format(varelem_template_location)))
        imf_name = 'salp'
    else:
        raise NameError('IMF type not understood')

    data = np.genfromtxt(model_spectra[0])
    lams = data[:, 0]

    model_Zs_names = ['m1.5', 'm1.0', 'm0.5', 'p0.0', 'p0.2']
    model_age_names = ['01', '03', '05', '09', '13']

    model_elem_order = ['Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', 'C+', 'C-', 'a/Fe+', 'N+', 'N-', 'as/Fe+', 'Ti+', 'Ti-', 'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+', 'V+', 'Cu+', 'Na+0.6', 'Na+0.9']

    Zs = [-1.5, -1.0, -0.5, 0.0, 0.2]
    ages = [float(a) for a in model_age_names]

    n_ages = len(model_age_names)
    n_zs = len(model_Zs_names)
    n_elems = len(model_elem_order)

    templates = np.empty((n_elems, n_ages, n_zs, len(lams)))

    for a, Z in enumerate(model_Zs_names):
        for b, age in enumerate(model_age_names):

            model = glob.glob('{}/atlas_ssp*t{}*{}*{}.s100'.format(varelem_template_location, age, Z, imf_name))[0]
            if verbose:
                print('Loading {}'.format(model))
            data = np.genfromtxt(model)

            for i, elem in enumerate(model_elem_order):
                templates[i, b, a, :] = data[:, i + 1]

    spectra = {}
    for i, elem in enumerate(model_elem_order):

        age_values = np.repeat(ages, n_zs).reshape(n_ages, n_zs)
        Z_values = np.repeat(Zs, n_ages).reshape(n_zs, n_ages).T

        spectra[elem] = Spectrum(lam=lams, lamspec=templates[i, :, :, :], age=age_values, Z=Z_values, wavesyst='vac', userdict={'elem': elem})

    return spectra


################################################################################################################################################################
def prepare_CvD_interpolator_twopartIMF(base_template_location, templates_lam_range, velscale, verbose=True, instrumental_resolution=None):
    """
    Set up the interpolator for the base SSP spectra using the log-rebinned templates we get from `prepare_CvD2_templates_twopartIMF`.

    Arguments:
        templates_lam_range (array-like): A 2 component vector with the start and stop wavelengths we want for the templates
        velscale (float): The velocity difference between two adjacent pixels in the log-rebinned spectrum. Use the same value
        for the templates as you *measure* from the spectrum!
        verbose (bool): Print information to the consolse

    Returns:
        (tuple): A two component tuple containing:
            * interp: the interpolate object. Axes are wavelength, age, Z, imf_x1 and imf_x2
            * logLam_template: the log-rebinned wavelength array of the templates
    """
    templates, logLam_template = prepare_CvD2_templates_twopartIMF(base_template_location, templates_lam_range, velscale, verbose=verbose, instrumental_resolution=instrumental_resolution)

    nimfs = 16
    ages = [1., 3., 5., 7., 9., 11.0, 13.5]
    Zs = [-1.5, -1.0, -0.5, 0.0, 0.2]
    n_imfs = 16
    imfs_X1 = 0.5 + np.arange(n_imfs) / 5.0
    imfs_X2 = 0.5 + np.arange(n_imfs) / 5.0

    linear_interp = si.RegularGridInterpolator(((logLam_template, ages, Zs, imfs_X1, imfs_X2)), templates, bounds_error=False, fill_value=None)

    return linear_interp, logLam_template

################################################################################################################################################################

def prepare_CvD2_templates_twopartIMF(template_location, templates_lam_range, velscale, verbose=True, instrumental_resolution=None):

    '''
    Load the CvD16 base spectra, those that vary age, [Z/H] and the IMF. The templates are log-rebinned to have a uniform wavelength
    spacing in **log** wavelength, to ensure the difference in velocity between two adjacent pixels is the same, regardless of wavelength. 

    They are returned in an array of shape (n_ages, n_Zs, n_pixels).

    Arguments: 
        templates_lam_range (array-like): A 2 component vector with the start and stop wavelengths we want for the templates
        velscale (float): The velocity difference between two adjacent pixels in the log-rebinned spectrum. Use the same value
        for the templates as you *measure* from the spectrum!
        verbose (bool): Print information to the consolse
    Returns:
        (tuple): A two component tuple:
            * templates (array): An array of templates with shape (N_wavelength, N_ages, N_metallicities, N_imf_x1, N_imf_x2)
            * logLam_template (array): The log-rebinned wavelength array of the templates
    '''
    
    template_glob=os.path.expanduser(f'{template_location}/VCJ_v8_mcut0.08_t*')

    vcj_models=sorted(glob.glob(template_glob))
    models=np.genfromtxt(vcj_models[-1])

    temp_lamdas=models[:, 0]

    n_ages=7
    n_zs=5
    n_imfs=16

    


    Zs=['m1.5', 'm1.0', 'm0.5', 'p0.0', 'p0.2']
    ages=['01.0', '03.0', '05.0', '07.0', '09.0', '11.0', '13.5']
    imfs_X1=0.5+np.arange(n_imfs)/5.0
    imfs_X2=0.5+np.arange(n_imfs)/5.0

    t_mask = ((temp_lamdas > templates_lam_range[0]) & (temp_lamdas <templates_lam_range[1]))



    y=models[t_mask, 1]
    x=temp_lamdas[t_mask]
    #Make a new lamda array, carrying on the delta lamdas of high resolution bit
    new_x=temp_lamdas[t_mask][0]+0.9*(np.arange(np.ceil((temp_lamdas[t_mask][-1]-temp_lamdas[t_mask][0])/0.9))+1)
    interp=si.interp1d(x, y, fill_value='extrapolate')
    out=interp(new_x)

    sspNew, logLam_template, template_velscale = P.log_rebin(templates_lam_range, out, velscale=velscale)
    templates=np.empty((len(sspNew), n_ages, n_zs, len(imfs_X1), len(imfs_X2)))

    #Resolution of the templates in km/s

    for a, Z in enumerate(Zs):    
        for b, age in enumerate(ages):
            model=glob.glob(os.path.expanduser(f'{template_location}/VCJ_v8_mcut0.08_t{age}*{Z}.ssp.imf_varydoublex.s100'))[0]
            print('Loading {}'.format(model))
            data=np.genfromtxt(model)

            for c, counter1 in enumerate(imfs_X1):
                for d, counter2 in enumerate(imfs_X2):
                
                    #Interpolate templates onto a uniform wavelength grid and then log-rebin
                    y=data[:, c*n_imfs+d+1][t_mask]   
                    x=temp_lamdas[t_mask]
                    
                    #Make a new lamda array, carrying on the delta lamdas of high resolution bit
                    new_x=temp_lamdas[t_mask][0]+0.9*(np.arange(np.ceil((temp_lamdas[t_mask][-1]-temp_lamdas[t_mask][0])/0.9))+1)

                    interp=si.interp1d(x, y, fill_value='extrapolate')
                    out=interp(new_x)

                    #log rebin them
                    sspNew, logLam_template, template_velscale = P.log_rebin(templates_lam_range, out, velscale=velscale)
                    if instrumental_resolution is not None:
                        sspNew = P.gaussian_filter1d(sspNew, instrumental_resolution/velscale)

                    templates[:, b, a, c, d]=sspNew#/np.median(sspNew)

    return templates, logLam_template

################################################################################################################################################################


################################################################################################################################################################
def prepare_CvD_correction_interpolators(varelem_template_location, templates_lam_range, velscale, elements, verbose=True, element_imf='kroupa', instrumental_resolution=None):

    """
    Set up the interpolator for the response functions using the log-rebinned templates we get from `prepare_CvD2_element_templates`.
    The response functions allow for variation in a general set of elements (Fe, Ca, Mg, etc), a set of elements which can only be 
    positive (Mn, K, V, etc), and Sodium, which can be extended from -0.3 dex to +1.0 dex. We have to treat each of these cases 
    separately, so we end up with three different interpolation objects. You can also vary the effective temperature of the isochrone, 
    but this is now deprecated in V2. 

    Arguments:
        templates_lam_range (array-like): A 2 component vector with the start and stop wavelengths we want for the templates
        velscale (float): The velocity difference between two adjacent pixels in the log-rebinned spectrum. Use the same value
        for the templates as you *measure* from the spectrum!
        elements (array): This is an array of integers corresponding to the elements we want to vary in the interpolators. The
        numbers come from the headings in the SSP RF file. Na is element 1, Ca is element 2, etc. TODO: Improve this! Make it clearer

        verbose (bool): Print information to the consolse

    Returns:
        (tuple): A two component tuple containing:
            * correction_interps: A tuple of the interpolators. The order is [general_interp, na_interp, positive_only_interp, T_interp). TODO explain this more
            * logLam_template: The log-rebinned wavelength array of the templates
    """

    all_corrections, logLam_template=prepare_CvD2_element_templates(varelem_template_location, templates_lam_range, velscale, elements, verbose=verbose, element_imf=element_imf, instrumental_resolution=instrumental_resolution)

    general_templates, na_templates, carbon_templates, positive_only_templates, T_templates=all_corrections

    positive_only_elems, Na_elem, Carbon_elem, normal_elems=elements


    #It's not clear if the last value here should be 13.5 or 13
    ages=np.array([  1.,   3.,   5.,   9.,  13.0])
    Zs=[-1.5, -1.0, -0.5, 0.0, 0.2]


    elem_steps=[-0.45, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.45]
    Na_elem_steps=[-0.45, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    C_elem_steps = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]
    positive_only_elem_steps=[0.0, 0.1, 0.2, 0.3, 0.45]
    T_steps=[-50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    
    na_interp=si.RegularGridInterpolator(((Na_elem_steps, ages, Zs, logLam_template)), na_templates, bounds_error=False, fill_value=None, method='linear')

    carbon_interp = si.RegularGridInterpolator(((C_elem_steps, ages, Zs, logLam_template)), carbon_templates, bounds_error=False, fill_value=None, method='linear')

    T_interp=si.RegularGridInterpolator(((T_steps, ages, Zs, logLam_template)), T_templates, bounds_error=False, fill_value=None, method='linear')

    #If we only have one positive element to check, we need to do something different- can't have a dimension with only one element in RegularGridInterpolator apparently.
    if len(positive_only_elems)>1:
        positive_only_interp=si.RegularGridInterpolator(((np.arange(len(positive_only_elems)), positive_only_elem_steps, ages, Zs, logLam_template)), positive_only_templates, bounds_error=False, fill_value=None, method='linear')
    else:
        positive_only_interp=si.RegularGridInterpolator((positive_only_elem_steps, ages, Zs, logLam_template), positive_only_templates[0, :], bounds_error=False, fill_value=None, method='linear')

    if len(normal_elems)>1:
        general_interp=si.RegularGridInterpolator(((np.arange(len(normal_elems)), elem_steps, ages, Zs, logLam_template)), general_templates, bounds_error=False, fill_value=None, method='linear')
    else:
        general_interp=si.RegularGridInterpolator((elem_steps, ages, Zs, logLam_template), general_templates[0, :], bounds_error=False, fill_value=None, method='linear')
    

    correction_interps=[general_interp, na_interp, carbon_interp, positive_only_interp, T_interp]

    return correction_interps, logLam_template

##########

def new_wavelength_array_high_resolution(wavelengths):
    """
    The templates have different spacings in wavelength between the blue and red ends. This function makes a new set of wavelength values which ensures the same wavelength spacing throughout 
    """

    new_wavelength_array = wavelengths[0]+0.9*(np.arange(np.ceil((wavelengths[-1]-wavelengths[0])/0.9))+1)

    return new_wavelength_array

def prepare_CvD2_element_templates(varelem_template_location, templates_lam_range, velscale, elements, verbose=True, element_imf='kroupa', instrumental_resolution=None):

    var_elem_spectra=load_varelem_CvD16ssps(varelem_template_location, imf=element_imf)

    ages=var_elem_spectra['Solar'].age[:, 0]
    Zs=var_elem_spectra['Solar'].Z[0, :]
    n_ages=len(ages)
    n_Zs=len(Zs)

    temp_lamdas=var_elem_spectra['Solar'].lam

    t_mask = ((temp_lamdas > templates_lam_range[0]) & (temp_lamdas <templates_lam_range[1]))


    positive_only_elems, Na_elem, Carbon_elem, normal_elems=elements

    elem_steps=[-0.45, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.45]
    Na_elem_steps=[-0.45, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    C_elem_steps = [-0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2]
    positive_only_elem_steps=[0.0, 0.1, 0.2, 0.3, 0.45]
    T_steps=[-50.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0]

    x=var_elem_spectra['Solar'].lam[t_mask]
    y=var_elem_spectra['Solar'].flam[-1, -1, t_mask]
    #Make a new lamda array, carrying on the delta lamdas of high resolution bit
    new_x = new_wavelength_array_high_resolution(var_elem_spectra['Solar'].lam[t_mask])
    interp=si.interp1d(x, y, fill_value='extrapolate')
    data=interp(new_x)


    sspNew, logLam_template, template_velscale = P.log_rebin(templates_lam_range, data, velscale=velscale)


    positive_only_templates = np.empty((len(positive_only_elems), len(positive_only_elem_steps), n_ages, n_Zs, len(sspNew)))
    general_templates = np.empty((len(normal_elems), len(elem_steps), n_ages, n_Zs, len(sspNew)))
    
    na_templates = np.empty((len(Na_elem_steps), n_ages, n_Zs, len(sspNew)))
    carbon_templates = np.empty((len(C_elem_steps), n_ages, n_Zs, len(sspNew)))
    T_templates = np.empty((len(T_steps), n_ages, n_Zs, len(sspNew)))

    print('Making the Positive-Only Correction templates')
    #Do the positve only correction templates:
    for a, elem in enumerate(positive_only_elems):
        print('\t{}'.format(elem))
        for b, step in enumerate(positive_only_elem_steps):
            for c, _ in enumerate(ages):
                for d, _ in enumerate(Zs):

                    if step !=0.0:
                        y=(var_elem_spectra[elem].flam[c, d, t_mask]/var_elem_spectra['Solar'].flam[c, d, t_mask] - 1.0)*((10**(step)-1.0)/(10**(0.3)-1.0))


                    else:
                        y=np.zeros_like(var_elem_spectra['Solar'].flam[c, d, t_mask])


                    x=var_elem_spectra[elem].lam[t_mask]
                    #Make a new lamda array, carrying on the delta lamdas of high resolution bit
                    new_x=new_wavelength_array_high_resolution(x)
                    interp=si.interp1d(x, y, fill_value='extrapolate')
                    data=interp(new_x)
                            
                    sspNew, logLam_template, template_velscale = P.log_rebin(templates_lam_range, data, velscale=velscale)
                    if instrumental_resolution is not None:
                        sspNew = P.gaussian_filter1d(sspNew, instrumental_resolution/velscale)

                    positive_only_templates[a, b, c, d, :]=sspNew


    print('Making the General Correction templates')
    #Do the general templates
    for a, elem in enumerate(normal_elems):
        
        print('\t{}'.format(elem))
        for b, step in enumerate(elem_steps):
            for c, _ in enumerate(ages):
                for d, _ in enumerate(Zs):

                    if step>0.0:
                        e='{}+'.format(elem)
                        gen_step=step
                    elif step<0.0:
                        e='{}-'.format(elem)
                        gen_step=np.abs(step)

                    if step !=0.0:
                        y=(var_elem_spectra[e].flam[c, d, t_mask]/var_elem_spectra['Solar'].flam[c, d, t_mask]-1)*((10**(gen_step)-1.0)/(10**(0.3)-1.0))
                    else:
                        y=np.zeros_like(var_elem_spectra['Solar'].flam[c, d, t_mask])

                    x=var_elem_spectra[e].lam[t_mask]
                    #Make a new lamda array, carrying on the delta lamdas of high resolution bit
                    new_x=new_wavelength_array_high_resolution(x)
                    interp=si.interp1d(x, y, fill_value='extrapolate')
                    data=interp(new_x)

                    #data=util.gaussian_filter1d(data, sigs/velscale)

                    sspNew, logLam_template, template_velscale = P.log_rebin(templates_lam_range, data, velscale=velscale)
                    if instrumental_resolution is not None:
                        sspNew = P.gaussian_filter1d(sspNew, instrumental_resolution/velscale)

                    general_templates[a, b, c, d, :]=sspNew

    # Make the Carbon templates        
    # Treat Carbon differently because it's templates are at +0.15 and -0.15 dex rather than +0.3 and -0.3 dex
    print('Making the Carbon Correction template')
    for b, step in enumerate(C_elem_steps):
        for c, _ in enumerate(ages):
            for d, _ in enumerate(Zs):

                if step>0.0:
                    e='{}+'.format('C')
                    gen_step=step
                elif step<0.0:
                    e='{}-'.format('C')
                    gen_step=np.abs(step)

                if step !=0.0:
                    y=(var_elem_spectra[e].flam[c, d, t_mask]/var_elem_spectra['Solar'].flam[c, d, t_mask]-1)*((10**(gen_step)-1.0)/(10**(0.15)-1.0))
                else:
                    y=np.zeros_like(var_elem_spectra['Solar'].flam[c, d, t_mask])

                x=var_elem_spectra[e].lam[t_mask]
                #Make a new lamda array, carrying on the delta lamdas of high resolution bit
                new_x=new_wavelength_array_high_resolution(x)
                interp=si.interp1d(x, y, fill_value='extrapolate')
                data=interp(new_x)

                #data=util.gaussian_filter1d(data, sigs/velscale)

                sspNew, logLam_template, template_velscale = P.log_rebin(templates_lam_range, data, velscale=velscale)
                if instrumental_resolution is not None:
                        sspNew = P.gaussian_filter1d(sspNew, instrumental_resolution/velscale)

                carbon_templates[b, c, d, :]=sspNew

    #Do the Na templates:
    print('Making the Na Correction template')
    for a, step in enumerate(Na_elem_steps):
        for b, _ in enumerate(ages):
            for c, _ in enumerate(Zs):

                if step <0.0:
                    e='Na-'
                    base_enhancement=0.3
                    Na_step=np.abs(step)                    
                elif 0.0<=step<0.45:
                    e='Na+'
                    base_enhancement=0.3
                    Na_step=step
                elif 0.45<=step<0.75:
                    e='Na+0.6'
                    base_enhancement=0.6
                    Na_step=step
                elif 0.75<=step<1.0:
                    e='Na+0.9'
                    base_enhancement=0.9
                    Na_step=step
                
                if step !=0.0:
                    y=(var_elem_spectra[e].flam[b, c, t_mask]/var_elem_spectra['Solar'].flam[b, c, t_mask]-1)*((10**(Na_step)-1.0)/(10**(base_enhancement)-1.0))

                else:

                    y=np.zeros_like(var_elem_spectra['Solar'].flam[b, c, t_mask])


                x=var_elem_spectra[e].lam[t_mask]
                #Make a new lamda array, carrying on the delta lamdas of high resolution bit
                new_x=new_wavelength_array_high_resolution(x)
                interp=si.interp1d(x, y, fill_value='extrapolate')
                data=interp(new_x)

                    
                sspNew, logLam_template, template_velscale = P.log_rebin(templates_lam_range, data, velscale=velscale)
                if instrumental_resolution is not None:
                        sspNew = P.gaussian_filter1d(sspNew, instrumental_resolution/velscale)

                na_templates[a, b, c, :]=sspNew


    print('Making the Temperature Correction template')
    for a, step in enumerate(T_steps):
        for b, _ in enumerate(ages):
            for c, _ in enumerate(Zs):

                if step>0.0:
                    e='T+'
                    T_step=step
                elif step<0.0:
                    e='T-'
                    T_step=np.abs(step)
            
                if step !=0.0:
                    y=(var_elem_spectra[e].flam[b, c, t_mask]/var_elem_spectra['Solar'].flam[b, c, t_mask]-1)*(T_step/50.0)

                else:

                    y=np.zeros_like(var_elem_spectra['Solar'].flam[b, c, t_mask])


                x=var_elem_spectra[e].lam[t_mask]
                #Make a new lamda array, carrying on the delta lamdas of high resolution bit
                new_x=new_wavelength_array_high_resolution(x)
                interp=si.interp1d(x, y, fill_value='extrapolate')
                data=interp(new_x)

                    
                sspNew, logLam_template, template_velscale = P.log_rebin(templates_lam_range, data, velscale=velscale)
                if instrumental_resolution is not None:
                        sspNew = P.gaussian_filter1d(sspNew, instrumental_resolution/velscale)

                T_templates[a, b, c, :]=sspNew

    return [general_templates, na_templates, carbon_templates, positive_only_templates, T_templates], logLam_template
