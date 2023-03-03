from dla_cnn.desi.DesiMock import DesiMock
import numpy as np
from scipy.interpolate import interp1d
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
import json

# read the frequently-used wavelength
wlh = json.load(open('wavelength.json', 'r'))

def get_spilt_point(data:DesiMock, id):
    '''
    Get the spilt point in the overlapping area. Before this point we reserve the spectrum from the last camera, and after this point we reserve the ones from the next camera.
    
    -----
    ### Parameters:
    `data` and `id`: the original dataset from which the sightline is extracted by `DesiMock().get_sightline()` and the id of this sightline.
    '''
    line_b = data.get_sightline(id=id, camera='b')
    line_z = data.get_sightline(id=id, camera='z')
    line_r = data.get_sightline(id=id, camera='r')
    spilt_loglam_br = np.average([np.max(line_b.loglam), np.min(line_r.loglam)])
    spilt_loglam_rz = np.average([np.min(line_z.loglam), np.max(line_r.loglam)])
    return spilt_loglam_br, spilt_loglam_rz

def get_between(array, max, min, maxif=False, minif=False):
    '''
    Get the indices of a part of `array` whose value is between `max` and `min`.
    
    ------
    ### Parameters:
    `array`: the array which we will process the above procedure on.
    `max` and `min`: the upper and lower limit for the new child array.
    `maxif` and `minif`: whether the values EQUAL to `max` and `min` can be included in the new child array.
    '''
    if max >= min:
        if max >= np.min(array) and min <= np.max(array):
            if maxif:
                if minif:
                    return np.intersect1d(np.where(array>=min)[0], np.where(array<=max)[0])
                else:
                    return np.intersect1d(np.where(array>min)[0], np.where(array<=max)[0])
            else:
                if minif:
                    return np.intersect1d(np.where(array>=min)[0], np.where(array<max)[0])
                else:
                    return np.intersect1d(np.where(array>min)[0], np.where(array<max)[0])
        else:
            raise ValueError('min~max out of range')
    else:
        raise ValueError('max < min, will return nothing')
    
def W(N):
    '''
    Calculate the equivalent width W of DLA.
    
    ------
    
    ### Parameters:
    `N`: the column density of the DLA. (in the form of 10^x)
    '''
    return 10**(0.5*(N-24.440751700479186))


def wing_correction(z_dla, nhi_dla, wav, dla_indicator):
    '''
    Correct the wing of the DLA absorption.
    
    -----
    
    ### Parameters:
    `z_dla`: the redshift of the DLA.
    `nhi_dla`: the column density of the DLA. (in the form of 10^x)
    `wav`: the observed wavelength of the sightline.
    `dla_indicator`(Bool array): the indicator along the wavelenght axis. It is `True` if there is a DLA at this wavelength, and `False` if not.
    '''
    w = W(nhi_dla)
    wav_dla = wav/(1+z_dla)
    dwav = wav/(1+z_dla) - 1215.67
    correction = np.exp(w**2 / (4*np.pi) * (wav_dla * dla_indicator / dwav)**2)
    correction[np.isnan(correction)] = 0.
    return correction

def overlap(sightline, data:DesiMock, id):
    '''
    Deal with the overlapping area between different cameras so that the result will not contain the overlaps.
    
    ---
    ### Parameters
    `sightline`: the spectra that is waiting to be rebinned. It must have been clipped by `clip()`.
    `data` and `id`: the original dataset from which the sightline is extracted by `DesiMock().get_sightline()` and the id of this sightline.
    '''
    spilt_loglam_br, spilt_loglam_rz = get_spilt_point(data, id)
    line_r = data.get_sightline(id=id, camera='r')
    line_b = data.get_sightline(id=id, camera='b')
    line_z = data.get_sightline(id=id, camera='z')
    
    loglam_r = line_r.loglam[0:np.where(line_r.loglam == np.max(line_r.loglam))[0][0]]
    indice_r = get_between(loglam_r, max=spilt_loglam_rz, min=spilt_loglam_br)
    indice_b = get_between(line_b.loglam, max=spilt_loglam_br, min=0, maxif=True)
    indice_z = get_between(line_z.loglam, max=np.Infinity, min=spilt_loglam_rz, minif=True)
    
    loglam_r, loglam_b, loglam_z = loglam_r[indice_r], line_b.loglam[indice_b], line_z.loglam[indice_z]
    flux_r, flux_b, flux_z = line_r.flux[indice_r], line_b.flux[indice_b], line_z.flux[indice_z]
    error_r, error_b, error_z = line_r.error[indice_r], line_b.error[indice_b], line_z.error[indice_z]
    sightline.loglam = np.concatenate((loglam_b, loglam_r, loglam_z))
    sightline.flux = np.concatenate((flux_b, flux_r, flux_z))
    sightline.error = np.concatenate((error_b, error_r, error_z))
    
def normalize(sightline, full_wavelength, full_flux):
    """
    Normalize this spectra using the lymann-forest part, using the median of the flux array with wavelength in rest frame between max(3800/(1+z_qso),1070) 
    and 1170. Normalize the error array at the same time to maintain the s/n. And for those spectrum cannot be normalzied, this function will assert error, when encounter this case.
    ---------------------------------------------------
    parameters:
    sightline: :class:`dla_cnn.data_model.sightline.Sightline` object, the spectrum to be normalized;(It must have been processed by `overlap()`)
    full_wavelength: numpy.ndarray, the whole wavelength array of this sightline, since the sightline may not contain the blue channel,
                we pass the wavelength array to this function
    full_flux:numpy.ndarray,the whole flux wavelength array of this sightline, take it as a parameter to solve the same problem above.
    """
    # determine the blue limit and red limit of the slice we use to normalize this spectra, and when cannot find such a slice, this function will assert error
    blue_limit = max(3800/(1+sightline.z_qso),1070)
    red_limit = 1170
    rest_wavelength = full_wavelength/(sightline.z_qso+1)
    assert blue_limit <= red_limit,"No Lymann-alpha forest, Please check this spectra: %i"%sightline.id#when no lymann alpha forest exists, assert error.
    #use the slice we chose above to normalize this spectra, normalize both flux and error array using the same factor to maintain the s/n.
    good_pix = (rest_wavelength>=blue_limit)&(rest_wavelength<=red_limit)
    norm_factor = np.median(full_flux[good_pix])
    sightline.flux_norm = sightline.flux/norm_factor
    # sightline.error = sightline.error / np.median(full_error[good_pix])
    sightline.error_norm = sightline.error/norm_factor
    return norm_factor      
        
def clip(sightline, unit_default=100, slope=2e-3, ratio=0.5):
    '''
    Clip the abnormal points in the spectra.
    
    ---
    
    ### Parameters
    `sightline`: the spectra that is waiting to be cliped. It must have been normalized by `normalize()`.
    `unit_default`: the default length of each bin that is used to conduct sigmaclip.
    `slope`: the critical value that decides whether a smaller bin will be applied. If the fit slope of current bin exceeds this value, a smaller bin will be used.
    `ratio`: how small the smaller bin will be compared with the default bin length.
    `plot`: if true, this function can generate a plot that shows every bin's clipping upper and lower limit as well as the original spectrum that has not been clipped, which can clearly show which points are clipped. Ofter used in jupyter notebook. 
    '''
    
    wavs = 10**sightline.loglam
    flux = sightline.flux_norm
    error = sightline.error_norm
    zero_point = np.where(wavs / (1+sightline.z_qso) >= wlh['LyALPHA'])[0][0]
    sightline.points_num = len(wavs)

    wavs_new = wavs[0:zero_point]
    flux_new = flux[0:zero_point]
    error_new = error[0:zero_point]
        
    unit = unit_default
    judge, start, end = True, zero_point, zero_point + unit
    while judge:

        if end >= len(wavs):
            end = len(wavs) - 1
            judge = False
            if start == end:
                break
        subwavs, subflux, suberror = wavs[start:end], flux[start:end], error[start:end]
        if end - start >= 3:
            slope_fit = curve_fit(lambda t,a,b: a*t+b, subwavs, subflux)[0][0]

            if np.abs(slope_fit) >= slope:
                unit = int(unit_default*ratio)
                end = start + unit
                if end >= len(wavs):
                    end = len(wavs) - 1
                    judge = False
                    if start == end:
                        break
                subwavs, subflux, suberror = wavs[start:end], flux[start:end], error[start:end]

            elif np.abs(slope_fit) < slope and unit != unit_default:
                unit = unit_default
                end = start + unit
                if end >= len(wavs):
                    end = len(wavs) - 1
                    judge = False
                    if start == end:
                        break
                subwavs, subflux, suberror = wavs[start:end], flux[start:end], error[start:end]

            mask = np.invert(sigma_clip(subflux, sigma=3).mask)
            flux_cliped = subflux[mask]
            wavs_cliped = subwavs[mask]
            error_cliped = suberror[mask]
        else:
            flux_cliped, wavs_cliped, error_cliped = subflux, subwavs, suberror
        wavs_new = np.concatenate((wavs_new, wavs_cliped))
        flux_new = np.concatenate((flux_new, flux_cliped))
        error_new = np.concatenate((error_new, error_cliped))
        start = start +  unit
        end = end + unit
        
    sightline.loglam_cliped = np.log10(wavs_new)
    sightline.flux_cliped = flux_new
    sightline.error_cliped = error_new
    
def get_dlnlambda(sightline):
    '''
    Generate the step length of restframe grid used in `rebin()`.
    
    ----

    ### Attention
    For the mock data, this function generate the same value for all the spectrum. I am not sure whether this characristic will remain the same for the actual data.
    '''
    wavelength = 10**sightline.loglam_cliped
    pixels_number = sightline.points_num
    max_wavelength = wavelength[-1]
    min_wavelength = wavelength[0]
    dlnlambda = np.log(max_wavelength/min_wavelength)/pixels_number
    return dlnlambda
    
def rebin(sightline, loglam_start, dlnlambda, max_index:int=int(1e6)):
    '''
    Rebin to the same restframe grid.
    
    --------

    ### Parameters:
    `sightline`: the spectra that is waiting to be rebinned. It must have been clipped by `clip()`.
    `loglam_start`: the start point of this restframe grid. Usually it is the start RESTFRAME wavelength of the spectra whose redshift is the largest.
    `dlnlambda`: the step length of this restframe grid. It can be derived with `get_dlnlambda()`.
    `max_index`: because different spectra has different range of wavelength in restframe, so it is necessary to make the restframe grid large enough to contain all of these spectrum. This parameter is the size of this grid, which is usually very big. You can change the default value if you think it is too big.
    '''
    
    wavelength = 10**sightline.loglam_cliped / (1+sightline.z_qso)
    flux = sightline.flux_cliped
    error = sightline.error_cliped
    
    max_wavelength = wavelength[-1]
    min_wavelength = wavelength[0]
    new_wavelength_total = 10**loglam_start * np.exp(dlnlambda * np.arange(max_index))
    indices = get_between(new_wavelength_total, max_wavelength, min_wavelength, maxif=True, minif=True)
    new_wavelength = new_wavelength_total[indices]
    
    # below is the code from dla_cnn.data_model.Spectrum.rebin()
    npix = len(wavelength)
    wvh = (wavelength + np.roll(wavelength, -1)) / 2.
    wvh[npix - 1] = wavelength[npix - 1] + \
                    (wavelength[npix - 1] - wavelength[npix - 2]) / 2.
    dwv = wvh - np.roll(wvh, 1)
    dwv[0] = 2 * (wvh[0] - wavelength[0])
    med_dwv = np.median(dwv)
    
    cumsum = np.cumsum(flux * dwv)
    cumvar = np.cumsum(error * dwv, dtype=np.float64)
    
    fcum = interp1d(wvh, cumsum,bounds_error=False)
    fvar = interp1d(wvh, cumvar,bounds_error=False)
    
    nnew = len(new_wavelength)
    nwvh = (new_wavelength + np.roll(new_wavelength, -1)) / 2.
    nwvh[nnew - 1] = new_wavelength[nnew - 1] + \
                     (new_wavelength[nnew - 1] - new_wavelength[nnew - 2]) / 2.
    
    bwv = np.zeros(nnew + 1)
    bwv[0] = new_wavelength[0] - (new_wavelength[1] - new_wavelength[0]) / 2.
    bwv[1:] = nwvh
    
    newcum = fcum(bwv)
    newvar = fvar(bwv)
    
    new_fx = (np.roll(newcum, -1) - newcum)[:-1]
    new_var = (np.roll(newvar, -1) - newvar)[:-1]
    
    # Normalize (preserve counts and flambda)
    new_dwv = bwv - np.roll(bwv, 1)
    new_fx = new_fx / new_dwv[1:]
    # Preserve S/N (crudely)
    med_newdwv = np.median(new_dwv)
    new_var = new_var / (med_newdwv/med_dwv) / new_dwv[1:]
    
    left = 0
    while np.isnan(new_fx[left])|np.isnan(new_var[left]):
        left = left+1
    right = len(new_fx)
    while np.isnan(new_fx[right-1])|np.isnan(new_var[right-1]):
        right = right-1
    
    test = np.sum((np.isnan(new_fx[left:right]))|(np.isnan(new_var[left:right])))
    assert test==0, 'Missing value in this spectra!'
    
    sightline.loglam_rebin_restframe = np.log10(new_wavelength[left:right])
    sightline.flux_rebin_restframe = new_fx[left:right]
    sightline.error_rebin_restframe = new_var[left:right]
    
def mask_dla(sightline):
    '''
    Mask the DLA region.
    
    ----
    
    ### Parameters:
    `sightline`: the spectra that is waiting to be masked. It must have been rebinned by `rebin()`.
    '''
    interp_flux = sightline.flux_rebin_restframe
    interp_error = sightline.error_rebin_restframe
    lamda = 10**sightline.loglam_rebin_restframe
    z = sightline.z_qso
    dla_indicator = np.ones_like(interp_flux, dtype=bool)
    dla_correction = np.ones_like(interp_flux)
    if len(sightline.dlas) > 0:
        for dla in sightline.dlas:
            z_dla = (dla.central_wavelength / wlh['LyALPHA']) - 1
            nhi = dla.col_density
            w = W(nhi)
            dla_indicator *= ~((lamda*(1+z) < 1215.67 * (1+z_dla) * (1.+w) ) & (lamda*(1+z)>1215.67*(1+z_dla)*(1.-w)))
            dla_correction *= wing_correction(z_dla, nhi, lamda*(1+z), dla_indicator)
        sightline.loglam_mask = sightline.loglam_rebin_restframe[dla_indicator]
        sightline.flux_mask = interp_flux[dla_indicator] * dla_correction[dla_indicator]
        sightline.error_mask = interp_error[dla_indicator] * dla_correction[dla_indicator]
    else:
        sightline.loglam_mask = sightline.loglam_rebin_restframe
        sightline.flux_mask = interp_flux
        sightline.error_mask = interp_error
