""" Code for pre-processing DESI data"""

''' Basic Recipe
0. Load the DESI mock spectrum
1. Resample to a constant dlambda/lambda dispersion
2. Renomalize the flux?
3. Generate a Sightline object with DLAs
4. Add labels 
5. Write to disk (numpy or TF)
'''

import numpy as np
from dla_cnn.spectra_utils import get_lam_data
from dla_cnn.data_model.DataMarker import Marker
from scipy.interpolate import interp1d
from os.path import join,exists
from os import remove
import csv
# Set defined items
#from dla_cnn.desi import defs
#REST_RANGE = defs.REST_RANGE
#kernel = defs.kernel


def label_sightline(sightline, kernel=400, REST_RANGE=[900,1316], pos_sample_kernel_percent=0.3):
    """
    Add labels to input sightline based on the DLAs along that sightline

    Parameters
    ----------
    sightline: dla_cnn.data_model.Sightline
    pos_sample_kernel_percent: float
    kernel: int
    REST_RANGE: list

    Returns
    -------
    classification: np.ndarray
        is 1 / 0 / -1 for DLA/nonDLA/border
    offsets_array: np.ndarray
        offset
    column_density: np.ndarray

    """
    lam, lam_rest, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, REST_RANGE)
    samplerangepx = int(kernel*pos_sample_kernel_percent/2) #60
    #kernelrangepx = int(kernel/2) #200
    ix_dlas=[]
    coldensity_dlas=[]
    for dla in sightline.dlas:
        if 912<(dla.central_wavelength/(1+sightline.z_qso))<1220:
            ix_dlas.append(np.abs(lam[ix_dla_range]-dla.central_wavelength).argmin()) 
            coldensity_dlas.append(dla.col_density)    # column densities matching ix_dlas

    '''
    # FLUXES - Produce a 1748x400 matrix of flux values
    fluxes_matrix = np.vstack(map(lambda f,r:f[r-kernelrangepx:r+kernelrangepx],
                                  zip(itertools.repeat(sightline.flux), np.nonzero(ix_dla_range)[0])))
    '''

    # CLASSIFICATION (1 = positive sample, 0 = negative sample, -1 = border sample not used
    # Start with all samples zero
    classification = np.zeros((np.sum(ix_dla_range)), dtype=np.float32)
    # overlay samples that are too close to a known DLA, write these for all DLAs before overlaying positive sample 1's
    for ix_dla in ix_dlas:
        classification[ix_dla-samplerangepx*2:ix_dla+samplerangepx*2+1] = -1
        # Mark out Ly-B areas
        lyb_ix = sightline.get_lyb_index(ix_dla)
        classification[lyb_ix-samplerangepx:lyb_ix+samplerangepx+1] = -1
    # mark out bad samples from custom defined markers
    #for marker in sightline.data_markers:
        #assert marker.marker_type == Marker.IGNORE_FEATURE              # we assume there are no other marker types for now
        #ixloc = np.abs(lam_rest - marker.lam_rest_location).argmin()
        #classification[ixloc-samplerangepx:ixloc+samplerangepx+1] = -1
    # overlay samples that are positive
    for ix_dla in ix_dlas:
        classification[ix_dla-samplerangepx:ix_dla+samplerangepx+1] = 1

    # OFFSETS & COLUMN DENSITY
    offsets_array = np.full([np.sum(ix_dla_range)], np.nan, dtype=np.float32)     # Start all NaN markers
    column_density = np.full([np.sum(ix_dla_range)], np.nan, dtype=np.float32)
    # Add DLAs, this loop will work from the DLA outward updating the offset values and not update it
    # if it would overwrite something set by another nearby DLA
    for i in range(int(samplerangepx+1)):
        for ix_dla,j in zip(ix_dlas,range(len(ix_dlas))):
            offsets_array[ix_dla+i] = -i if np.isnan(offsets_array[ix_dla+i]) else offsets_array[ix_dla+i]
            offsets_array[ix_dla-i] =  i if np.isnan(offsets_array[ix_dla-i]) else offsets_array[ix_dla-i]
            column_density[ix_dla+i] = coldensity_dlas[j] if np.isnan(column_density[ix_dla+i]) else column_density[ix_dla+i]
            column_density[ix_dla-i] = coldensity_dlas[j] if np.isnan(column_density[ix_dla-i]) else column_density[ix_dla-i]
    offsets_array = np.nan_to_num(offsets_array)
    column_density = np.nan_to_num(column_density)

    # Append these to the Sightline
    sightline.classification = classification
    sightline.offsets = offsets_array
    sightline.column_density = column_density

    # classification is 1 / 0 / -1 for DLA/nonDLA/border
    # offsets_array is offset
    return classification, offsets_array, column_density

def rebin(sightline, v):
    """
    Resample and rebin the input Sightline object's data to a constant dlambda/lambda dispersion.
    Parameters
    ----------
    sightline: :class:`dla_cnn.data_model.Sightline.Sightline`
    v: float, and np.log(1+v/c) is dlambda/lambda, its unit is m/s, c is the velocity of light
    Returns
    -------
    :class:`dla_cnn.data_model.Sightline.Sightline`:
    """
    # TODO -- Add inline comments
    c = 2.9979246e8
    dlnlambda = np.log(1+v/c)
    wavelength = 10**sightline.loglam
    max_wavelength = wavelength[-1]
    min_wavelength = wavelength[0]
    pixels_number = int(np.round(np.log(max_wavelength/min_wavelength)/dlnlambda))+1
    new_wavelength = wavelength[0]*np.exp(dlnlambda*np.arange(pixels_number))
    
    npix = len(wavelength)
    wvh = (wavelength + np.roll(wavelength, -1)) / 2.
    wvh[npix - 1] = wavelength[npix - 1] + \
                    (wavelength[npix - 1] - wavelength[npix - 2]) / 2.
    dwv = wvh - np.roll(wvh, 1)
    dwv[0] = 2 * (wvh[0] - wavelength[0])
    med_dwv = np.median(dwv)
    
    cumsum = np.cumsum(sightline.flux * dwv)
    cumvar = np.cumsum(sightline.error * dwv, dtype=np.float64)
    
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
    
    sightline.loglam = np.log10(new_wavelength[left:right])
    sightline.flux = new_fx[left:right]
    sightline.error = new_var[left:right]
    
    return sightline


def normalize(sightline, full_wavelength, full_flux):
    """
    Normalize this spectra using the lymann-forest part, using the median of the flux array with wavelength in rest frame between max(3800/(1+z_qso),1070) 
    and 1170. Normalize the error array at the same time to maintain the s/n. And for those spectrum cannot be normalzied, this function will assert error, when encounter this case.
    ---------------------------------------------------
    parameters:
    sightline: :class:`dla_cnn.data_model.sightline.Sightline` object, the spectrum to be normalized;
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
    sightline.flux = sightline.flux/np.median(full_flux[good_pix])
    # sightline.error = sightline.error / np.median(full_error[good_pix])
    sightline.error = sightline.error/np.median(full_flux[good_pix])

def estimate_s2n(sightline):
    """
    Estimate the s/n of a given sightline, using the lymann forest part and excluding dlas.
    -------------------------------------------------------------------------------------
    parameters；
    sightline: class:`dla_cnn.data_model.sightline.Sightline` object, we use it to estimate the s/n,
               and since we use the lymann forest part, the sightline's wavelength range should contain 1070~1170

    --------------------------------------------------------------------------------------
    return:
    s/n : float, the s/n of the given sightline.
    """
    #determine the lymann forest part of this sightline
    blue_limit = max(3800/(1+sightline.z_qso),1070)
    red_limit = 1170
    wavelength = 10**sightline.loglam
    rest_wavelength = wavelength/(sightline.z_qso+1)
    #lymann forest part of this sightline, contain dlas 
    test = (rest_wavelength>blue_limit)&(rest_wavelength<red_limit)
    #when excluding the part of dla, we remove the part between central_wavelength+-delta
    dwv = rest_wavelength[1]-rest_wavelength[0]
    dv = dwv/rest_wavelength[0] * 3e5  # km/s
    delta = int(np.round(3000./dv))
    for dla in sightline.dlas:
        test = test&((wavelength>dla.central_wavelength+delta)|(wavelength<dla.central_wavelength-delta))
    s2n = sightline.flux/sightline.error
    #return s/n
    return np.median(s2n[test])

