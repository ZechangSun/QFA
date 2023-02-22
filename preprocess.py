from dla_cnn.desi.DesiMock import DesiMock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import interp1d
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit

LyALPHA = 1215.6701
LyBETA = 1025.7220
MgII1 = 1482.890
MgII2 = 1737.628
CIV1 = 1550
CIV2 = 1910

def overlap(sightline, data:DesiMock, id):
    '''
    here sightline = data.get_sightline(id=id)
    '''
    
    
    def get_spilt_point(data:DesiMock, id):
        line_b = data.get_sightline(id=id, camera='b')
        line_z = data.get_sightline(id=id, camera='z')
        line_r = data.get_sightline(id=id, camera='r')
        spilt_loglam_br = np.average([np.max(line_b.loglam), np.min(line_r.loglam)])
        spilt_loglam_rz = np.average([np.min(line_z.loglam), np.max(line_r.loglam)])
        return spilt_loglam_br, spilt_loglam_rz

    def get_between(array, max, min, maxif=False, minif=False):
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
    
    
# def clip(sightline, unit, plot=False):
#     wavs = 10**sightline.loglam
#     flux = sightline.flux
#     zero_point = np.where(wavs / (1+sightline.z_qso) >= LyALPHA)[0][0]
#     i = 0

#     wavs_new = wavs[0:zero_point]
#     flux_new = flux[0:zero_point]
    
#     if plot:
#         sigmaup, sigmadown = np.zeros(zero_point), np.zeros(zero_point)
        
#     judge = True
#     while judge:
#         start = zero_point + i * unit
#         end = zero_point + (i+1) * unit
#         if end >= len(wavs):
#             end = len(wavs) - 1
#             judge = False
#             if start == end:
#                 break
#         subwavs = wavs[start:end]
#         subflux = flux[start:end]
#         mask = np.invert(sigma_clip(subflux, sigma=3).mask)
#         flux_cliped = subflux[mask]
#         wavs_cliped = subwavs[mask]
#         wavs_new = np.concatenate((wavs_new, wavs_cliped))
#         flux_new = np.concatenate((flux_new, flux_cliped))
#         if plot:
#             sigma = np.std(subflux)
#             mean = np.average(subflux)
#             sigmaup = np.concatenate((sigmaup, np.ones_like(wavs_cliped)*(mean+3*sigma)))
#             sigmadown = np.concatenate((sigmadown, np.ones_like(wavs_cliped)*(mean-3*sigma)))
#         i = i + 1
        
#     sightline.loglam_cliped = np.log10(wavs_new)
#     sightline.flux_cliped = flux_new
    
#     if plot:
#         plt.plot(wavs, flux)
#         plt.plot(wavs_new[zero_point:], sigmaup[zero_point:])
#         plt.plot(wavs_new[zero_point:], sigmadown[zero_point:])
#         plt.axvline(LyALPHA*(1+sightline.z_qso), linestyle='--')
    
def clip(sightline, unit_default=100, slope=2e-3, plot=False):
    '''
    This function can automatically change the unit according to the slope of the being-clipped area
    '''
    
    def line_fit(xdata, ydata):
    
        def linear(x, *args):
            return args[0] * x + args[1]
        
        popt, pcov = curve_fit(f=linear, xdata=xdata, ydata=ydata, p0=(1e-3, 0))
        return popt
    
    wavs = 10**sightline.loglam
    flux = sightline.flux
    error = sightline.error
    zero_point = np.where(wavs / (1+sightline.z_qso) >= LyALPHA)[0][0]

    wavs_new = wavs[0:zero_point]
    flux_new = flux[0:zero_point]
    error_new = error[0:zero_point]
    
    if plot:
        sigmaup, sigmadown = np.zeros(zero_point), np.zeros(zero_point)
        
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
            slope_fit = line_fit(subwavs, subflux)[0]

            if np.abs(slope_fit) >= slope:
                unit = int(unit_default/2)
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
        if plot:
            sigma = np.std(subflux)
            mean = np.average(subflux)
            sigmaup = np.concatenate((sigmaup, np.ones_like(wavs_cliped)*(mean+3*sigma)))
            sigmadown = np.concatenate((sigmadown, np.ones_like(wavs_cliped)*(mean-3*sigma)))
        
    sightline.loglam_cliped = np.log10(wavs_new)
    sightline.flux_cliped = flux_new
    sightline.error_cliped = error_new
    
    if plot:
        plt.plot(wavs, flux)
        plt.plot(wavs_new[zero_point:], sigmaup[zero_point:])
        plt.plot(wavs_new[zero_point:], sigmadown[zero_point:])
        plt.axvline(LyALPHA*(1+sightline.z_qso), linestyle='--')
        plt.show()
    