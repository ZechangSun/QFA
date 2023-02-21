import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dla_cnn.desi.DesiMock import DesiMock

# Prepare the wavelengths of some important emission lines
# Wavelengths here may be WRONG. Please check them before using.
LyALPHA = 1215.6701
LyBETA = 1025.7220
MgII1 = 1482.890
MgII2 = 1737.628
CIV1 = 1550
CIV2 = 1910
lams = np.array([LyBETA, LyALPHA, MgII1, CIV1, MgII2, CIV2])
names = ['LyBETA', 'LyALPHA', 'MgII1', 'CIV1', 'MgII2', 'CIV2']
lines = {}
for i, name in enumerate(names):
    lines[name] = lams[i]
    
# prepare for the data path
def generate_suffix(prefix):
    suffix = {}
    for preid in os.listdir(prefix):
        suffix[preid] = os.listdir(prefix+preid)
    return suffix


def generate_seperated_catalog(prefix):

    # prefix = './desi-0.2-100/spectra-16/' # this need to be specialized
    suffix = generate_suffix(prefix=prefix)
    # generate a catalog (csv format) under each folder

    data = {}
    for suffix1 in tqdm(suffix.keys()):
        for suffix2 in tqdm(suffix[suffix1]):
            path = prefix + suffix1 + '/' + suffix2 + '/'
            if len(os.listdir(path)) == 3:
                path_spectra = path + 'spectra-16-' + suffix2 +'.fits'
                path_truth = path + 'truth-16-' + suffix2 +'.fits'
                path_zbest = path + 'zbest-16-' + suffix2 +'.fits'
                data = DesiMock()
                data.read_fits_file(path_spectra, path_truth, path_zbest)
                total = pd.DataFrame()
                for id in data.data:
                    sline = data.get_sightline(id=id)
                    wav_max, wav_min = 10**np.max(sline.loglam - np.log10(1+sline.z_qso)), 10**np.min(sline.loglam - np.log10(1+sline.z_qso))
                    info = pd.DataFrame()
                    info['id'] = np.ones(1, dtype='i8') * int(id)
                    info['z_qso'] = np.ones(1) * sline.z_qso
                    info['snr'] = np.ones(1) * sline.s2n
                    for name in names:
                        info[name] = [lines[name] >= wav_min and lines[name] <= wav_max]
                    total = pd.concat([total, info])
                total['file'] = np.ones(len(total), dtype='i8') * int(suffix2)
                total = total[['file', 'id', 'z_qso', 'snr', 'LyBETA', 'LyALPHA', 'MgII1', 'CIV1', 'MgII2', 'CIV2']]
                total.to_csv(prefix + suffix1 + '/' + suffix2 + '/catalog.csv', index=False)
                
# delete all the catalog
def delete_all_calalog(prefix):
    suffix = generate_suffix(prefix=prefix)
    for suffix1 in suffix.keys():
        for suffix2 in suffix[suffix1]:
            path = prefix + suffix1 + '/' + suffix2 + '/'
            files = os.listdir(path)
            if len(files) == 4:
                for file in files:
                    if '.csv' in file:
                        os.remove(path + file)
    if 'catalog_total.csv' in os.listdir(prefix):
        os.remove(prefix+'catalog_total.csv')
        
# generate a total catalog
# this should be done AFTER the catalog of each folder has been generated
def generate_total_catalog(prefix):
    suffix = generate_suffix(prefix=prefix)
    catalog = pd.DataFrame()
    for suffix1 in suffix.keys():
        for suffix2 in suffix[suffix1]:
            path = prefix + suffix1 + '/' + suffix2 + '/'
            files = os.listdir(path)
            if len(files) == 4:
                for file in files:
                    if '.csv' in file:
                        this = pd.read_csv(path+file)
                        catalog = pd.concat([catalog, this])

    catalog.to_csv(prefix+'catalog_total.csv')