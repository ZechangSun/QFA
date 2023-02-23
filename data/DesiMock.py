from astropy.io import fits
import numpy as np
from dla_cnn.data_model.Sightline import Sightline
from dla_cnn.data_model.Dla import Dla
from dla_cnn.desi import preprocess
from .defs import best_v


class DesiMock: 
    """
    a class to load all spectrum from a mock DESI data v9 fits file, each file contains about 1186 spectrum.
    --------------------------------------------------------------------------------------------------
    attributes:
    wavelength: array-like, the wavelength of all spectrum (all spectrum share same wavelength array)
    data: dict, using each spectra's id as its key and a dict of all data we need of this spectra as its value
      its format like spectra_id: {'FLUX':flux,'ERROR':error,'z_qso':z_qso, 'RA':ra, 'DEC':dec, 'DLAS':a tuple of Dla objects containing the information of dla}
    split_point_br: int,the length of the flux_b,the split point of  b channel data and r channel data
    split_point_rz: int,the length of the flux_b and flux_r, the split point of  r channel data and z channel data
    data_size: int,the point number of all data points of wavelength and flux
    """

    def __init__(self, wavelength = None, data = {}, split_point_br = None, split_point_rz = None, data_size = None):
        self.wavelength = wavelength
        self.data = data
        self.split_point_br = split_point_br
        self.split_point_rz = split_point_rz
        self.data_size = data_size

    def read_fits_file(self, spec_path, truth_path, zbest_path):
        """
        read Desi Mock spectrum from a fits file, load all spectrum as a DesiMock object
        ------------------------------------------------------------------------------------------------
        parameters:
        
        spec_path:  str, spectrum file path
        truth_path: str, truth file path
        zbest_path: str, zbest file path
        ------------------------------------------------------------------------------------------------
        return:
        
        self.wavelength,self.data(contained all information we need),self.split_point_br,self.split_point_rz,self.data_size
        """
        spec = fits.open(spec_path)
        truth = fits.open(truth_path)
        zbest = fits.open(zbest_path)

        # spec[2].data ,spec[7].data and spec[12].data are the wavelength data for the b, r and z cameras.
        self.wavelength = np.hstack((spec[2].data.copy(), spec[7].data.copy(), spec[12].data.copy()))
        self.data_size = len(self.wavelength)

        dlas_data = truth[3].data[truth[3].data.copy()['NHI']>19.3]
        spec_dlas = {}
        # item[2] is the spec_id, item[3] is the dla_id, and item[0] is NHI, item[1] is z_qso
        for item in dlas_data:
            if item[2] not in spec_dlas:
                spec_dlas[item[2]] = [Dla((item[1]+1)*1215.6701, item[0], '00'+str(item[3]-item[2]*1000))]
            else:
                spec_dlas[item[2]].append(Dla((item[1]+1)*1215.6701, item[0], '00'+str(item[3]-item[2]*1000)))

        test = np.array([True if item in dlas_data['TARGETID'] else False for item in spec[1].data['TARGETID'].copy()])
        for item in spec[1].data['TARGETID'].copy()[~test]:
            spec_dlas[item] = []

        # read data from the fits file above, one can directly get those varibles meanings by their names.
        spec_id = spec[1].data['TARGETID'].copy()
        flux_b = spec[3].data.copy()
        flux_r = spec[8].data.copy()
        flux_z = spec[13].data.copy()
        flux = np.hstack((flux_b,flux_r,flux_z))
        ivar_b = spec[4].data.copy()
        ivar_r = spec[9].data.copy()
        ivar_z = spec[14].data.copy()
        error = 1./np.sqrt(np.hstack((ivar_b,ivar_r,ivar_z)))
        self.split_point_br = flux_b.shape[1]
        self.split_point_rz = flux_b.shape[1]+flux_z.shape[1]
        z_qso = zbest[1].data['Z'].copy()
        ra = spec[1].data['TARGET_RA'].copy()
        dec = spec[1].data['TARGET_DEC'].copy()

        self.data = {spec_id[i]:{'FLUX':flux[i],'ERROR': error[i], 'z_qso':z_qso[i] , 'RA': ra[i], 'DEC':dec[i], 'DLAS':spec_dlas[spec_id[i]]} for i in range(len(spec_id))}

    def get_sightline(self, id, camera = 'all', rebin=False, normalize=False):
        """
        using id(int) as index to retrive each spectra in DesiMock's dataset, return  a Sightline object.
        ---------------------------------------------------------------------------------------------------
        parameters:
        id: spectra's id , a unique number for each spectra.
        camera: str, 'b' : Load up the wavelength and data for the blue camera., 'r': Load up the wavelength and data for the r camera,
                     'z' : Load up the wavelength and data for the z camera, 'all':  Load up the wavelength and data for all cameras.
        rebin: bool, if True rebin the spectra to the best dlambda/lambda, default False,
        normalize: bool, if True normalize the spectra, using the slice of flux from wavelength ~1070 to 1170, default False.
        ---------------------------------------------------------------------------------------------------
        return:
        sightline: dla_cnn.data_model.Sightline.Sightline object
        """
        assert camera in ['all', 'r', 'z', 'b'], "No such camera! The parameter 'camera' must be in ['all', 'r', 'b', 'z']"
        sightline = Sightline(id)

        # this inside method can get the data(wavelength, flux, error) from the start_point(int) to end_point(int)
        def get_data(start_point=0, end_point=self.data_size):
            """

            Parameters
            ----------
            start_point: int, the start index of the slice of the data(wavelength, flux, error), default 0
            end_point: int, the end index of the slice of the data(wavelength, flux, error), default the length of the data array
            
            Returns
            -------
            
            """
            sightline.flux = self.data[id]['FLUX'][start_point:end_point]
            sightline.error = self.data[id]['ERROR'][start_point:end_point]
            sightline.z_qso = self.data[id]['z_qso']
            sightline.ra = self.data[id]['RA']
            sightline.dec = self.data[id]['DEC']
            sightline.dlas = self.data[id]['DLAS']
            sightline.loglam = np.log10(self.wavelength[start_point:end_point])
            sightline.split_point_br = self.split_point_br
            sightline.split_point_rz = self.split_point_rz
            sightline.s2n = preprocess.estimate_s2n(sightline)

        # invoke the inside function above to select different camera's data.
        if camera == 'all':
            get_data()
            #this part is to deal with the overlap between different cameras.
            if rebin:
                sortedindex = np.argsort(sightline.loglam)
                sightline.flux = sightline.flux[sortedindex]
                sightline.loglam = sightline.loglam[sortedindex]
                sightline.error = sightline.error[sortedindex]
        elif camera == 'b':
            get_data(end_point = self.split_point_br)
        elif camera == 'r':
            get_data(start_point= self.split_point_br, end_point= self.split_point_rz)
        else:
            get_data(start_point=self.split_point_rz)

        # if the parameter rebin is True, then rebin this sightline using rebin method in preprocess.py and the v we determined previously(defs.py/best_v) .
        if rebin:
            preprocess.rebin(sightline, best_v[camera])
        #if the parameter normalize is True, then normalize this sightline using the method in preprocess.py
        if normalize:
            preprocess.normalize(sightline, self.wavelength, self.data[id]['FLUX'])

        # Return the Sightline object
        return sightline
