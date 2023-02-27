import numpy as np


class Sightline(object):

    def __init__(self, id, ra=None,dec=None,dlas=None, flux=None, loglam=None,error=None, z_qso=None, split_point_br = None, split_point_rz = None,s2n = None,normalize = False):
        """

        Args:
            id (int):  Index identifier for the sightline
            dlas (list): List of DLAs
            flux (np.ndarray):
            wavelength (np.ndarray):
                observed wavelength values
            z_qso (float):
                Quasar redshift
            normalize(bool):
                if True, the Sightlne has been normalized, default not.
            split_point_br(int):
                the split index of b channel and r channel for desi spectra
            split_point_rz(int):
                the split index of r channel and z channel for desi spectra
            s2n(float):
                the s/n of lymann forest part(about 1070 -1170 A in rest frame (avoid dlas +- 3000 km/s)) 
            
        """
        self.ra=ra
        self.dec=dec
        self.flux = flux
        self.loglam = loglam
        self.id = id
        self.dlas = dlas
        self.z_qso = z_qso
        self.data_makers = []
        self.error = error # error = 1./ np.sqrt(ivar)
        self.normalize = normalize
        self.split_point_br = split_point_br
        self.split_point_rz = split_point_rz
        self.s2n = s2n

        # Attributes
        self.prediction = None
        self.classification = None
        self.offsets = None
        self.column_density = None


    # Returns the data in the legacy data1, qso_z format for code that hasn't been updated to the new format yet
    def get_legacy_data1_format(self):
        raw_data = {}
        raw_data['flux'] = self.flux
        raw_data['loglam'] = self.loglam
        raw_data['plate'] = self.id.plate if hasattr(self.id, 'plate') else 0
        raw_data['mjd'] = self.id.mjd if hasattr(self.id, 'mjd') else 0
        raw_data['fiber'] = self.id.fiber if hasattr(self.id, 'fiber') else 0
        raw_data['ra'] = self.id.ra if hasattr(self.id, 'ra') else 0
        raw_data['dec'] = self.id.dec if hasattr(self.id, 'dec') else 0
        return raw_data, self.z_qso


    # Clears all fields of the DLA
    def clear(self):
        self.flux = None
        self.loglam = None
        self.id = None
        self.dlas = None
        self.z_qso = None
        self.prediction = None
        self.data_markers = []


    def is_lyb(self, peakix):
        """
        Returns true if the given peakix (from peaks_ixs) is the ly-b of another DLA in the set peaks_ixs in prediction
        :param peakix:
        :return: boolean
        """
        assert self.prediction is not None and peakix in self.prediction.peaks_ixs

        lambda_higher = (10**self.loglam[peakix]) / (1025.722/1215.67)

        # An array of how close each peak is to beign the ly-b of peakix in spectrum reference frame
        peak_difference_spectrum = np.abs(10**self.loglam[self.prediction.peaks_ixs] - lambda_higher)
        nearest_peak_ix = np.argmin(peak_difference_spectrum)

        # get the column density of the identfied nearest peak
        _, potential_lya_nhi, _, _ = \
            self.prediction.get_coldensity_for_peak(self.prediction.peaks_ixs[nearest_peak_ix])
        _, potential_lyb_nhi, _, _ = \
            self.prediction.get_coldensity_for_peak(peakix)

        # Validations: check that the nearest peak is close enough to match
        #              sanity check that the LyB is at least 0.3 less than the DLA
        is_nearest_peak_within_range = peak_difference_spectrum[nearest_peak_ix] <= 15
        is_nearest_peak_larger_coldensity = potential_lyb_nhi < potential_lya_nhi - 0.3

        return is_nearest_peak_within_range and is_nearest_peak_larger_coldensity


    def get_lyb_index(self, peakix):
        """
        Returns the index location of the Ly-B absorption for a given peak index value
        :param peakix:
        :return: index location of Ly-B
        """
        spectrum_higher = 10**self.loglam[peakix]
        spectrum_lambda_lower = spectrum_higher * (1025.722 / 1215.67)
        log_lambda_lower = np.log10(spectrum_lambda_lower)
        ix_lambda_lower = (np.abs(self.loglam - log_lambda_lower)).argmin()
        return ix_lambda_lower

    def process(self, model_path):
        """  The following should follow the algorithm in process_catalog
        :param model_path:
        :return:
        """
        import json
        from dla_cnn.data_loader import scan_flux_sample
        from dla_cnn.localize_model import predictions_ann as predictions_ann_c2
        from dla_cnn.data_loader import compute_peaks, get_lam_data
        #from dla_cnn.data_loader import add_abs_to_sightline
        from dla_cnn.absorption import add_abs_to_sightline
        from dla_cnn.data_model.Prediction import Prediction
        # Fluxes
        fluxes = scan_flux_sample(self.flux, self.loglam, self.z_qso, -1, stride=1)[0]
        # Model
        with open(model_path+"_hyperparams.json",'r') as f:
            hyperparameters = json.load(f)
        loc_pred, loc_conf, offsets, density_data_flat = predictions_ann_c2(hyperparameters, fluxes, model_path)
        self.prediction = Prediction(loc_pred=loc_pred, loc_conf=loc_conf, offsets=offsets, density_data=density_data_flat)
        # Peaks
        _ = compute_peaks(self)
        # Absorbers?
        add_abs_to_sightline(self)
