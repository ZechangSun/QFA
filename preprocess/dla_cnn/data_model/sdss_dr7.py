""" Data object for SDSS DR7.  Uses IGMSPEC for the data """

from pkg_resources import resource_filename
import pdb

import numpy as np

from astropy.table import Table

from specdb.specdb import IgmSpec
from specdb import cat_utils

from dla_cnn.data_model import Data
from dla_cnn.data_model import Id
from dla_cnn.data_model import Sightline
from dla_cnn.data_model import data_utils

cache = {}      # Cache for multiprocessing

class SDSSDR7(Data.Data):

    def __init__(self, catalog_file=None):
        super(SDSSDR7,self).__init__()

        # Load IGMSpec which holds our data
        self.igmsp = IgmSpec()
        self.group = 'SDSS_DR7'
        self.meta = self.igmsp[self.group].meta  # Loads up the table

        # For multi-processing
        global cache
        cache['igmsp'] = self.igmsp

        # Catalog
        if catalog_file is None:
            self.catalog_file = resource_filename('dla_cnn', 'catalogs/sdss_dr7/dr7_set.csv')
        self.load_catalog()

    def gen_ID(self, plate, fiber, group_id, ra=None, dec=None):
        """

        Args:
            plate: int
            fiber: int
            group_id: int
            ra: float, optional
            dec: float, optional

        Returns:
            Id: Id object

        """
        return Id_DR7(plate, fiber, ra=ra, dec=dec, group_id=group_id)

    def load_catalog(self, csv=True):
        """
        Load up the source catalog

        Uses self.catalog_file

        Args:
            csv: bool, optional

        Returns:

        """
        # Load it
        self.catalog = Table.read(self.catalog_file)
        # Add IDs
        meta_pfib = self.meta['PLATE']*1000000 + self.meta['FIBER']
        cat_pfib = self.catalog['PLATE']*1000000 + self.catalog['FIBER']
        mIDs = cat_utils.match_ids(cat_pfib, meta_pfib, require_in_match=False)
        self.catalog['GROUP_ID'] = mIDs

    def load_IDs(self, pfiber=None):
        """
        Load up a list of Id objects from the catalog

        Args:
            pfiber: tuple, optional
              Plate, fiber to restrict IDs to a single sightline

        Returns:
            ids : list of Id objects

        """
        ids = [self.gen_ID(c['PLATE'],c['FIBER'],c['GROUP_ID'], ra=c['RA'],
                           dec=c['DEC']) for ii,c in enumerate(self.catalog)]
        # Single ID using plate/fiber?
        if pfiber is not None:
            plates = np.array([iid.plate for iid in ids])
            fibers = np.array([iid.fiber for iid in ids])
            imt = np.where((plates==pfiber[0]) & (fibers==pfiber[1]))[0]
            if len(imt) != 1:
                print("Plate/Fiber not in DR7!!")
                pdb.set_trace()
            else:
                ids = [ids[imt[0]]]
        return ids


def load_data(id):
    """
    Load the spectrum for a single object

    Needs to be a stand-alone method for multi-processing

    Args:
        id:

    Returns:
        raw_data: dict
          Contains the spectral info
        z_qso: float
          Quasar redshift

    """
    global cache
    data, meta = cache['igmsp'][id.group].grab_specmeta(id.group_id, use_XSpec=False)
    z_qso = meta['zem_GROUP'][0]

    flux = data['flux'].flatten() #np.array(spec[0].flux)
    sig = data['sig'].flatten() # np.array(spec[0].sig)
    loglam = np.log10(data['wave'].flatten())

    gdi = np.isfinite(loglam)

    (loglam_padded, flux_padded, sig_padded) = data_utils.pad_loglam_flux(
        loglam[gdi], flux[gdi], z_qso, sig=sig[gdi]) # Sanity check that we're getting the log10 values
    assert np.all(loglam < 10), "Loglam values > 10, example: %f" % loglam[0]

    raw_data = {}
    raw_data['flux'] = flux_padded
    raw_data['sig'] = sig_padded
    raw_data['loglam'] = loglam_padded
    raw_data['plate'] = id.plate
    raw_data['mjd'] = 0
    raw_data['fiber'] = id.fiber
    raw_data['ra'] = id.ra
    raw_data['dec'] = id.dec
    assert np.shape(raw_data['flux']) == np.shape(raw_data['loglam'])
    #sys.stdout = stdout
    # Return
    return raw_data, z_qso


def read_sightline(id):
    """
    Instantiate a Sightline object for a given Id
    Fills in the spectrum with a call to load_data()

    Needs to be a stand-alone method for multi-processing

    Args:
        id: Id object

    Returns:
        sightline: Sightline object

    """
    sightline = Sightline.Sightline(id=id)
    # Data
    data1, z_qso = load_data(id)
    # Fill
    sightline.id.ra = data1['ra']
    sightline.id.dec = data1['dec']
    sightline.flux = data1['flux']
    sightline.sig = data1['sig']
    sightline.loglam = data1['loglam']
    sightline.z_qso = z_qso
    # Giddy up
    return sightline

class Id_DR7(Id.Id):
    """
    SDSS-DR7 specific Id object
    """
    def __init__(self, plate, fiber, ra=0, dec=0, group_id=-1):
        super(Id_DR7,self).__init__()
        self.plate = plate
        self.fiber = fiber
        self.ra = ra
        self.dec = dec
        self.group_id = group_id
        self.group = 'SDSS_DR7'

    def id_string(self):
        return "%05d-%05d" % (self.plate, self.fiber)


def process_catalog_dr7(kernel_size=400, pfiber=None, make_pdf=False,
                        model_checkpoint=None, #default_model,
                        output_dir="../tmp/visuals_dr7",
                        debug=False):
    """ Runs a SDSS DR7 DLA search using the SDSSDR7 data object

    Parameters
    ----------
    kernel_size
    pfiber: tuple, optional
      plate, fiber  (int)
      Restrict the run to a single sightline
    make_pdf: bool, optional
      Generate PDFs
    model_checkpoint
    output_dir

    Returns
    -------
      Nothing
    Code generates predictions.json which contains
      the information on DLAs in each processed sightline

    """
    from dla_cnn.data_loader import process_catalog
    # Instantiate the data object
    data = SDSSDR7()
    # Load the IDs
    ids = data.load_IDs(pfiber=pfiber)
    # Run
    process_catalog(ids, kernel_size, model_checkpoint, make_pdf=make_pdf,
                    CHUNK_SIZE=500, output_dir=output_dir, data=data, debug=debug,
                    data_read_sightline=read_sightline)
