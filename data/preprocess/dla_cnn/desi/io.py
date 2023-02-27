""" Methods for I/O for DESI project"""

from dla_cnn.data_model import Sightline


def read_mock_spectrum(id, dla_catalog):
    sightline = Sightline(id=id)

    """ Rest of this needs building """


def read_sightline(id):
    """
    Read Sightline from hard drive

    May need separate methods for Mocks vs. real data

    Parameters
    ----------
    id: int

    Returns
    -------
    dla_cnn.data_model.Sightline

    """
    sightline = Sightline(id=id)

    """ Rest of this needs building """
