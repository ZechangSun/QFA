""" Parent class for a set of Data, training or real"""

from abc import ABCMeta

from dla_cnn.data_model import Id
from dla_cnn.data_model import Sightline


class Data(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.catalog_file = None
        self.catalog = None

    # Required methods -- All of these are dummy
    def gen_ID(self):
        return Id.Id()

    def load_data(self, id):
        raw_data = {}
        z_qso = 0.
        return raw_data, z_qso

    def load_catalog(self):
        self.cat = None

    def load_sightline(self, id):
        return Sightline.Sightline(id)
