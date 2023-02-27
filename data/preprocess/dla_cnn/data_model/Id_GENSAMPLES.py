""" Mainly used for training and validation
"""
from __future__ import print_function, absolute_import, division, unicode_literals

import os
from dla_cnn.data_model.Id import Id


class Id_GENSAMPLES(Id):
    def __init__(self, ix,
                 hdf5_datafile='../data/gensample_hdf5_files/test_dlas_96451_5000.hdf5',
                 json_datafile='../data/gensample_hdf5_files/test_dlas_96451_5000.json',
                 sightlineid=-1,):
        super(Id_GENSAMPLES, self).__init__()
        self.ix = ix
        self.sightlineid = sightlineid          # an index number identifying the underlying sightline that was uesd
        self.hdf5_datafile = hdf5_datafile
        self.json_datafile = json_datafile

    def id_string(self):
        filename = os.path.split(self.hdf5_datafile)[-1]
        basename = os.path.splitext(filename)[0]
        return basename + "_ix_" + "%04d"%self.ix
