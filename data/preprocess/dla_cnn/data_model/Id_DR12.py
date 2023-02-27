from .Id import Id
from pkg_resources import resource_filename


class Id_DR12(Id):
    @classmethod
    def from_csv(cls, plate, fiber):
        from astropy.table import Table
        import numpy as np
        csv_file = resource_filename('dla_cnn', 'catalogs/boss_dr12/dr12_set.csv')
        csv = Table.read(csv_file)
        # Match
        idx = np.where((csv['PLATE']==plate) & (csv['FIB']==fiber))[0]
        if len(idx) == 0:
            raise IOError("Bad plate/fiber for BOSS DR12")
        # Init
        mjd = csv['MJD'][idx[0]]
        id_dr12 = cls(plate, mjd, fiber, ra=csv['RA'][idx[0]], dec=csv['DEC'][idx[0]])
        # Return
        return id_dr12

    def __init__(self, plate, mjd, fiber, ra=0, dec=0):
        super(Id_DR12,self).__init__()
        self.plate = plate
        self.mjd = mjd
        self.fiber = fiber
        self.ra = ra
        self.dec = dec

    def id_string(self):
        return "%05d-%04d-%05d" % (self.plate, self.mjd, self.fiber)

