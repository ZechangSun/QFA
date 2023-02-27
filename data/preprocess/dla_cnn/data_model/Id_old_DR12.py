from .Id import Id


class Id_old_DR12(Id):
    def __init__(self, plate, mjd, fiber, ra=0, dec=0):
        super(Id_old_DR12,self).__init__()
        self.plate = plate
        self.mjd = mjd
        self.fiber = fiber
        self.ra = ra
        self.dec = dec

    def id_string(self):
        return "%05d-%04d-%05d" % (self.plate, self.mjd, self.fiber)

