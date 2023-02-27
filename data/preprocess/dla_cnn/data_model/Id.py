
class Id(object):
    def __init__(self):
        self.ra = 0
        self.dec = 0

    def id_string(self):
        return "ra=%0.5f, dec=%0.5f" % (self.ra, self.dec)

