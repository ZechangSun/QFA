

class Marker(object):
    IGNORE_FEATURE = -10

    def __init__(self, lam_rest_location, marker_type=IGNORE_FEATURE):
        self.lam_rest_location = lam_rest_location
        self.marker_type = marker_type
