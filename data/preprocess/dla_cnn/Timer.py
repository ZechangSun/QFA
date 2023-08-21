import time


class Timer:
    def __init__(self, disp=None):
        self.disp = disp


    def __enter__(self, disp=None):
        self.start = time.time()
        if (self.disp is not None):
            print ("%s [Begin]" % self.disp)
        return self


    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if(self.disp is not None):
            print ("%s [%.1fs]" % (self.disp, self.interval))