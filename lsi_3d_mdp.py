class LsiGridworld(object):
    def __init__(self, params):
        # set height,width,shape,etc.
        return NotImplementedError()

    @staticmethod
    def from_config(filename):
        # read in layout file and return new LsiGridworld instance
        return NotImplementedError()