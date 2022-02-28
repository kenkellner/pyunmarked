import numpy as np

# Input data object
class UnmarkedFrame(object):
    def __init__(self, y, site_covs=None, obs_covs=None):
        self.y = y
        self.site_covs = site_covs
        self.obs_covs = obs_covs
