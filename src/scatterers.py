import numpy as np
import gmm

# -----------------------------------------------------------------------------------------------------------------
# parent class for one-dimensional scatter distributions
class ScatterDistribution:
    def __init__(self, x):
        self.x          = x

        self.create_distribution()

    def create_distribution(self):
        self.scatter    = np.zeros_like(self.x)

# -----------------------------------------------------------------------------------------------------------------
# discrete scatter distribution composed of up to three point scatterers
# delta locations are assumed to lie exactly on the sampled x grid
class Delta(ScatterDistribution):
    def __init__(self, x, loc1=0., loc2=0.49, loc3=0.49, amp1=1., amp2=0., amp3=0.):
        self.model          = 'Delta'
        self.loc1           = loc1
        self.loc2           = loc2
        self.loc3           = loc3
        self.amp1           = amp1
        self.amp2           = amp2
        self.amp3           = amp3

        super().__init__(x)

        self.delta_count    = np.count_nonzero(self.scatter)

    def create_distribution(self):
        self.scatter        = np.zeros_like(self.x)
        self.scatter[np.where(self.x == self.loc1)] = self.amp1
        self.scatter[np.where(self.x == self.loc2)] = self.amp2
        self.scatter[np.where(self.x == self.loc3)] = self.amp3
        self.scatter       /= np.sum(self.scatter)

# -----------------------------------------------------------------------------------------------------------------
# equal-variance two-component Gaussian-mixture scatter distribution
class GMM(ScatterDistribution):
    def __init__(self, x, mu1=0., mu2=0., sigma=1., w1=.5):
        self.model      = 'GMM'
        self.mu1        = mu1
        self.mu2        = mu2
        self.sigma      = sigma
        self.w1         = w1

        super().__init__(x)

    def create_distribution(self):
        params          = [self.mu1, self.mu2, self.sigma, self.w1]
        self.scatter    = gmm.gmm_pdf(self.x, params)