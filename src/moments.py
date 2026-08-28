import numpy as np

# -----------------------------------------------------------------------------------------------------------------
# Computes the statistical moments of a one-dimensional discretized distribution
# and saves them in the array 'self.moments'.
# Expects x and pdf to be arrays of equal length, and pdf to be nonnegative.
# Kurtosis is Pearson kurtosis, with a lower bound of 1.
class Moments:
    def __init__(self, x, pdf):
        self.x              = np.asarray(x)
        self.pdf            = np.asarray(pdf)
        self.mean           = np.nan
        self.variance       = np.nan
        self.std            = np.nan
        self.skew           = np.nan
        self.kurtosis       = np.nan
        self.inv_kurt       = np.nan
        self.hyperskewness  = np.nan
        self.hyperkurtosis  = np.nan

        if np.sum(self.pdf) > 0:
            self.pdf           /= np.sum(self.pdf)
            self.mean           = np.sum(self.x * self.pdf)
            self.variance       = np.sum((self.x - self.mean) ** 2 * self.pdf)
            self.std            = np.sqrt(self.variance)
            if self.variance > 0:
                self.skew           = np.sum((self.x - self.mean) ** 3 * self.pdf) / self.std ** 3
                self.kurtosis       = np.sum((self.x - self.mean) ** 4 * self.pdf) / self.std ** 4
                self.inv_kurt       = 1/self.kurtosis
                self.hyperskewness  = np.sum((self.x - self.mean) ** 5 * self.pdf) / self.std ** 5
                self.hyperkurtosis  = np.sum((self.x - self.mean) ** 6 * self.pdf) / self.std ** 6

        self.moments        = np.asarray([self.mean, self.variance, self.std, self.skew, self.kurtosis,
                                            self.inv_kurt, self.hyperskewness, self.hyperkurtosis])