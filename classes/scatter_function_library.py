#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import gmm_library as gmm
import matplotlib.pyplot as plt

from scipy.optimize import brentq


# In[ ]:


# -----------------------------------------------------------------------------------------------------------------
# parent class for scatter distributions
class ScatterDistribution:
    def __init__(self, x):
        self.x          = x

        self.create_distribution()

    def create_distribution(self):
        self.scatter    = np.zeros_like(self.x)

    def compute_mean(self, offset):
        scatter         = self._generate_scatter(offset)
        return np.sum(self.x * scatter)

    # to be used in child classes. allows for control of location of distribution's mean
    def find_offset_for_mean(self, target_mean):
        def objective(offset):
            return self.compute_mean(offset) - target_mean
        lower   = self.x[0] + self.radius + 1e-6
        upper   = self.x[-1] - 1e-6
        return brentq(objective, lower, upper)

# -----------------------------------------------------------------------------------------------------------------
# scatter disrtibution comprised of delta functions, simulating a collection of point scatterers
class Delta(ScatterDistribution):
    def __init__(self, x, loc1=0., loc2=0.49, loc3=0.49, amp1=0., amp2=0., amp3=0.):
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
# gaussian mixture model scatter distribution
class GMM(ScatterDistribution):
    def __init__(self, x, mu1=0., mu2=0., simga=1., w1=.5):
        self.model      = 'GMM'
        self.mu1        = mu1
        self.mu2        = mu2
        self.sigma      = simga
        self.w1         = w1

        super().__init__(x)

    def create_distribution(self):
        targets         = [self.mu1, self.mu2, self.sigma, self.w1]
        self.scatter    = gmm.gmm_pdf(self.x, targets)

# -----------------------------------------------------------------------------------------------------------------
# the scatter distribution originating from a tilted plane with finite length.
# simulated as the visible cross-sectional area of the for every dt as a function of depth
class Plane(ScatterDistribution):
    def __init__(self, x, radius=1., std=1., offset=0., by_std=True):
        self.model          = 'Plane'
        self.radius         = radius
        self.std            = std
        self.offset         = offset
        self.by_std         = by_std
        self.std_to_radius  = 1.732

        super().__init__(x)

    def create_distribution(self):
        if self.by_std:
            self.radius     = self.std*self.std_to_radius
        low                 = self.offset - self.radius
        high                = self.offset + self.radius
        mask                = (self.x >= low) & (self.x <= high)
        self.scatter        = np.zeros_like(self.x)
        self.scatter[mask]  = 1.
        self.scatter       /= np.sum(self.scatter)

# -----------------------------------------------------------------------------------------------------------------
# the scatter distribution originating from a solid sphere.
# simulated as the visible cross-sectional area of the for every dt as a function of depth
class Sphere(ScatterDistribution):
    def __init__(self, x, radius=1., offset=0., std=1., by_mean=False, by_std=True):
        self.model          = 'Sphere'
        self.radius         = radius
        self.offset         = offset
        self.std            = std
        self.by_mean        = by_mean
        self.by_std         = by_std
        self.std_to_radius  = 4.2426

        super().__init__(x)

    def create_distribution(self):
        if self.by_std:
            self.radius     = self.std*self.std_to_radius
        if self.by_mean:
            self.offset     = self.find_offset_for_mean(self.offset)
        self.scatter        = self._generate_scatter(self.offset)

    def _generate_scatter(self, offset):
        temp                = np.zeros_like(self.x)
        dx                  = self.x - offset
        mask                = (dx <= 0) & (dx >= -self.radius)
        temp[mask]          = np.pi * (self.radius**2 - dx[mask]**2)
        scatter             = np.zeros_like(self.x)
        scatter[1:]         = np.maximum(temp[1:] - temp[:-1], 0)
        scatter            /= np.sum(scatter)
        return scatter

# -----------------------------------------------------------------------------------------------------------------
# the scatter distribution originating from a solid cylinder laying perfectly on its side.
# simulated as the visible cross-sectional area of the for every dt as a function of depth
class Cylinder(ScatterDistribution):
    def __init__(self, x, radius=1., offset=0., std=1., by_mean=False, by_std=True):
        self.model          = 'Cylinder'
        self.radius         = radius
        self.offset         = offset
        self.std            = std
        self.by_mean        = by_mean
        self.by_std         = by_std
        self.std_to_radius  = 4.48

        super().__init__(x)

    def create_distribution(self):
        if self.by_std:
            self.radius     = self.std*self.std_to_radius
        if self.by_mean:
            self.offset     = self.find_offset_for_mean(self.offset)
        self.scatter        = self._generate_scatter(self.offset)

    def _generate_scatter(self, offset):
        temp                = np.zeros_like(self.x)
        dx                  = self.x - offset
        mask                = (dx <= 0) & (dx >= -self.radius)
        temp[mask]          = np.sqrt(self.radius**2 - dx[mask]**2)
        scatter             = np.zeros_like(self.x)
        scatter[1:]         = np.maximum(temp[1:] - temp[:-1], 0)
        scatter            /= np.sum(scatter)
        return scatter

