#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# include the necessary libraries for the code in the cell below
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# -----------------------------------------------------------------------------------------------------------------
# parent class for handling creation of outbound interrogating wave, including
class BaseFunction:
    def __init__(self, f_low=2*np.pi, f_high=4*np.pi, t=[0], symmetry='even', normalized=True):
        self.name       = 'BaseFunction'                    # name of the function type
        self.f_low      = f_low                             # lowest frequency
        self.f_high     = f_high                            # highest frequency
        self.t          = t                                 # array of time domain values to sample the function at
        self.signal     = self.create_function(self.t)      # the outbound signal
        self.symmetry   = symmetry                          # the symmetry of the function. 'odd', 'even', or 'none'
        if normalized:
            self.signal = self.normalize(self.signal)

    # the base function. to be defined in subclasses.
    def create_function(self, t):
        return np.zeros(len(t))

    # a composite function creating by scattering the base function with amplitudes A at time shifts tshifts
    def create_composite(self, scatter):
        class_comp = np.convolve(self.signal, scatter, mode='same')
        return class_comp

    # simple call to normalize the function if so desired
    def normalize(self, signal):
        return signal/np.sqrt(np.mean(np.abs(signal)**2))

    # in case, for whatever reason, it's important to nornalize only the superoscillating portion of a wave.
    # unused in this experiment...I think.
    def normalize_to_polyfit_region(self, signal, SampleRate=1000, NormWindow=1):
        WindowLen   = SampleRate*NormWindow
        t1          = int((len(self.t)-WindowLen)/2)
        t2          = int((len(self.t)+WindowLen)/2)
        signal      = signal/np.sqrt(np.mean(np.real(signal[t1:t2])**2))
        return signal

    def create_return_wave(self, scatter, noise=None, add_noise=False, snr=100, real=True, plot=False):
        composite = self.create_composite(scatter)
        if real:
            composite = np.real(composite)
        composite = self.normalize(composite)
        if add_noise:
            if noise is None:
                if real:
                    noise = np.random.normal(0, 1/snr, len(self.t))
                else:
                    noise = np.random.normal(0, 1/snr, len(self.t)) + 1j*np.random.normal(0, 1/snr, len(self.t))
            composite += noise
        if plot:
            plt.plot(self.t, np.real(composite))
            plt.show()
        return composite

# -----------------------------------------------------------------------------------------------------------------
# creates a modified sum of sincs, bandlimited to be between f_low and f_high
class Sinc(BaseFunction):
    def __init__(self, f_low=2*np.pi, f_high=4*np.pi, t=[0], sinc_order=1, normalized=True):
        self.sinc_order = sinc_order        # controls the bandwidth of the sinc

        super().__init__(f_low=f_low, f_high=f_high, t=t, normalized=normalized)

        self.name       = 'Sinc'

    # modified sinc to the sinc_order order, with frequencies exclusively between w_high and w_low
    def create_function(self, t):
        new_t_high   = self.f_high*t/self.sinc_order
        new_t_low    = self.f_low*t/self.sinc_order
        ratio        = self.f_high/self.f_low
        # store func as a complex array
        func    = np.array((((ratio*(np.sin(new_t_high)/(new_t_high))**self.sinc_order) -
                             ((np.sin(new_t_low)/(new_t_low))**self.sinc_order)) +
                            1j*(((np.cos(new_t_high)/(new_t_high))**self.sinc_order) -
                             ((np.cos(new_t_low)/(new_t_low))**self.sinc_order))), dtype=complex)
        return func

# -----------------------------------------------------------------------------------------------------------------
class Gauss(BaseFunction):
    def __init__(self, t=[0], normalized=True):
        super().__init__(t=t, normalized=normalized)

        self.name   = 'Gauss'

    # gaussian pulse
    def create_function(self, t):
        func = np.exp(-(2*np.pi*t)**2)
        return func

# -----------------------------------------------------------------------------------------------------------------
# class for waves with 'num_freqs' evenly spaced frequencies between 'f_low' and 'f_high', with corresponding
# amplitude coefficients as defined by 'coefs'
class SuperRandom(BaseFunction):
    def __init__(self, f_low=2*np.pi, f_high=4*np.pi, t=[0], symmetry='even', coefs=[0], normalized=True, num_freqs=6):
        self.N          = num_freqs
        self.symmetry   = symmetry
        self.coefs      = coefs

        super().__init__(f_low=f_low, f_high=f_high, t=t, normalized=normalized)

        self.name       = 'SuperSinc'

    # creates a complex signal for either even or odd functions for each of the constituent frequencies in the
    # final series
    def basis(self, times, n):
        dw = (self.f_high - self.f_low)/(self.N-1)
        g = np.zeros(len(times), dtype=complex)
        if self.symmetry == 'even':
            g = np.exp(1j*(self.f_low + dw*n)*times)
        elif self.symmetry == 'odd':
            g = np.exp(1j*(self.f_low + dw*n)*(times + np.pi/2))
        return g

    # uses self.basis() to create the complete frequency series in time
    def create_function(self, t):
        self.N  = len(self.coefs)
        func    = np.zeros(len(t), dtype=complex)
        for i in range(0, self.N):
            func += self.coefs[i]*self.basis(t, i)
        return func

