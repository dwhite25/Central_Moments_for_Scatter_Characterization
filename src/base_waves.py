import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------------------------------
# parent class for handling creation of outbound interrogating wave, including
class BaseFunction:
    def __init__(self, t=[0], normalized=True):
        self.name       = 'BaseFunction'                    # name of the function type
        self.t          = t                                 # array of time domain values to sample the function at
        self.signal     = self.create_function(self.t)      # the outbound signal
        if normalized:
            self.signal = self.normalize(self.signal)

    # base function to be defined in subclasses
    def create_function(self, t):
        return np.zeros(len(t))

    # circularly convolves the interrogating waveform with the sampled scatter distribution using FFT multiplication
    def create_composite(self, scatter):
        scatter_shift   = np.fft.ifftshift(scatter)
        scatter_fft     = np.fft.fft(scatter_shift)
        signal_fft      = np.fft.fft(self.signal)
        composite       = np.fft.ifft(scatter_fft * signal_fft)
        return composite

    # normalizes the waveform to unit average power
    def normalize(self, signal):
        return signal/np.sqrt(np.mean(np.abs(signal)**2))

    # create the return wave by convolving the interrogating wave with the scatter function
    # and optionally adding noise. snr in units of db.
    def create_return_wave(self, scatter, noise=None, add_noise=False, snr=10, real=True, plot=False):
        composite       = self.create_composite(scatter)
        if real:
            composite   = np.real(composite)
        composite       = self.normalize(composite)
        if add_noise:
            if noise is None:
                snr_linear      = 10**(snr/10)
                noise_power     = 1/snr_linear
                noise_std       = np.sqrt(noise_power)
                if real:
                    noise       = np.random.normal(0, noise_std, len(self.t))
                else:
                    noise       = (np.random.normal(0, noise_std/np.sqrt(2), len(self.t)) +
                                   1j*np.random.normal(0, noise_std/np.sqrt(2), len(self.t)))
            composite += noise
        if plot:
            plt.plot(self.t, np.real(composite))
            plt.show()
        return composite

# -----------------------------------------------------------------------------------------------------------------
# class for waves with frequencies 'freqs', with corresponding amplitude coefficients as defined by 'coefs'
class Spectral(BaseFunction):
    def __init__(self, freqs=[0], t=[0], coefs=[1], normalized=True):
        self.coefs      = coefs
        self.freqs      = freqs

        super().__init__(t=t, normalized=normalized)

        self.name       = 'Spectral'

    # constructs the interrogating waveform as a sum of complex exponentials
    def create_function(self, t):
        self.N  = len(self.coefs)
        func    = np.zeros(len(t), dtype=complex)
        for i in range(0, self.N):
            func += self.coefs[i]*np.exp(1j*self.freqs[i]*t)
        return func