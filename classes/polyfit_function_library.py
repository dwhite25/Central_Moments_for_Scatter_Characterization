#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import gmm_library as gmm
import moments_function_library as mfl
import base_waves_class_library as bwl
import scatter_function_library as sfl

from numpy.polynomial import Polynomial, Chebyshev


# In[ ]:


# -----------------------------------------------------------------------------------------------------------------
# create cross-correlations between two time series. while not used in the final paper, this is useful for seeing
# how two signals observably differ
# returns:
#   - correlation between two signals in time with no lag adjustment
#   - maximum correlation between two signals across all lag adjustments
#   - lag value at which maximum correlation was found
def cross_correlations(outbound, return_wave):
    outbound            = np.asarray(outbound)
    return_wave         = np.asarray(return_wave)
    x                   = outbound - np.mean(outbound)
    y                   = return_wave - np.mean(return_wave)
    full_corr           = np.correlate(x, y, mode='full')
    lags                = np.arange(-len(x)+1, len(y))
    zero_lag_corr       = full_corr[len(x)-1]
    idx_max             = np.argmax(np.abs(full_corr))
    max_corr            = full_corr[idx_max]
    best_lag            = lags[idx_max]
    return [np.abs(zero_lag_corr), np.abs(max_corr), int(best_lag)]

# creates a Chebyshev polynomial fit of 'segments' segments of the return signal, up to order 'power', and
# returns all coefficients. can handle either real or complex signals.
def poly_fit(t, return_sig, FitWindow=None, WavePeriod=5, SampleRate=1000, power=4, real=False, segments=5):
    if real:
        return_sig      = np.real(return_sig)
    if FitWindow is None:
        FitWindow       = WavePeriod / segments
    SignalLen           = len(return_sig)
    WindowLen           = int(SampleRate * FitWindow)
    WaveLen             = int(SampleRate * WavePeriod)
    cheby_coefs_all     = []
    for seg in range(segments):
        t1              = int((SignalLen - WaveLen) /2 + (WindowLen * seg))
        t2              = int((SignalLen - WaveLen) /2 + (WindowLen * (seg + 1)))
        fitted_cheby    = Chebyshev.fit(t[t1:t2], return_sig[t1:t2], power, domain=[t[0], t[-1]])
        cheby_coefs     = fitted_cheby.coef
        cheby_coefs_all.extend(cheby_coefs)
    if not real:
        cheby_real      = np.real(np.array(cheby_coefs_all))
        cheby_imag      = np.imag(np.array(cheby_coefs_all))
        cheby_coefs_all = np.concatenate((cheby_real, cheby_imag))
    return np.array(cheby_coefs_all)

# finds the largest 'freqs' frequencies in the return signal from 'base', where 'base' is the class object from bwl
# additionally finds the phase wrapping of the frequency spectrum, which should relate to the return signal's phase
# offset in time.
# while this approach was unused in the final paper, it offers an approach to understanding the return signal in
# frequency space, rather than relying on polynomial fits in time. future work may compare the accuracy of both methods
def add_frequencies(base, freqs=6):
    f                   = np.fft.rfft(np.real(base.return_sig)) if real else np.fft.fft(base.return_sig)
    indexed_values      = list(enumerate(f))
    top_n               = sorted(sorted(indexed_values, key=lambda x: abs(x[1]), reverse=True)[:freqs],
                            key=lambda x: x[0])
    f_complex           = [val for idx, val in top_n]
    f_amps              = np.abs(f_complex)
    f_phases            = np.angle(f_complex)
    f_phases[::2]      -= np.pi
    f_unwrapped         = np.unwrap(f_phases)
    f_slope, _          = np.polyfit([i for i in range(len(f_unwrapped))], f_unwrapped, 1)
    final_array         = np.append(f_amps, f_slope)
    return final_array

# finds and returns all polynomial coefficients for the return signal originating from the outbound signal convolved
# with a series of delta functions as scatterers, with optional noise added. additionally returns cross-correlations
# with the outbound wave, and optionally returns target frequencies from add_frequencies().
def add_poly_coefs(row, x, base, power, FitWindow=None, add_noise=False, real=False, snr=100,
                    segments=3, freqs=6, plot=False, SampleRate=1000, add_freqs=False, f_low=0, use_base=False):
    delta               = sfl.Delta(x, loc1=row['loc_1'], loc2=row['loc_2'], loc3=row['loc_3'],
                                    amp1=row['amp_1'], amp2=row['amp_2'], amp3=row['amp_3'])
    base.return_sig     = base.create_return_wave(delta.scatter, plot=plot,
                                                  add_noise=add_noise, snr=snr, real=real)
    if use_base:
        base.return_sig = base.return_sig * np.exp(-1j * (f_low + .5) * base.t)
    poly_coefs          = poly_fit(base.t, base.return_sig, SampleRate=SampleRate, real=real,
                                   power=power, FitWindow=FitWindow, segments=segments)
    final_array         = np.array(row)
    final_array         = np.append(final_array, poly_coefs)
    if add_freqs:
        f_vals          = add_frequencies(base, freqs=freqs)
        final_array     = np.append(final_array, f_vals)
    corrs               = cross_correlations(base.signal, base.return_sig)
    final_array         = np.append(final_array, corrs)
    return final_array

# same as add_poly_coefs, except using a Gaussian Mixture Model as the scatterer. in addition to cross-correlations
# with the outbound signal, also returns cross-correlations with the return signal from delta scatterers with the
# same standardized central moments, up to kurtosis, as the GMM.
def add_poly_coefs_gmm(row, x, base, power, FitWindow=None, add_noise=False, real=False, snr=100,
                       segments=3, freqs=6, plot=False, SampleRate=1000, add_freqs=False, f_low=0, use_base=False):
    delta               = sfl.Delta(x, loc1=row['loc_1'], loc2=row['loc_2'], loc3=row['loc_3'],
                                    amp1=row['amp_1'], amp2=row['amp_2'], amp3=row['amp_3'])
    delta_return        = base.create_return_wave(delta.scatter, plot=plot,
                                                  add_noise=add_noise, snr=snr, real=real)
    results             = [row['mu1'], row['mu2'], row['sigma'], row['w1']]
    scatter             = gmm.gmm_pdf(x, results)
    moments             = mfl.Moments(x, scatter).moments
    final_array         = np.append(results, moments)
    base.return_sig     = base.create_return_wave(scatter, plot=plot, add_noise=add_noise, snr=snr, real=real)
    if use_base:
        base.return_sig = base.return_sig * np.exp(-1j * (f_low + .5) * base.t)
    poly_coefs          = poly_fit(base.t, base.return_sig, SampleRate=SampleRate, real=real,
                                    power=power, FitWindow=FitWindow, segments=segments)
    final_array         = np.append(final_array, poly_coefs)
    if add_freqs:
        f_vals          = add_frequencies(base, freqs=freqs)
        final_array     = np.append(final_array, f_vals)
    corrs               = cross_correlations(base.signal, base.return_sig)
    final_array         = np.append(final_array, corrs)
    corrs2              = cross_correlations(base.return_sig, delta_return)
    final_array         = np.append(final_array, corrs2)
    return final_array

