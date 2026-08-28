'''
Signal-processing and statistical-moment estimation utilities.

This module contains tools for:
    - time-domain comparison and polynomial feature extraction
    - Fourier coefficient extraction
    - estimation of a scatter distribution's spectrum from a measured return
    - Taylor-series estimation of raw moments or cumulants
    - conversion to standardized central moments
    - assembly of features used in simulation databases

The central-moment analysis in the associated paper primarily uses the spectral-estimation pathway. 
The time-domain and correlation utilities are retained for related and ongoing signal-processing applications.
'''

import gmm
import numpy as np
import moments as mfl
import scatterers as sfl
import matplotlib.pyplot as plt

from scipy.linalg import lstsq
from scipy.special import factorial
from numpy.polynomial import Chebyshev
from scipy.optimize import minimize_scalar

# =============================================================================
# Time-domain and general signal-processing utilities
# =============================================================================

def cross_correlations(outbound, return_wave):
    '''
    create cross-correlations between two time series. returns:
       - correlation between two signals in time with no lag adjustment
       - maximum correlation between two signals across all lag adjustments
       - lag value at which maximum correlation was found
    '''
    outbound        = np.asarray(outbound)
    return_wave     = np.asarray(return_wave)
    x               = outbound - np.mean(outbound)
    y               = return_wave - np.mean(return_wave)
    full_corr       = np.correlate(x, y, mode='full')
    full_corr      /= np.linalg.norm(x) * np.linalg.norm(y)
    lags            = np.arange(-len(x)+1, len(y))
    zero_lag_corr   = full_corr[len(x)-1]
    idx_max         = np.argmax(np.abs(full_corr))
    max_corr        = full_corr[idx_max]
    best_lag        = lags[idx_max]
    return [np.abs(zero_lag_corr), np.abs(max_corr), int(best_lag)]

def poly_fit(t, return_sig, FitWindow=None, WavePeriod=5, SampleRate=1000, power=4, real=False, segments=5):
    '''
    fit piecewise Chebyshev polynomials to a selected time-domain waveform
    window of length 'WavePeriod' centered in the signal is divided into 'segments' intervals;
    each interval is fit independently through order 'power'
        - for real signals, the fitted coefficients are returned directly
        - for complex signals, the real and imaginary coefficient arrays are concatenated
    provides a description of local waveform shape independent of the Fourier-domain moment estimator
    '''
    if real:
        return_sig  = np.real(return_sig)
    if FitWindow is None:
        FitWindow   = WavePeriod / segments
    SignalLen       = len(return_sig)
    WindowLen       = int(SampleRate * FitWindow)
    WaveLen         = int(SampleRate * WavePeriod)
    coefs           = []
    for seg in range(segments):
        t1          = int((SignalLen - WaveLen)/2 + (WindowLen * seg))
        t2          = int((SignalLen - WaveLen)/2 + (WindowLen * (seg + 1)))
        fitted      = Chebyshev.fit(t[t1:t2], return_sig[t1:t2], power, domain=[t[0], t[-1]])
        coefs.extend(fitted.coef)
    coefs   = np.asarray(coefs)
    if real:
        return coefs
    return np.concatenate((np.real(coefs), np.imag(coefs)))

# =============================================================================
# Fourier-domain coefficient extraction
# =============================================================================

def fft_with_freqs(signal, t, real=False):
    '''Compute the discrete Fourier transform and its corresponding frequency grid'''
    signal  = np.asarray(signal)
    t       = np.asarray(t)
    dt      = t[1] - t[0]
    N       = len(signal)
    if real:
        f           = np.fft.rfft(np.real(signal))
        fft_freqs   = np.fft.rfftfreq(N, d=dt)
    else:
        f           = np.fft.fft(signal)
        fft_freqs   = np.fft.fftfreq(N, d=dt)
    return f, fft_freqs

def extract_return_coefs(sig, t, freqs, real=False):
    '''
    extract Fourier coefficients nearest to a requested set of frequencies
        - each requested frequency is mapped to the nearest FFT bin 
        - in the paper simulations, the interrogating frequencies are chosen 
          to coincide with the discrete spectral grid
    '''
    freqs           = np.asarray(freqs)
    f, fft_freqs    = fft_with_freqs(sig, t, real=real)
    bin_indices     = [np.argmin(np.abs(fft_freqs - f0)) for f0 in freqs]
    f_complex       = f[bin_indices]
    return f_complex, fft_freqs[bin_indices], bin_indices

def add_frequencies(base, freqs, real=False):
    '''
    returns selected Fourier coefficients as a real-valued array of form
    [Re(F_1), ..., Re(F_N), Im(F_1), ..., Im(F_N)]
    arguments:
        - base:     class object that contains the return signal
        - freqs:    array of relevant frequency values
    '''
    f_complex, _, _ = extract_return_coefs(base.return_sig, base.t, freqs, real=real)
    final_array     = np.concatenate([np.real(f_complex), np.imag(f_complex)])
    return final_array

# =============================================================================
# Spectral moment estimation
# =============================================================================

def estimate_scatter_spectrum_from_return(base, freqs, real=False, debug=False):
    '''
    estimate the scatter distribution's spectrum at the interrogating frequencies
        - for a return formed by convolution, 
              R(f) = G(f)H(f), 
          where G is the known interrogating-wave spectrum and H is the scatter distribution's spectrum, 
          the sampled scatter spectrum is estimated as
              H(f_k) = R(f_k)/G(f_k) 
          at each interrogating frequency.
    '''
    freqs   = np.asarray(freqs, dtype=float)
    R_coefs, R_freqs, R_bins    = extract_return_coefs(base.return_sig, base.t, freqs, real=real)
    G_coefs, G_freqs, G_bins    = extract_return_coefs(base.signal, base.t, freqs, real=real)
    H_coefs                     = R_coefs / G_coefs
    if debug:
        debug_vals              = {'R_coefs': R_coefs, 'R_freqs': R_freqs, 'R_bins': R_bins,
                                   'G_coefs': G_coefs, 'G_freqs': G_freqs, 'G_bins': G_bins,
                                   'H_coefs': H_coefs}
        return H_coefs, debug_vals
    return H_coefs

def prepare_spectral_fit(spectral_coefs, freqs, power=4, max_fit_order=4, dc_tol=1e-2):
    '''
    validate and prepare sampled spectral coefficients for moment estimation
        - estimator assumes that the first supplied frequency is DC 
        - frequencies are converted from cycles per unit time to angular frequency before
          constructing the Taylor expansion
    '''
    if len(spectral_coefs) != len(freqs):
        raise ValueError('"spectral_coefs" and "freqs" must be the same length.'
                        f'Got len(spectral_coefs) = {len(spectral_coefs)}, len(freqs) = {len(freqs)}.')
    if np.abs(spectral_coefs[0]) < dc_tol:
        raise ValueError('DC component is too small.' f'Got spectral_coefs[0] = {spectral_coefs[0]}.')
    if not np.isclose(freqs[0], 0.0, atol=1e-8):
        raise ValueError(f'This spectral fit assumes freqs[0] is DC.' f'Got freqs[0] = {freqs[0]}.')
    spectral_coefs  = np.asarray(spectral_coefs, dtype=complex).copy()
    freqs           = np.asarray(freqs, dtype=float)
    omegas          = 2*np.pi*freqs
    fit_order       = max_fit_order
    if len(omegas) < power:
        return spectral_coefs, omegas, fit_order, False
    return spectral_coefs, omegas, fit_order, True

def poly_design_matrix(omegas, fit_order, intercept=False):
    '''
    construct the Taylor-series design matrix for the scatter spectrum
        - for raw moments m_n, the normalized scatter spectrum is expanded as
            H(omega) = 1 + sum_n [(-i*omega)^n / n!] m_n
        - each matrix column therefore contains (-i*omega)^n / n! 
        - if 'intercept' is True, an additional constant column is included and fitted
    '''
    start_col   = int(intercept)
    A           = np.zeros((len(omegas), fit_order+start_col), dtype=complex)
    if intercept:
        A[:, 0] = 1.0
    for i in range(1, fit_order+1):
        A[:, i - 1 + start_col] = ((-1j * omegas)**i) / factorial(i)
    return A

def complex_lstsq(A, y):
    '''
    solve a complex least-squares problem subject to real-valued coefficients
        - real and imaginary parts of the complex equations are stacked before solving with lstsq
    '''
    A_set       = np.vstack([A.real, A.imag])
    y_set       = np.concatenate([y.real, y.imag])
    coefs, *_   = lstsq(A_set, y_set)
    return coefs

def estimate_common_phase(y, omegas, power=4, fit_order=4, weights=None):
    '''
    estimate and remove a frequency-independent phase from the spectrum
        - physical model assumes an approximately constant scattering phase
        - a trial phase is removed from the measured spectrum, and the phase that minimizes the spectral 
          polynomial-fit residual is selected 
        - requiring a positive fitted intercept resolves sign ambiguity
    '''
    y                       = np.asarray(y, dtype=complex)
    omegas                  = np.asarray(omegas, dtype=float)

    def phase_objective(phi):
        y_rotated           = y * np.exp(-1j * phi)
        _, _, residual, _   = fit_spectrum_poly(y_rotated, omegas, power=power, fit_order=fit_order, intercept=True,
                                                residual=True, weights=weights)
        return float(residual)

    result                  = minimize_scalar(phase_objective, bounds=(-np.pi, np.pi), method="bounded")
    phase                   = float(result.x)
    y_rotated               = y * np.exp(-1j * phase)
    _, c0                   = fit_spectrum_poly(y_rotated, omegas, power=power, fit_order=fit_order, intercept=True,
                                                residual=False, weights=weights)
    if c0 < 0:
        phase               = (phase + 2*np.pi) % (2*np.pi) - np.pi
        y_rotated           = -y_rotated
    return phase, y_rotated

def fit_spectrum_poly(y, omegas, power=4, fit_order=4, intercept=False, residual=False, fixed_c0=1.0,
                      weights=None):
    '''
    fit a truncated spectral Taylor expansion by weighted least squares
    - parameters:
        - power (int): 
            - number of fitted coefficients returned to the caller 
            - for moment estimation, these correspond to the low-order moments of interest
        - fit_order (int): 
            - total Taylor order included in the fit 
            - this may exceed 'power'; higher-order terms are then included in the regression but not returned
            - for example, a sixth-order fit may be used while retaining only the first four estimated moments
        - intercept (bool): 
            - if True, fit the constant spectral coefficient c0
            - if False, hold it fixed at 'fixed_c0'
        - weights (array_like or None):
            - least-squares weights applied to the sampled spectral frequencies
    '''
    if weights is None:
        weights = np.ones(len(omegas))
    weights     = np.asarray(weights, dtype=float)
    w           = np.sqrt(weights)
    start_col   = int(intercept)
    A0          = poly_design_matrix(omegas, fit_order, intercept=intercept)
    A           = A0 * w[:, None]
    if intercept:
        y_fit   = y
    else:
        y_fit   = y - fixed_c0
    y_fit       = y_fit * w
    coefs       = complex_lstsq(A, y_fit)
    c0          = coefs[0] if intercept else fixed_c0
    returns     = coefs[start_col:power+start_col]
    if residual:
        fit     = A0 @ coefs
        if not intercept:
            fit = fit + fixed_c0
        resid   = np.sum(weights * np.abs(y - fit)**2)
        if intercept:
            return returns, fit, resid, c0
        return returns, fit, resid
    if intercept:
        return returns, c0
    return returns

def fit_spectral_moments(spectral_coefs, freqs, power=4, residual=False, max_fit_order=4, norm='intercept',
                         log_space=False, use_weights=False, alpha=0.6, q=4, remove_phase=True):
    '''
    estimate raw moments or cumulants from sampled scatter-spectrum coefficients
        - the scatter spectrum is fit with a truncated Taylor expansion around DC
        - when 'log_space=False', fitted coefficients correspond to raw moments
        - when 'log_space=True', the logarithm of the spectrum is fit and coefficients correspond to cumulants
    - normalization modes:
        'intercept':
            - fit the DC/intercept amplitude and normalize the recovered moment coefficients by that fitted value
        'dc':
            - normalize the spectrum by its measured DC value, after which the intercept is fixed to one
        'fixed':
            - assume the spectrum has already been normalized and fix the intercept to one
    - optional frequency-weighting emphasizes the low-frequency region where the moment expansion is most accurate
    '''
    spectral_coefs, omegas, fit_order, valid = prepare_spectral_fit(spectral_coefs, freqs, power=power,
                                                                    max_fit_order=max_fit_order)
    if not valid:
        if residual:
            return np.full(power, np.nan), np.full_like(spectral_coefs, np.nan), np.nan
        return np.full(power, np.nan)
    s_coefs         = spectral_coefs
    weights         = None
    if use_weights:
        omega_max   = np.max(np.abs(omegas))
        u           = np.abs(omegas) / omega_max
        weights     = np.exp(-(u / alpha)**q)
    if remove_phase and not log_space:
        common_phase, s_coefs = estimate_common_phase(s_coefs, omegas, power=power, fit_order=fit_order, weights=weights)
    if norm == 'dc':
        s_coefs     = spectral_coefs / spectral_coefs[0]
        norm        = 'fixed'
    if norm not in ('intercept', 'fixed'):
        raise ValueError('Norm must be either "dc", "intercept", or "fixed".')
    if log_space:
        phase       = np.unwrap(np.angle(s_coefs))
        fit_coefs   = np.log(np.abs(s_coefs)) + 1j*phase
        fixed_c0    = 0.0
    else:
        fit_coefs   = s_coefs
        fixed_c0    = 1.0
    intercept       = (norm == 'intercept')
    result          =  fit_spectrum_poly(fit_coefs, omegas, power=power, fit_order=fit_order,
                                         intercept=intercept, residual=residual, fixed_c0=fixed_c0,
                                         weights=weights)
    if intercept:
        if residual:
            vals, fit, resid, c0 = result
            if not log_space:
                vals = vals / c0
            return vals, fit, resid
        vals, c0 = result
        if not log_space:
            vals = vals / c0
        return vals
    return result

# =============================================================================
# Moment conversion and spectral diagnostics
# =============================================================================

def raw_to_std_moments(raw_moments):
    '''
    convert the first four raw moments into mean, variance, std dev, skewness, and Pearson kurtosis
    '''
    raw_moments     = np.asarray(raw_moments)
    if len(raw_moments) != 4:
        raise ValueError('raw moments must be an array of four moments [m1, m2, m3, m4]')
    m1, m2, m3, m4  = raw_moments
    mean            = m1
    var             = m2 - m1**2
    std             = np.sqrt(var)
    mu3             = m3 - 3*m1*m2 + 2*m1**3
    mu4             = m4 - 4*m1*m3 + 6*m1**2*m2 - 3*m1**4
    skew            = mu3 / std**3
    kurt            = mu4 / std**4
    return np.array([mean, var, std, skew, kurt])

def cumulants_to_std_moments(cumulants, excess=False):
    '''
    convert the first four cumulants into mean, variance, std dev, skewness, and Pearson kurtosis
    '''
    cumulants       = np.asarray(cumulants)
    if len(cumulants) != 4:
        raise ValueError('cumulants must be an array of four cumulants [k1, k2, k3, k4]')
    k1, k2, k3, k4  = cumulants
    mean            = k1
    var             = k2
    std             = np.sqrt(var)
    skew            = k3 / std**3
    kurt            = k4 / std**4
    if not excess:
        kurt       += 3
    return np.array([mean, var, std, skew, kurt])

def spectrum_diagnostics(spectral_coefs, mag_floor=0.05, phase_jump_limit=1.5):
    '''
    flag spectral conditions that can make log-spectrum fitting unreliable
    - log_code:
        - 0 = no flagged condition
        - 1 = spectral magnitude below threshold
        - 2 = excessive phase jump
        - 3 = both conditions
    '''
    spectral_coefs      = np.asarray(spectral_coefs, dtype=complex)
    mag                 = np.abs(spectral_coefs)
    phase               = np.unwrap(np.angle(spectral_coefs))
    phase_jumps         = np.diff(phase)
    phi_min_abs         = np.min(mag)
    phase_max_jump      = np.max(np.abs(phase_jumps)) if len(phase_jumps) > 0 else 0.0
    small_phi           = phi_min_abs < mag_floor
    phase_bad           = phase_max_jump > phase_jump_limit
    log_code            = int(small_phi) + 2*int(phase_bad)
    log_valid           = log_code == 0
    return np.array([phi_min_abs, phase_max_jump, log_valid, log_code])

# =============================================================================
# Database feature generation
# =============================================================================

def add_spectral_moments(base, freqs, power=4, real=False, max_fit_order=4, norm='intercept',
                         from_cumulants=False, add_diagnostics=False, mag_floor=0.05, phase_jump_limit=1.5,
                         excess=False, use_weights=False, alpha=0.6, q=4, remove_phase=True):
    '''
    estimate standardized scatter-distribution moments from a simulated return
        - extracts the return and interrogating-wave Fourier coefficients
        - estimates the scatter spectrum by spectral division
        - fits either raw moments or cumulants near DC
        - converts the fitted quantities to [mean, variance, std, skewness, kurtosis]
    - optional spectral diagnostics may be appended to the returned array
    '''
    H_coefs         = estimate_scatter_spectrum_from_return(base, freqs, real=real)
    fit_vals        = fit_spectral_moments(H_coefs, freqs, power=power, max_fit_order=max_fit_order, norm=norm,
                                           log_space=from_cumulants, use_weights=use_weights, alpha=alpha, q=q,
                                           remove_phase=remove_phase)
    if from_cumulants:
        std_moments = cumulants_to_std_moments(fit_vals, excess=excess)
    else:
        std_moments = raw_to_std_moments(fit_vals)
    if add_diagnostics:
        diagnostics = spectrum_diagnostics(H_coefs, mag_floor=mag_floor, phase_jump_limit=phase_jump_limit)
        return np.append(std_moments, diagnostics)
    return std_moments

def add_metrics(row, x, freqs, base, scatterer, power=4, FitWindow=None, add_noise=False, real=False, snr=100,
                segments=3, plot=False, SampleRate=1000, add_freqs=True, add_polys=False,
                add_spectrals=True, add_corrs=True, norm='intercept', max_fit_order=4, from_cumulants=False,
                diagnostics=True, use_weights=False, alpha=0.6, q=4, remove_phase=True):
    '''
    generate a simulated return and assemble the requested analysis features
        - input database row defines either a discrete delta-scatterer distribution 
          or its matched two-component GMM 
        - selected scatter distribution is convolved with the supplied interrogating waveform and
          optional noise is added at the requested SNR
        - depending on the feature flags, the returned database row may include:
            - segmented time-domain polynomial coefficients
            - selected complex Fourier coefficients
            - estimated statistical moments from the scatter spectrum
            - spectral diagnostics
            - cross-correlation metrics
        - the ordering of these appended values must remain consistent with the column definitions in databases.py
    '''
    delta               = sfl.Delta(x, loc1=row['loc_1'], loc2=row['loc_2'], loc3=row['loc_3'],
                                    amp1=row['amp_1'], amp2=row['amp_2'], amp3=row['amp_3'])
    # create return wave from appropriate type of scatterer. if gmm, also create return wave from delta
    # so it can be used to compare to gmm return wave for correlations
    if scatterer == 'delta':
        scatter_func    = delta.scatter
        final_array     = np.array(row)
    elif scatterer == 'gmm':
        if add_corrs:
            delta_return    = base.create_return_wave(delta.scatter, plot=plot, add_noise=add_noise, snr=snr, real=real)
        results         = [row['mu1'], row['mu2'], row['sigma'], row['w1']]
        scatter_func    = gmm.gmm_pdf(x, results)
        moments         = mfl.Moments(x, scatter_func).moments
        final_array     = np.append(results, moments)
    base.return_sig     = base.create_return_wave(scatter_func, plot=plot, add_noise=add_noise, snr=snr, real=real)
    # add all other relevant fits, metrics, and tests
    if add_polys:
        poly_coefs      = poly_fit(base.t, base.return_sig, SampleRate=SampleRate, real=real,
                                   power=power, FitWindow=FitWindow, segments=segments)
        final_array     = np.append(final_array, poly_coefs)
    if add_freqs:
        f_vals          = add_frequencies(base, freqs, real=real)
        final_array     = np.append(final_array, f_vals)
    if add_spectrals:
        spectral_vals   = add_spectral_moments(base, freqs, power=power, real=real, max_fit_order=max_fit_order,
                                               norm=norm, from_cumulants=from_cumulants, add_diagnostics=diagnostics,
                                               use_weights=use_weights, alpha=alpha, q=q, remove_phase=remove_phase)
        final_array     = np.append(final_array, spectral_vals)
    if add_corrs:
        corrs           = cross_correlations(base.signal, base.return_sig)
        final_array     = np.append(final_array, corrs)
        if scatterer == 'gmm':
            corrs2      = cross_correlations(base.return_sig, delta_return)
            final_array = np.append(final_array, corrs2)
    if plot:
        plt.plot(base.t, base.signal, label='signal')
        plt.plot(base.t, base.return_sig, label='return')
        plt.legend()
        plt.show()
        plt.plot(np.abs(np.fft.fft(base.return_sig)))
        plt.show()
    return final_array