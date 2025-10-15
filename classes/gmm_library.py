#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


# In[ ]:


# creates a gmm pdf from the given parameters
def gmm_pdf(x, results):
    mu1, mu2, sigma, w1 = results
    w2                  = 1 - w1
    pdf                 = w1*norm.pdf(x, mu1, sigma) + w2*norm.pdf(x, mu2, sigma)
    pdf                /= np.sum(pdf)
    return pdf

# -------------------------------------------------------------------------------------
# APPROACH 1
# -------------------------------------------------------------------------------------

# finds the moments of a pdf
def gmm_moments(x, pdf):
    mean_gmm        = np.sum(pdf * x) / np.sum(pdf)
    variance_gmm    = np.sum(pdf * (x - mean_gmm)**2) / np.sum(pdf)
    std_gmm         = np.sqrt(variance_gmm)
    skewness_gmm    = np.sum(pdf * (x - mean_gmm)**3) / (np.sum(pdf) * (variance_gmm)**(3/2))
    kurtosis_gmm    = np.sum(pdf * (x - mean_gmm)**4) / (np.sum(pdf) * (variance_gmm)**2)
    moments         = np.array([mean_gmm, variance_gmm, std_gmm, skewness_gmm, kurtosis_gmm])
    return moments

# calculates the error between the constructed pdf's moments and the target moments
def gmm_error(params, target_moments):
    target_moments  = np.array(target_moments)
    x               = np.linspace(-2, 2, 2000)
    pdf             = gmm_pdf(x, params)
    moments         = gmm_moments(x, pdf)
    error           = 0
    for i in range(len(moments)):
        error      += (moments[i] - target_moments[i])**2
    return error, moments

# uses pdf construction to find the correct parameters that lead to the desired target moments
def fit_gmm_to_target_moments(target_moments):
    initial_guess   = [-.5, .5, .1, 0.5]
    bounds          = [(-np.inf, np.inf),   # mu1
                        (-np.inf, np.inf),  # mu2
                        (1e-5, np.inf),     # sigma
                        (0.0001, .9999)]    # w1
    result          = minimize(lambda params: gmm_error(params, target_moments)[0],
                               initial_guess, bounds=bounds, method='Nelder-Mead')
    error, moments  = gmm_error(result.x, target_moments)
    return result.x, moments, error

# -------------------------------------------------------------------------------------
# APPROACH 2
# -------------------------------------------------------------------------------------

# numerically calculates the standardized central moments rather than building a pdf first
def gmm_moments_numerical(params):
    mu1, mu2, sigma, w1 = params
    w2                  = 1 - w1
    mean                = w1*mu1 + w2*mu2
    variance            = sigma**2 + w1*mu1**2 + w2*mu2**2 - mean**2
    std                 = np.sqrt(variance)
    third_central       = 3*sigma**2*(w1*(mu1 - mean) + w2*(mu2 - mean)) + w1*(mu1 - mean)**3 + w2*(mu2 - mean)**3
    fourth_central      = (3*sigma**4 + 6*sigma**2*(w1*(mu1 - mean)**2 + w2*(mu2 - mean)**2) +
                           w1*(mu1 - mean)**4 + w2*(mu2 - mean)**4)
    skewness            = third_central / (variance**(3/2))
    kurtosis            = fourth_central / (variance**2)
    return np.array([mean, variance, std, skewness, kurtosis])

# calculates the error using numerical calculation of moments rather than pdf construction
def gmm_error2(params, target_moments):
    target_moments  = np.array(target_moments)
    moments         = gmm_moments_numerical(params)
    error           = 0
    for i in range(len(moments)):
        error      += (moments[i] - target_moments[i])**2
    return error, moments

# -------------------------------------------------------------------------------------

# finds the correct parameters that lead to the desired target moments
def match_moments(target_moments):
    initial_guess   = [-.5, .4, .1, 0.5]
    bounds          = [(-np.inf, .2),   # mu1
                        (-.2, np.inf),  # mu2
                        (1e-3, np.inf),     # sigma
                        (0.0001, .9999)]    # w1
    result          = minimize(lambda params: gmm_error2(params, target_moments)[0],
                               initial_guess, bounds=bounds, method='Nelder-Mead')
    error, moments  = gmm_error2(result.x, target_moments)
    return result.x, moments, error

