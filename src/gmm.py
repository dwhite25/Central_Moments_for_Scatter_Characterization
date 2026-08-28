import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# creates a gmm pdf from the given parameters
def gmm_pdf(x, results):
    mu1, mu2, sigma, w1 = results
    w2                  = 1 - w1
    pdf                 = w1*norm.pdf(x, mu1, sigma) + w2*norm.pdf(x, mu2, sigma)
    pdf                /= np.sum(pdf)
    return pdf

# analytically calculates moment parameters from the given parameters of an equal-variance, 
# two-component GMM
def gmm_moments(params):
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

# calculates the error using analytic calculation of moments
def gmm_error(params, target_moments):
    target_moments  = np.array(target_moments)
    moments         = gmm_moments(params)
    error           = np.sum((moments - target_moments)**2)
    return error, moments

# finds the correct parameters that lead to the desired target moments
# Constrain mu1 to the left component and mu2 to the right component
# means for each record in the database span approximately [-0.4, 0.4], so allowing the GMM
# component means to cross slightly through the center provides sufficient coverage of the parameter space
def match_moments(target_moments):
    initial_guess   = [-.5, .4, .1, 0.5]
    bounds          = [(-np.inf, .2),       # mu1
                        (-.2, np.inf),      # mu2
                        (1e-3, np.inf),     # sigma
                        (0.0001, .9999)]    # w1
    result          = minimize(lambda params: gmm_error(params, target_moments)[0],
                               initial_guess, bounds=bounds, method='Nelder-Mead')
    error, moments  = gmm_error(result.x, target_moments)
    return result.x, moments, error