import numpy as np

def rejection_sampler(n_draws, f, bounds):
    """
    Rejection sampler
    
    Arguments:
        int n_draws:      number of draws
        callable f:       probability density to sample from
        iterable bounds:  upper and lower bound
        callable selfunc: selection function, must support numpy arrays
    
    Returns:
        np.ndarray: samples
    """
    n_draws = int(n_draws)
    x = np.linspace(*bounds, 1000)
    top     = np.max(f(x))
    samples = []
    while len(samples) < n_draws:
        pts   = np.random.uniform(*bounds, size = n_draws)
        probs = f(pts)
        h     = np.random.uniform(0, top, size = n_draws)
        samples.extend(pts[np.where(h < probs)])
    return np.array(samples)[:n_draws]

def log_gaussian(x, m, s):
    return -0.5*((x-m)/s)**2 - 0.5*np.log(2*np.pi) - np.log(s)

def log_gaussian_2d(x, mu, cov):
    """
    Multivariate Normal logpdf
    
    Arguments:
        np.ndarray x:   value
        np.ndarray mu:  mean vector
        np.ndarray cov: covariance matrix
    
    Returns:
        double: MultivariateNormal(m,s).logpdf(x)
    """
    inv_cov  = np.linalg.inv(cov)
    v        = x-mu
    exponent = -0.5*(inv_cov[0,0]*v[0]**2 + inv_cov[1,1]*v[1]**2 + 2*inv_cov[0,1]*v[0]*v[1])
    lognorm  = 0.5*len(mu)*LOG2PI+0.5*np.log(np.linalg.det(cov))
    return -lognorm+exponent
