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
