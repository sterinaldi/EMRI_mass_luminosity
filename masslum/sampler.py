import numpy as np
from scipy.interpolate import interp1d
from scipy.special import logsumexp
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from masslum.luminosity import schechter, schechter_unnorm, log_schechter, mass_luminosity_relation, mass_luminosity_inverse_relation, apparent_magnitude, Llow, Lhigh, zmax
from masslum.utils import log_gaussian, log_gaussian_2d, rejection_sampler, create_subcatalog
from masslum.cosmology import Planck15, dVdz_approx_planck15

HUGE_NEG_NUMBER = -9999999

class Sampler():
    def __init__(self, catalog,
                       gw_mass,
                       gw_redshift,
                       gw_skypos,
                       gw_unc_mass,
                       gw_unc_redshift,
                       gw_unc_skypos,
                       m_th = 19,
                       n_MC_samples = 1e3,
                       ):
        # Galaxy catalog (L, ra, dec, z, dz)
        self.catalog = catalog
        self.m_th    = m_th
        self.z_max   = zmax
        self.D_max   = Planck15.LuminosityDistance(zmax)
        # GW posteriors
        self.gw_mass         = gw_mass
        self.gw_unc_mass     = gw_unc_mass
        self.gw_redshift     = gw_redshift
        self.gw_unc_redshift = gw_unc_redshift
        self.gw_skypos       = gw_skypos
        self.gw_unc_skypos   = gw_unc_skypos
        # Subcatalogues per event
        self.subcat    = [create_subcatalog(self.catalog, z, dz, pos, dpos) for z, dz, pos, dpos in zip(self.gw_redshift, self.gw_unc_redshift, self.gw_skypos, self.gw_unc_skypos)]
        self.log_p_gal = [self.evaluate_subcatalog(subcat, z, dz, pos, dpos) for subcat, z, dz, pos, dpos in zip(self.subcat, self.gw_redshift, self.gw_unc_redshift, self.gw_skypos, self.gw_unc_skypos)]
        # Compute in/out-catalog probability
        self.n_MC_samples = n_MC_samples
        self.DL_samples   = []
        self.z_samples    = []
        self.N_samples    = []
        self.log_norm     = self.compute_p_outcat()
        self.logp_z       = np.atleast_2d([self.compute_p_z_out(z, dz) for z, dz in zip(self.gw_redshift, self.gw_unc_redshift)]).T
        self.z_samples    = np.array(self.z_samples) # Numba required
        self.DL_samples   = np.array(self.DL_samples) # Numba required
        self.p_incat      = self.compute_p_incat()
        self.M_samples    = np.array([np.random.normal(M, dM, int(self.n_MC_samples)) for M, dM in zip(self.gw_mass, self.gw_unc_mass)])
        # Parameters
        self.names  = ['a','b']
        self.bounds = [[0.,3.],[0,2.]]
    
    def compute_p_incat(self):
        """
        MonteCarlo evaluation of the overall in-catalog probability
        """
        p = []
        L_samples  = rejection_sampler(int(self.n_MC_samples), schechter, [Llow, Lhigh])
        for dl in self.DL_samples:
            m = apparent_magnitude(L_samples, dl)
            p.append(np.sum(m < self.m_th)/len(L_samples))
        return p
        
    def log_weighted_unobs(self, L, DL, z):
        """
        Probability conditioned on NOT observing the galaxy
        """
        m    = apparent_magnitude(L, DL)
        return np.where(m > self.m_th, 0., -np.inf)

    def log_L_unobs(self, L, DL):
        """
        Probability conditioned on NOT observing the galaxy
        """
        m    = apparent_magnitude(L, DL)
        p_L  = log_schechter(L)
        return np.where(m > self.m_th, p_L, -np.inf)
    
    def compute_p_outcat(self, pts_L = 10000, pts_D = 10000):
        """
        Callable representing the normalised luminosity probability conditioned on a specific GW event being out-of-catalog
        """
        # 1D arrays and differentials
        L_grid  = 10**np.linspace(np.log10(Llow)-3, np.log10(Lhigh)+3, int(pts_L)+1)
        D_grid  = np.linspace(0, 10000, int(pts_D)+1)[1:]
        self.dL_grid = np.diff(L_grid)
        self.L_grid  = L_grid[:-1]
        dD_grid      = D_grid[1]-D_grid[0]
        # Meshgrid
        mg_L, mg_D = np.meshgrid(self.L_grid, D_grid)
        # Integral + normalisation
        log_p_L     = self.log_L_unobs(mg_L, mg_D).reshape((pts_D, pts_L))
        log_norm = logsumexp(log_p_L+np.log(self.dL_grid), axis = 1)
        return interp1d(D_grid, log_norm, bounds_error = False, fill_value = 'extrapolate')
    
    def evaluate_subcatalog(self, subcat, z, dz, pos, dpos):
        """
        Evaluate probability of each galaxy in subcatalog to be the host based on position
        """
        z_cat  = subcat[:,3]
        dz_cat = subcat[:,4]
        radec  = subcat[:,[1,2]]
        L_cat  = subcat[:,0]
        # 1/Ntot to account for marginalisation over galaxy label
        p_subcat = log_gaussian(z_cat, z, np.sqrt(dz**2)) + log_gaussian_2d(radec, pos, dpos) - np.log(dVdz_approx_planck15(z_cat)) - log_schechter(L_cat)
        return p_subcat - np.log(len(self.subcat))
    
    def log_jacobian(self, L, a, b):
        return np.log(10**(7+b-10*a)) + np.log(a) + (a-1)*np.log(L)
    
    def compute_p_z_out(self, z, dz):
        self.z_samples.append(np.random.normal(z, dz, int(self.n_MC_samples)))
        self.DL_samples.append(Planck15.LuminosityDistance(self.z_samples[-1]))
        self.N_samples.append(self.log_norm(self.DL_samples[-1]))
        return np.log(np.mean(dVdz_approx_planck15(self.z_samples[-1])))
    
    def log_likelihood(self, x):
        a, b      = x
        logL      = 0.
        Mlow      = mass_luminosity_relation(Llow, a, b)
        if np.log10(Mlow) < 4:
            return HUGE_NEG_NUMBER
        L_samples = mass_luminosity_inverse_relation(self.M_samples, a, b)
        logp_L_z  = logsumexp(np.where(L_samples > Llow, self.log_weighted_unobs(L_samples, self.DL_samples, self.z_samples) - self.N_samples, HUGE_NEG_NUMBER), axis = 1) - np.log(self.n_MC_samples)
        for i in range(len(self.gw_mass)):
            # In-catalog term
            if self.p_incat[i] > 0:
                L       = self.subcat[i][:,0]
                M       = mass_luminosity_relation(L, a, b)
                logp_M  = np.where(M > Mlow, log_gaussian(M, self.gw_mass[i], self.gw_unc_mass[i]), HUGE_NEG_NUMBER) + self.log_jacobian(L, a, b)
                logL_in = logsumexp(logp_M + self.log_p_gal[i] + np.log(self.p_incat[i]))
            else:
                logL_in = HUGE_NEG_NUMBER
            # Out-of-catalog
            logL_out = logp_L_z[i] + np.log(1-self.p_incat[i]) - np.log(4*np.pi)
            # Combination
            logL += logsumexp([logL_in, logL_out])
        return logL

    def run_mcmc(self, n_samples, burn = 2000, thin = 15, nwalkers = 10, verbose = True):
        """
        Wrapper for Eryn sampler
        """
        priors = ProbDistContainer({i: uniform_dist(*b) for i, b in enumerate(self.bounds)})
        coords = priors.rvs(size = (nwalkers,))
        sampler = EnsembleSampler(nwalkers    = nwalkers,
                                  ndims       = 2,
                                  log_like_fn = self.log_likelihood,
                                  priors      = priors,
                                  args        = [],
                                  )
        out = sampler.run_mcmc(initial_state = coords,
                               nsteps        = n_samples//nwalkers,
                               burn          = burn,
                               progress      = verbose,
                               thin_by       = thin,
                               )
        return sampler.get_chain()['model_0'].reshape(-1, 2)
