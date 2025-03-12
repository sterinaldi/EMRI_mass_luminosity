import numpy as np
import raynest
import raynest.model
from scipy.interpolate import interp1d
from scipy.special import logsumexp
from masslum.luminosity import schechter, mass_luminosity_relation, apparent_magnitude, Llow, Lhigh, z_max
from masslum.utils import log_gaussian, log_gaussian_2d, rejection_sampler, create_subcatalog
from masslum.cosmology import Planck15, dVdz_approx_planck15

def Sampler(raynest.model.Model):
    
    def __init__(self, catalog,
                       gw_mass,
                       gw_redshift,
                       gw_skypos,
                       gw_unc_mass,
                       gw_unc_redshift,
                       gw_unc_skypos,
                       m_th = 18.,
                       n_MC_samples = 1e5,
                       ):
        # Galaxy catalog (L, ra, dec, z, dz)
        self.catalog = catalog
        self.m_th    = m_th
        self.z_max   = z_max
        self.D_max   = Planck15.LuminosityDistance(z_max)
        self.z_grid  = np.linspace(0, z_max, 1000)
        self.dz_grid = self.z_grid[1]-self.z_grid[0]
        self.log_p_z = np.log(dVdz_approx_planck15(self.z_grid))
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
        # Compute in-catalog probability
        self.n_MC_samples = n_MC_samples
        self.p_incat      = self.compute_p_incat()
        # Compute interpolants for p(L|out-of-cat)
        self.p_L_out = self.compute_p_outcat()
        # RAYNEST parameters
        self.names  = ['a','b']
        self.bounds = [[0,4],[0,5]]
    
    def compute_p_incat(self, L_samples):
        """
        MonteCarlo evaluation of the overall in-catalog probability
        """
        L_samples  = rejection_sampler(int(self.n_MC_samples), schechter, [Llow, Lhigh])
        DL_samples = rejection_sampler(int(self.n_MC_samples), lambda x: x**2, [0, self.D_max])
        m          = apparent_magnitude(L_samples, DL_samples)
        return np.sum(m < self.m_th)/len(L_samples)
    
    def weighted_unobs(self, L, DL):
        """
        Probability conditioned on NOT observing the galaxy
        """
        m    = apparent_magnitude(L, DL)
        p_L  = schechter(L)
        p_DL = DL**2*3/(self.D_max**3)
        return (m > self.m_th)*p_L*p_DL
    
    def compute_p_outcat(self, pts_L = 100, pts_D = 200):
        """
        Callable representing the normalised luminosity probability conditioned on a specific GW event being out-of-catalog
        """
        # 1D arrays and differentials
        L_grid  = 10**(np.linspace(np.log10(Llow), np.log10(Lhigh), int(pts_L)+1))
        D_grid  = np.linspace(0, self.D_max, int(pts_D))
        self.dL_grid = np.diff(L_grid)
        self.L_grid  = L_grid[:-1]
        dD_grid = D_grid[1]-D_grid[0]
        # Meshgrid
        mg_L, mg_D = np.meshgrid(L_grid, D_grid)
        # Integral + normalisation
        p_L     = self.weighted_unobs(mg_L, mg_D, DL, dDL).reshape((pts_L, pts_D)).sum(axis = -1)*dD_grid
        p_L    /= np.sum(p_L*self.dL_grid)
        return interp1d(self.L_grid, p_L)
    
    def evaluate_subcatalog(self, subcat, z, dz, pos, dpos):
        """
        Evaluate probability of each galaxy in subcatalog to be the host based on position
        """
        z_cat  = subcat[:,3]
        dz_cat = subcat[:,4]
        radec  = subcat[:,[1,2]]
        # 1/Ntot to account for marginalisation over galaxy label
        p_subcat = log_gaussian(z_cat, z, np.sqrt(dz**2+dz_cat**2)) + log_gaussian_2d(radec, pos, dpos) - np.log(len(self.catalog))
        return p_subcat
    
    def log_jacobian(self, L, a, b):
        return (np.log(L) - b)/a - np.log(a*L)
    
    def log_prior(self, x):
        logP = super(Sampler,self).log_prior(x)
        if np.isfinite(logP):
            return logP
        return -np.inf
    
    def log_likelihood(self, x):
        a         = x['a']
        b         = x['b']
        logL      = 0.
        M_grid    = mass_luminosity_relation(self.L_grid, a, b)
        logJ_grid = self.log_jacobian(self.L_grid, a, b)
        p_L_out   = self.p_L_out(self.L_grid)
        for i in range(len(self.gw_mass)):
            # In-catalog term
            L       = self.subcat[i][:,0]
            M       = mass_luminosity_relation(L, a, b)
            logp_M  = log_gaussian(M, self.gw_mass[i], self.gw_unc_mass[i])
            logJ    = self.log_jacobian(L, a, b)
            logL_in = logsumexp(logp_M + logJ + self.log_p_gal[i] + np.log(self.p_incat))
            # Out-of-catalog
            logp_z   = logsumexp(log_gaussian(self.z_grid, self.gw_redshift[i], self.gw_unc_redshift[i]) + self.log_p_z + np.log(self.dz_grid))
            logp_M   = log_gaussian(M_grid, self.gw_mass[i], self.gw_unc_mass[i])
            logp_L   = logsumexp(logp_M + logJ_grid + p_L_out + np.log(self.dL_grid))
            logL_out = -np.log(4*np.pi) + logp_z + logp_L + np.log(1-self.p_incat)
            # Combination
            logL += logsumexp([logL_in, logL_out])
        return logL
