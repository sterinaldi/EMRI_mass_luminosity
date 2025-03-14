import numpy as np
import raynest
import raynest.model
from scipy.interpolate import interp1d
from scipy.special import logsumexp
from masslum.luminosity import schechter, log_schechter, mass_luminosity_relation, mass_luminosity_inverse_relation, apparent_magnitude, Llow, Lhigh, zmax
from masslum.utils import log_gaussian, log_gaussian_2d, rejection_sampler, create_subcatalog
from masslum.cosmology import Planck15, dVdz_approx_planck15

class Sampler(raynest.model.Model):
    
    def __init__(self, catalog,
                       gw_mass,
                       gw_redshift,
                       gw_skypos,
                       gw_unc_mass,
                       gw_unc_redshift,
                       gw_unc_skypos,
                       m_th = 19.5,
                       n_MC_samples = 1e4,
                       ):
        super(Sampler,self).__init__()
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
        self.logp_z       = [self.compute_p_z_out(z, dz) for z, dz in zip(self.gw_redshift, self.gw_unc_redshift)]
        self.p_L_out      = self.compute_p_outcat()
        self.p_incat      = self.compute_p_incat()
        self.M_samples    = np.array([np.random.normal(M, dM, int(self.n_MC_samples)) for M, dM in zip(self.gw_mass, self.gw_unc_mass)])
        # RAYNEST parameters
        self.names  = ['a','b']
        self.bounds = [[0.1,2],[-2,2]]
    
    def compute_p_incat(self):
        """
        MonteCarlo evaluation of the overall in-catalog probability
        """
        p = []
        L_samples  = rejection_sampler(int(self.n_MC_samples), schechter, [Llow, Lhigh])
        for dl in self.DL_samples:
            m = apparent_magnitude(L_samples, dl)
            p.append(np.sum(m < self.m_th)/len(L_samples))
        print(p)
        return np.array(p)
#        DL_samples = rejection_sampler(int(self.n_MC_samples), lambda x: x**2, [0, self.D_max])
#        m          = apparent_magnitude(L_samples, DL_samples)
#        return np.sum(m < self.m_th)/len(L_samples)
    
    def weighted_unobs(self, L, DL):
        """
        Probability conditioned on NOT observing the galaxy
        """
        m    = apparent_magnitude(L, DL)
        p_L  = schechter(L)
        p_DL = DL**2*3/(self.D_max**3)
        return (m > self.m_th)*p_L*p_DL*L
    
    def compute_p_outcat(self, pts_L = 100, pts_D = 200):
        """
        Callable representing the normalised luminosity probability conditioned on a specific GW event being out-of-catalog
        """
        # 1D arrays and differentials
        L_grid  = np.linspace(np.log10(Llow), np.log10(Lhigh), int(pts_L)+1)
#        L_grid  = np.linspace(Llow, Lhigh, int(pts_L)+1)
        D_grid  = np.linspace(0, self.D_max, int(pts_D)+1)[1:]
        self.dL_grid = np.diff(L_grid)
        self.L_grid  = L_grid[:-1]
        dD_grid      = D_grid[1]-D_grid[0]
        # Meshgrid
        mg_L, mg_D = np.meshgrid(self.L_grid, D_grid)
        # Integral + normalisation
        p_L     = self.weighted_unobs(10**mg_L, mg_D).reshape((pts_D, pts_L)).sum(axis = 0)*dD_grid
        if np.sum(p_L) > 0:
            p_L    /= np.sum(p_L*self.dL_grid)
        logp_L = np.where(p_L > 1e-15, np.log(p_L), -np.inf)
        return interp1d(self.L_grid, logp_L, bounds_error = False, fill_value = -np.inf)
    
    def evaluate_subcatalog(self, subcat, z, dz, pos, dpos):
        """
        Evaluate probability of each galaxy in subcatalog to be the host based on position
        """
        z_cat  = subcat[:,3]
        dz_cat = subcat[:,4]
        radec  = subcat[:,[1,2]]
        # 1/Ntot to account for marginalisation over galaxy label
        p_subcat = log_gaussian(z_cat, z, np.sqrt(dz**2+dz_cat**2)) + log_gaussian_2d(radec, pos, dpos)
        return p_subcat - np.log(len(self.catalog))
    
    def log_jacobian(self, L, a, b):
        return -(np.log(L) - b)/a + np.log(a*L)
    
    def compute_p_z_out(self, z, dz):
        self.z_samples  = np.random.normal(z, dz, int(self.n_MC_samples))
        self.DL_samples.append(Planck15.LuminosityDistance(self.z_samples))
        return np.log(np.mean(dVdz_approx_planck15(self.z_samples)))
    
    def log_prior(self, x):
        logP = super(Sampler,self).log_prior(x)
        if np.isfinite(logP):
            return logP
        return -np.inf
    
    def log_likelihood(self, x):
        a         = x['a']
        b         = x['b']
        logL      = 0.
#        M_grid    = mass_luminosity_relation(self.L_grid, a, b)
#        logJ_grid = self.log_jacobian(self.L_grid, a, b)
#        p_L_out   = self.p_L_out(self.L_grid)
        L_samples = np.log10(mass_luminosity_inverse_relation(self.M_samples, a, b))
        logp_L    = logsumexp(self.p_L_out(L_samples), axis = 1) - np.log(self.n_MC_samples)
        for i in range(len(self.gw_mass)):
            # In-catalog term
            L       = self.subcat[i][:,0]
            M       = mass_luminosity_relation(L, a, b)
            logp_M  = log_gaussian(M, self.gw_mass[i], self.gw_unc_mass[i])
            logJ    = self.log_jacobian(L, a, b)
            logL_in = logsumexp(logp_M + logJ + np.log(self.p_incat[i]))
            # Out-of-catalog
#            logp_M   = log_gaussian(M_grid, self.gw_mass[i], self.gw_unc_mass[i])
#            logp_L   = logsumexp(logp_M + logJ_grid + p_L_out + np.log(self.dL_grid))
            logL_out = -np.log(4*np.pi) + self.logp_z[i] + logp_L[i] + np.log(1-self.p_incat[i])
            # Combination
            logL += logsumexp([logL_in, logL_out])
        return logL
