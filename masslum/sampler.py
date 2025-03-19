import numpy as np
import raynest
import raynest.model
from scipy.interpolate import interp1d
from scipy.special import logsumexp
from masslum.luminosity import schechter, schechter_unnorm, log_schechter, mass_luminosity_relation, mass_luminosity_inverse_relation, apparent_magnitude, Llow, Lhigh, zmax
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
                       m_th = 19,
                       n_MC_samples = 1e3,
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
        self.z_samples    = []
        self.N_samples    = []
        self.log_norm     = self.compute_p_outcat()
        self.logp_z       = np.atleast_2d([self.compute_p_z_out(z, dz) for z, dz in zip(self.gw_redshift, self.gw_unc_redshift)]).T
        self.z_samples    = np.array(self.z_samples) # Numba required
        self.DL_samples   = np.array(self.DL_samples) # Numba required
        self.p_incat      = self.compute_p_incat()
        self.M_samples    = np.array([np.random.normal(M, dM, int(self.n_MC_samples)) for M, dM in zip(self.gw_mass, self.gw_unc_mass)])
        # RAYNEST parameters
        self.names  = ['a','b']
        self.bounds = [[0,3],[0,3]]
    
    def compute_p_incat(self):
        """
        MonteCarlo evaluation of the overall in-catalog probability
        """
        p = []
        L_samples  = rejection_sampler(int(self.n_MC_samples), schechter_unnorm, [Llow, Lhigh])
        for dl in self.DL_samples:
            m = apparent_magnitude(L_samples, dl)
            p.append(np.sum(m < self.m_th)/len(L_samples))
        print(p)
        return np.array(p)
    
    def weighted_unobs(self, L, DL):
        """
        Probability conditioned on NOT observing the galaxy
        """
        m    = apparent_magnitude(L, DL)
        p_L  = schechter(L)
        p_DL = DL**2*3/(self.D_max**3)
        return (m > self.m_th)*p_L*p_DL*L
        
    def log_weighted_unobs(self, L, DL, z):
        """
        Probability conditioned on NOT observing the galaxy
        """
        m    = apparent_magnitude(L, DL)
        p_L  = log_schechter(L)
        p_DL = np.log(dVdz_approx_planck15(z))
        return np.log(m > self.m_th) + p_L + p_DL

    def log_L_unobs(self, L, DL):
        """
        Probability conditioned on NOT observing the galaxy
        """
        m    = apparent_magnitude(L, DL)
        p_L  = log_schechter(L)
        return np.log(m > self.m_th) + p_L
    
    def compute_p_outcat(self, pts_L = 1000, pts_D = 1000):
        """
        Callable representing the normalised luminosity probability conditioned on a specific GW event being out-of-catalog
        """
        # 1D arrays and differentials
        L_grid  = 10**np.linspace(np.log10(Llow), np.log10(Lhigh), int(pts_L)+1)
#        L_grid  = np.linspace(Llow, Lhigh, int(pts_L)+1)
        D_grid  = np.linspace(0, self.D_max, int(pts_D)+1)[1:]
        self.dL_grid = np.diff(L_grid)
        self.L_grid  = L_grid[:-1]
        dD_grid      = D_grid[1]-D_grid[0]
        # Meshgrid
        mg_L, mg_D = np.meshgrid(self.L_grid, D_grid)
        # Integral + normalisation
        log_p_L     = self.log_L_unobs(mg_L, mg_D).reshape((pts_D, pts_L))
        log_norm = logsumexp(log_p_L+np.log(self.dL_grid), axis = 1)
        return interp1d(D_grid, log_norm, bounds_error = False, fill_value = np.inf)
    
    def evaluate_subcatalog(self, subcat, z, dz, pos, dpos):
        """
        Evaluate probability of each galaxy in subcatalog to be the host based on position
        """
        z_cat  = subcat[:,3]
        dz_cat = subcat[:,4]
        radec  = subcat[:,[1,2]]
        # 1/Ntot to account for marginalisation over galaxy label
        p_subcat = log_gaussian(z_cat, z, np.sqrt(dz**2+dz_cat**2)) + log_gaussian_2d(radec, pos, dpos)
        return p_subcat - np.log(len(self.subcat))
    
    def log_jacobian(self, L, a, b):
        return -(np.log(L) - b)/a + np.log(a*L)
    
    def compute_p_z_out(self, z, dz):
        self.z_samples.append(np.random.normal(z, dz, int(self.n_MC_samples)))
        self.DL_samples.append(Planck15.LuminosityDistance(self.z_samples[-1]))
        self.N_samples.append(self.log_norm(self.DL_samples[-1]))
        return np.log(np.mean(dVdz_approx_planck15(self.z_samples[-1])))
    
    def log_prior(self, x):
        logP = super(Sampler,self).log_prior(x)
        if np.isfinite(logP):
            return logP
        return -np.inf
    
    def log_likelihood(self, x):
        a         = x['a']
        b         = x['b']
        logL      = 0.
        L_samples = mass_luminosity_inverse_relation(self.M_samples, a, b)
        logp_L_z  = logsumexp(self.log_weighted_unobs(L_samples, self.DL_samples, self.z_samples) - self.N_samples, axis = 1) - np.log(self.n_MC_samples)
        for i in range(len(self.gw_mass)):
            # In-catalog term
            if self.p_incat[i] > 0:
                L       = self.subcat[i][:,0]
                M       = mass_luminosity_relation(L, a, b)
                logp_M  = log_gaussian(M, self.gw_mass[i], self.gw_unc_mass[i])
                logL_in = logsumexp(logp_M + self.log_p_gal[i] + np.log(self.p_incat[i]))
            else:
                logL_in = -np.inf
            # Out-of-catalog
            logL_out = -np.log(4*np.pi) + logp_L_z[i] + np.log(1-self.p_incat[i])
            # Combination
#            print([logL_in, logL_out])
            logL += logsumexp([logL_in, logL_out])
        return logL
