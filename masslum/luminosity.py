import numpy as np
from numba import njit
from masslum.cosmology import Planck15 as omega, dVdz_approx_planck15
from masslum.utils import rejection_sampler

# Same parameters as Gray et al (2019)
alpha = -1.07
Lsun  = 3.8*1e26
L0    = 3.0128*1e28/Lsun
Lstar = 1.2*(1e10)/(omega.h**2)
Llow  = 0.0001*Lstar
Lhigh = 3*Lstar
mth   = 19
zmax  = 0.12
gal_density = 1./10000 #gal/Mpc^3

# Mass-luminosity relation from Ding et al (2020) – https://arxiv.org/pdf/1910.11875
# log(M/10^7 Msun) = 0.49 + 0.90 log(L/10^10 Lsun)
a_ding = 0.90
b_ding = 0.49

@njit
def mass_luminosity_relation(L, a, b):
    return 10**(b + a*np.log10(L*1e-10))*1e7

@njit
def mass_luminosity_inverse_relation(M, a, b):
    return 10**((np.log10(M*1e-7)-b)/a)*1e10

def schechter_unnorm(L):
    return ((L/Lstar)**alpha)*np.exp(-L/Lstar)/Lstar

def redshift_distribution_unnorm(z):
    return omega.ComovingVolumeElement(z)/(1+z)

L_norm         = 10**(np.linspace(np.log10(Llow), np.log10(Lhigh),1001))
dL             = np.diff(L_norm)
norm_schechter = np.sum(schechter_unnorm(L_norm[:-1])*dL)

z_norm = np.linspace(0, zmax, 1001)[1:]
dz     = z_norm[1]-z_norm[0]
norm_z = np.sum(redshift_distribution_unnorm(z_norm)*dz)

@njit
def schechter(L):
    return ((L/Lstar)**alpha)*np.exp(-L/Lstar)/(Lstar*norm_schechter)

@njit
def log_schechter(L):
    return alpha*np.log(L/Lstar) -L/Lstar - np.log(Lstar) -np.log(norm_schechter)

def redshift_distribution(z):
    return omega.ComovingVolumeElement(z)/((1+z)*norm_z)

@njit
def apparent_magnitude(L, DL):
    return 25 - 2.5*np.log10(L/L0) + 5*np.log10(DL)

def selection_function(x):
    DL = omega.LuminosityDistance(x[:,1])
    m  = apparent_magnitude(x[:,0], DL)
    return m < mth

def sample_catalog(n_galaxies = 1, select = False):
    L   = rejection_sampler(n_galaxies, schechter_unnorm, [Llow,Lhigh])
    M   = mass_luminosity_relation(L, a_ding, b_ding)
    z   = rejection_sampler(n_galaxies, redshift_distribution, [0,zmax])
    dz  = z*0.05
    DL  = omega.LuminosityDistance(z)
    m   = apparent_magnitude(L, DL)
    ra  = np.random.uniform(0,2*np.pi,n_galaxies)
    dec = np.arccos(np.random.uniform(-1,1,n_galaxies)) - np.pi/2.
    cat = np.array([L, m, M, ra, dec, z, DL, dz]).T
    if select:
        return cat[m<mth]
    else:
        return cat

if __name__ == '__main__':

    from corner import corner
    from figaro import plot_settings
    
    n_evs = 1000
    ngal  = int(omega.ComovingVolume(zmax)*gal_density)
    cat   = sample_catalog(ngal)
    hosts = cat[np.random.randint(ngal, size = n_evs)]
    
    np.savetxt(f'simulated_data/catalog_mth_{mth}.txt', cat[cat[:,1]<mth], header = 'L m M ra dec z DL dz')
    np.savetxt(f'simulated_data/hosts_mth_{mth}.txt', hosts, header = 'L m M ra dec z DL dz')
    
    hosts[:,2] = np.log10(hosts[:,2])
    cat[:,2] = np.log10(cat[:,2])
    hosts[:,0] = np.log10(hosts[:,0])
    cat[:,0] = np.log10(cat[:,0])
    
    fig = corner(cat[cat[:,1]<mth][:,:-1], labels = ['$\\log\\mathrm{L}/\\mathrm{L}_\\odot$', '$m$','$\\log\\mathrm{M}/\\mathrm{M}_\\odot$', '$\\alpha$', '$\\delta$', '$z$', '$\\mathrm{D_L}\ [\\mathrm{Mpc}]$'])
    fig.savefig(f'simulated_data/observed_catalog_mth_{mth}.pdf', bbox_inches = 'tight')

    fig = corner(hosts[:,:-1], labels = ['$\\log\\mathrm{L}/\\mathrm{L}_\\odot$', '$m$','$\\log\\mathrm{M}/\\mathrm{M}_\\odot$', '$\\alpha$', '$\\delta$', '$z$', '$\\mathrm{D_L}\ [\\mathrm{Mpc}]$'])
    fig.savefig(f'simulated_data/potential_hosts_mth_{mth}.pdf', bbox_inches = 'tight')
    
    
