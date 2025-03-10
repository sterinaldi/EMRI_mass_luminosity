import numpy as np
from masslum.cosmology import Planck15 as omega, dVdz_approx_planck15
from masslum.utils import rejection_sampler

# Same parameters as Gray et al (2019)
alpha = -1.07
L0    = 3.0128*1e28
Lsun  = 3.8*1e26
Lstar = 1.2*(1e10)*Lsun/(omega.h**2)
Llow  = 0.001*Lstar
Lhigh = 10*Lstar
mth   = 19.5
zmax  = 0.4

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

def schechter(L):
    return ((L/Lstar)**alpha)*np.exp(-L/Lstar)/(Lstar*norm_schechter)

def redshift_distribution(z):
    return omega.ComovingVolumeElement(z)/((1+z)*norm_z)

def apparent_magnitude(L, DL):
    return 25 - 2.5*np.log10(L/L0) + 5*np.log10(DL)

def selection_function(x):
    DL = omega.LuminosityDistance(x[:,1])
    m  = apparent_magnitude(x[:,0], DL)
    return m < mth

def sample_catalog(n_galaxies = 1):
    L  = rejection_sampler(n_galaxies, schechter, [Llow,Lhigh])
    z  = rejection_sampler(n_galaxies, redshift_distribution, [0,zmax])
    DL = omega.LuminosityDistance(z)
    m  = apparent_magnitude(L, DL)
    return np.array([L/Lsun, z, DL, m]).T[m<mth]



if __name__ == '__main__':

    from corner import corner
    from figaro import plot_settings
    cat = sample_catalog(50000)
    
    fig = corner(cat, labels = ['$\\mathrm{L}\ [\\mathrm{L}_\\odot]$', '$z$', '$\\mathrm{D_L}\ [\\mathrm{Mpc}]$','$m$'])
    fig.savefig('full_catalog.pdf', bbox_inches = 'tight')
    fig = corner(cat[cat[:,3]<mth], labels = ['$\\mathrm{L}\ [\\mathrm{L}_\\odot]$', '$z$', '$\\mathrm{D_L}\ [\\mathrm{Mpc}]$','$m$'])
    fig.savefig('observed_catalog.pdf', bbox_inches = 'tight')
