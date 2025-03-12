import numpy as np
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio") # Silence LAL warnings with ipython
from scipy.optimize import newton
from scipy.interpolate import interp1d
from numba import njit
from lal._lal import LuminosityDistance, ComovingTransverseDistance, ComovingLOSDistance, HubbleDistance, HubbleParameter, UniformComovingVolumeDensity, ComovingVolumeElement, ComovingVolume, CreateCosmologicalParameters

class CosmologicalParameters:
    """
    Wrapper for LAL functions in a single class.
    From FIGARO (https://github.com/sterinaldi/FIGARO/blob/3e3e7ec86fa291a59774478e22bcb3d1bb579772/figaro/cosmology.py)
    
    Arguments:
        double h:   normalised hubble constant h = H0/100 km/Mpc/s
        double om:  matter energy density
        double ol:  cosmological constant density
        double w0:  0th order dark energy equation of state parameter
        double w1:  1st order dark energy equation of state parameter
        double w2:  2nd order dark energy equation of state parameter
        
    Returns:
        CosmologicalParameters: instance of CosmologicalParameters class
    """
    def __init__(self, h, om, ol, w0, w1, w2):
        self.h = h
        self.om = om
        self.ol = ol
        self.ok = 1.-om-ol
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self._CosmologicalParameters = CreateCosmologicalParameters(self.h, self.om, self.ol, self.w0, self.w1, self.w2)
        self.HubbleDistance = HubbleDistance(self._CosmologicalParameters)
        
    def _vectorise(func):
        def vectorised_func(self, x):
            if hasattr(x, "__iter__"):
                return np.array([func(self, xi) for xi in x])
            else:
                return func(self, x)
        return vectorised_func
    
    @_vectorise
    def HubbleParameter(self, z):
        return HubbleParameter(z, self._CosmologicalParameters)

    @_vectorise
    def LuminosityDistance(self, z):
        return LuminosityDistance(self._CosmologicalParameters, z)
    
    @_vectorise
    def ComovingTransverseDistance(self, z):
        return ComovingTransverseDistance(self._CosmologicalParameters, z)
     
    @_vectorise
    def ComovingLOSDistance(self, z):
        return ComovingLOSDistance(self._CosmologicalParameters, z)
     
    @_vectorise
    def UniformComovingVolumeDensity(self, z):
        return UniformComovingVolumeDensity(z, self._CosmologicalParameters)
    
    @_vectorise
    def ComovingVolumeElement(self, z):
        return ComovingVolumeElement(z, self._CosmologicalParameters)
    
    @_vectorise
    def ComovingVolume(self, z):
        return ComovingVolume(self._CosmologicalParameters, z)
    
    @_vectorise
    def dDTdDC(self, DC):
        if self.ok == 0.:
            return 1.
        elif self.ok > 0.:
            return np.cosh(np.sqrt(self.ok)*DC/self.HubbleDistance)
        else:
            return np.cos(np.sqrt(-self.ok)*DC/self.HubbleDistance)
    
    @_vectorise
    def dDLdz(self, z):
        DC   = self.ComovingLOSDistance(z)
        DT   = self.ComovingTransverseDistance(z)
        invE = self.HubbleParameter(z)
        return DT + (1.+z)*self.dDTdDC(DC)*self.HubbleDistance*invE
    
    @_vectorise
    def Redshift(self, DL):
        if DL == 0.:
            return 0.
        else:
            def objective(z, self, DL):
                return DL - self.LuminosityDistance(z)
            return newton(objective,1.0,args=(self, DL))

Planck18 = CosmologicalParameters(0.674, 0.315, 0.685, -1, 0, 0)
Planck15 = CosmologicalParameters(0.679, 0.3065, 0.6935, -1, 0, 0)

# Interpolants up to z = 2.5
z = np.linspace(0,0.5,1000)
dvdz_planck18 = Planck18.ComovingVolumeElement(z)/(1e9*(1+z)) # In Gpc^3
dvdz_planck15 = Planck15.ComovingVolumeElement(z)/(1e9*(1+z)) # In Gpc^3

@njit
def dVdz_approx_planck15(x):
    return np.interp(x, z/np.sum(dvdz_planck15*(z[1]-z[0])), dvdz_planck15)

@njit
def dVdz_approx_planck18(x):
    return np.interp(x, z/np.sum(dvdz_planck18*(z[1]-z[0])), dvdz_planck18)
