""" Functions for detection with TianQin and LISA """

# Import packages and functions need
import numpy as np
import random as rdm

from numpy import sin, cos, sqrt, pi, conj, exp, arctan2, log2, ceil, where
from scipy.fft import fft, fftfreq

# Define the speed of light
c = 2.998e8 # in m/s


### The parameters used
# f: the frequency of the wave/in the detector in Hz
# f_min: the mininmal frequency observed in Hz
# f_max: the maximal frequency observed in Hz
# qS: the source's polar angles in rad [0,pi]
# phiS: the source's azimuthal angle in rad [0,2pi]
# qK: the spin's polar angle in rad [0,pi]
# phiK: the spin's azimuthal anglein rad [0,pi]
# col: the source's ecliptic colatitude in rad [-pi/2,pi/2]
# lon: the source's ecliptic longitude in rad [0,2pi]
# time: the time steps of the waveform considred in s
# h, h1, h2: the strain of the waveforms considred
# hp, hc: the plus and cross polarizations of the waveform
# PSD: the power spectrum denisity of the detectors considered


# The PSD of TianQin
def PSD_TQ(f):
    # Define the length and reference frequency
    L_TQ = sqrt(3)*1e8 # in m
    f_TQ = c/(2*pi*L_TQ) # in Hz

    # Define the noise parameters
    P_x = 1e-24
    P_a = 1e-30

    # Return the PSD
    return sqrt((10/(3*L_TQ*L_TQ))*(P_x + 4*P_a/pow(2*pi*f,4)*(1 + 0.0001/f))*(1 + 0.6*pow(f/f_TQ,2)))


# The PSD of LISA
def PSD_LISA(f):
    # Define the length and reference frequency
    L_LISA = 2.5e9 # in m
    f_LISA = c/(2*pi*L_LISA) # in Hz

    # Define the noise curves
    P_oms = 2.25e-22*(1 + 1.6e-11/pow(f,4))
    P_acc = 9e-30*(1 + 1.6e-7/(f*f))*(1 + (pow(f,4))/4096)

    # Define the foreground
    FG = sqrt(9e-45/pow(f,7/3)*np.exp(-pow(f,0.138)-221*f*sin(521*f))*(1 + np.tanh(1680*(1.13e-3-f))))

    # Return the PSD
    return sqrt((10/(3*L_LISA*L_LISA))*(P_oms + 2*P_acc/pow(2*pi*f,4)*(1 + pow(cos(f/f_LISA),2)))*(1 + 0.6*pow(f/f_LISA,2))) + FG


# The response function of TianQin
def F_TQ(time,qS,phiS,qK,phiK):
    '''https://arxiv.org/pdf/1901.02159.pdf Eq. 2-4''' 

    #  beta, lamda are the source’s ecliptic colatitude and longitude
    beta = qS - pi/2
    lamda = phiS
    
    q_j0806 = 1.65
    phi_j0806 = 2.10
    f_sc = 1/(3.65*24*3600) 
    k = 2*pi*f_sc*time

    cos2k = cos(2*k)
    sin2k = sin(2*k)
    cos2beta = cos(2*beta)
    sin2beta = sin(2*beta)
    cosbeta = cos(beta)
    sinbeta = sin(beta)
    
    cosbeta_j0806 = cos(q_j0806)
    sinbeta_j0806 = sin(q_j0806)
    cos2beta_j0806 = cos(2*q_j0806)
    sin2beta_j0806 = sin(2*q_j0806)
    
    sin2lamda_2phi_j0806 = sin(2*lamda - 2*phi_j0806)
    cos2lamda_2phi_j0806 = cos(2*lamda - 2*phi_j0806)

    sinlamda_phi_j0806 = sin(lamda - phi_j0806)
    coslamda_phi_j0806 = cos(lamda - phi_j0806)
   

    D_plus = (sqrt(3)/ 32)*(4*cos2k*((3+cos2beta)*cosbeta_j0806*sin2lamda_2phi_j0806 +     2*sinlamda_phi_j0806*sin2beta*sinbeta_j0806) - sin2k*(3 +                 cos2lamda_2phi_j0806*(9+cos2beta*(3+cos2beta_j0806)) -6*cos2beta_j0806*sinlamda_phi_j0806*sinlamda_phi_j0806 - 6*cos2beta*sinbeta_j0806*sinbeta_j0806 + 4*coslamda_phi_j0806*sin2beta*sin2beta_j0806))

    D_cross = (sqrt(3)/8)*(-4*cos2k*(cos2lamda_2phi_j0806*cosbeta*cosbeta_j0806 + coslamda_phi_j0806*sinbeta*sinbeta_j0806) + sin2k *(-cosbeta*(3+sin2beta_j0806)*sin2lamda_2phi_j0806 -2*sinlamda_phi_j0806*sinbeta*sin2beta_j0806))

    
    """Get the psi"""
    '''https://arxiv.org/pdf/2104.04582.pdf eq.8'''
    # the source coordinate System in spin frame. 

    cqS = cos(qS)
    sqS = sin(qS)

    cphiS = cos(phiS)
    sphiS = sin(phiS)

    cqK = cos(qK)
    sqK = sin(qK)

    cphiK = cos(phiK)
    sphiK = sin(phiK)

    # get the polarization angle

    up_ldc = cqS*sqK*cos(phiS - phiK) - cqK*sqS
    dw_ldc = sqK*sin(phiS - phiK)

    if dw_ldc != 0.0:
        psi_ldc = -np.arctan2(up_ldc, dw_ldc)
    else:
        psi_ldc = 0.5 * pi

    # Compute the response functions and return them
    F_plus = D_plus*cos(psi_ldc) - D_cross*sin(psi_ldc)
    F_cross = D_plus*sin(psi_ldc) + D_cross*cos(psi_ldc)

    return F_plus, F_cross


# The response function of LISA
def F_LISA(time,qS,phiS,qK,phiK):
    # Compute the position of the source
    cosqS = cos(qS)
    sinqS = sin(qS)
    cosqK = cos(qK)
    sinqK = sin(qK)
    cosphiK = cos(phiK)
    sinphiK = sin(phiK)
    halfsqrt3 = sqrt(3)/2

    # Compute the evolutionf of LISA's orbit
    orbphs = 2*pi*time/31536000

    cosorbphs = cos(orbphs-phiS)
    sinorbphs = sin(orbphs-phiS)
    cosq = 0.5*cosqS - halfsqrt3*sinqS*cosorbphs
    phiw = orbphs + np.arctan2(halfsqrt3*cosqS+0.5*sinqS*cosorbphs, sinqS*sinorbphs)

    # Compute psi    
    psiup = 0.5*cosqK - halfsqrt3*sinqK*cos(orbphs-phiK) - cosq*(cosqK*cosqS + sinqK*sinqS*cos(phiK-phiS))
    psidown = 0.5*sinqK*sinqS*sin(phiK-phiS) - halfsqrt3*cos(orbphs)*(cosqK*sinqS*sin(phiS)-cosqS*sinqK*sin(phiK)) - halfsqrt3*sin(orbphs)*(cosqS*sinqK*cos(phiK)-cosqK*sinqS*cos(phiS))
    psi = arctan2(psiup,psidown)

    cosq1  = 0.5*(1+cosq*cosq)
    cos2phi = cos(2*phiw)
    sin2phi = sin(2*phiw)
    cos2psi = cos(2*psi)
    sin2psi = sin(2*psi)

    # Compute the response functions and return them
    F_plus = cosq1*cos2phi*cos2psi-cosq*sin2phi*sin2psi
    F_cross = cosq1*cos2phi*sin2psi+cosq*sin2phi*cos2psi

    return F_plus, F_cross


# A function for the on-off time of TianQin
def OO_TQ(time):
    # Set the polar coordinate of RX J0806.3+1527
    phi_j0806 = 2.10

    # Compute the orbital phase modulo pi of TianQin around the sun
    f = 1/(365*24*3600) 
    pha = 2*pi*f*time % pi

    # Set the on and off times (3-3-months)
    oo1 = np.where((pha > phi_j0806-pi/4) & (pha < phi_j0806+pi/4), 1, 0)
    oo2 = np.where((pha > phi_j0806+3*pi/4) & (pha < phi_j0806+5*pi/4), 1, 0)

    # Return the on and off times
    return oo1 + oo2


# A function for the on-off time of LISA (assuming 85 % average on time)
def OO_LISA(time):
    # Create an array to save the times
    oo = np.zeros(len(time))

    # Randomly determine the on times
    for i in range(len(time)//100-1):
        if rdm.random() <= 0.85:
            oo[100*i:100*(i+1)] = 1

    # Return the array with on-off times
    return oo


# A function for zero-padding
def padding(time, h, f_min):
    # Save the original length of the waveform and determine the next power-of-two length for three times the original length
    olen = len(h)
    n = ceil(log2(1.25*olen))

    # Forward padding
    h = np.pad(h, (len(h)//8, 0), mode='constant')
    start = (time[0]-time[1])*(len(time)//8) + time[0]
    time = np.pad(time, (len(time)//8, 0), mode='linear_ramp', end_values=(start))

    # Add a smoothing function at the start
    if time[olen//8] != 0:
        print('WARNING: original time does not start with zero as assumed. Smoothing at the start not performed!')
    else:
        for i in range(olen//8-1,-1,-1):
            h[i] = h[olen//8]*exp(f_min*time[i]/32)*cos(f_min*time[i]/2)

    # Backward padding
    h = np.pad(h, (0, int(pow(2,n)-len(h))), mode='constant')
    end = (time[-1]-time[-2])*int(pow(2,n)-len(time)) + time[-1]
    time = np.pad(time, (0, int(pow(2,n)-len(time))), mode='linear_ramp', end_values=(end))

    # Add a smoothing function at the end
    if time[olen+olen//8] <= 0:
        print('WARNING: original time at the end is not positive as assumed. Smoothing at the end not performed!')
    else:
        for i in range(olen+olen//8,len(h)):
            h[i] = h[olen+olen//8-1]*exp(-f_min*(time[i]-time[olen+olen//8])/32)*cos(f_min*(time[i]-time[olen+olen//8])/2)

    return time, h


# A function to Fourier transform the detected signal in TianQin
def fourier_TQ(time, hp, hc, col, lon, f_min, f_max):
    # Generate the response pattern for TianQin (spin aligned with z-axis)
    F_plus, F_cross = F_TQ(time,col+pi/2,lon,0,0)

    # Generate the waveform detected
    h = (hp*F_plus + hc*F_cross)*OO_TQ(time)

    # Window the data (commented out as it is slow)
    #time, h = padding(time,h,f_min)
    
    # Fourier transform the wave and time
    f_help = fftfreq(time.shape[-1], d=abs(time[1]-time[0]))
    h_help = fft(h).real

    # Return the frequencies and strain in the detectable range
    f = f_help[where((f_help > f_min) & (f_help < f_max))]
    h = h_help[where((f_help > f_min) & (f_help < f_max))]
    
    return f, h


# A function to Fourier transform the detected signal in LISA
def fourier_LISA(time, hp, hc, col, lon, f_min, f_max):
    # Generate the response pattern for TianQin (spin aligned with z-axis)
    F_plus, F_cross = F_LISA(time,col+pi/2,lon,0,0)

    # Generate the waveform detected
    h = (hp*F_plus + hc*F_cross)*OO_LISA(time)

    # Window the data (commented out as it is slow)
    #time, h = padding(time,h,f_min)
    
    # Fourier transform the wave and time
    f_help = fftfreq(time.shape[-1], d=abs(time[1]-time[0]))
    h_help = fft(h).real

    # Return the frequencies and strain in the detectable range
    f = f_help[where((f_help > f_min) & (f_help < f_max))]
    h = h_help[where((f_help > f_min) & (f_help < f_max))]

    return f, h    


# The inner product
def inner_product(h1, h2, f, PSD):
    # Check if the length of the arrays fits
    if len(h1) != len(h2) or len(h2) != len(f) or len(f) != len(PSD):
        print('Problem: Arrays have different lengths.')
        return 0

    # Check if there are negative frequencies; write a warning if so and return zero
    if np.any(f < 0):
        print('Problem: Negative frequencies were passed.')
        return 0

    # Compute S_n
    S_n = pow(PSD,2)

    # Define the integrand
    integrand = h1*conj(h2)/S_n

    # Compute and return the inner product (integral)
    return 2*np.trapz(integrand,f).real
