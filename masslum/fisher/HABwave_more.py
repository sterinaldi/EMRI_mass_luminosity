""" Waveform model for highly asymmetric binaries with mode reduction """

# Import packages and functions need
import numpy as np

from scipy.special import jv
from numpy import sin, cos, sqrt, pi, ceil, interp

# Define the constants
G = 6.674e-11 # in m^3/(kg*s^2)
c = 2.998e8 # in m/s
Mpc = 3.086e22 # in m
M_sun = 1.989e30 # in kg


### The parameters of the source:
# m1: the mass of the big black hole in M_sun
# m2: the mass of the small black hole in M_sun
# S: the spin of the big black hole (assume the spin and angular momentum are always aligned)
# e0: the initial eccentricity of the orbit
# D: the distance of the source in Mpc
# iota: the angle between the spin and the orbital angular momentum
# delta: the angle between the angular momentum and the GW's propagation direction in rad
# alpha0: the initial angle of the first precession in rad
# gamma0: the initial angle of the second precession in rad
# eta0: the initial angle of the detector's position in rad
# PHI0: the initial phase in rad
# fo_min: the lowest frequency considered in Hz
# fo_max: the highest frequency considered in Hz
# n_lim: the maximal number of modes considered
# to: the observation time for the waveform in s

# The "Bessel" functions an, bn, cn 
def bessel(e,n):
    # Compute the semi-minor axis
    b = sqrt(1-e*e)

    # Compute the individual bessel functions of first kind
    jm2 = jv(n-2,e*n)
    jp2 = jv(n+2,e*n)
    jm1 = jv(n-1,e*n)
    jp1 = jv(n+1,e*n)

    # Compute the "Bessel" functions
    an = (jm2-jp2-2*e*jm1+2*e*jp1)/n
    bn = b*b*(jp2-jm2)/n
    cn = b*(jm2+jp2-e*jm1-e*jp1)/n

    # Return the "Bessel" functions
    return an, bn, cn


# Function to determine the lowest and highest mode that contribute significantly to observation, and the step size for the modes
def mmm(e0,f_min,f_max,fo_min,fo_max,n_lim):
    # Determine the maximal and minimal possible modes that are inside the band
    n_min = max(1,int(fo_min/f_min))
    n_max = int(fo_max/f_max)

    # Define the modes considered
    n = np.arange(n_min,n_max+1)

    # Compute the "Bessel" functions times n
    ann, bnn, cnn = n*bessel(e0,n)

    # Compute the power of the modes and find the maximum
    gn00 = ann*ann + bnn*bnn + 3*cnn*cnn - ann*bnn
    gn0 = n*n*gn00
    gn = n*n*gn0
    g_max = max(gn)

    # Determine the modes that contribute the most
    for i in range(len(gn)):
        if gn[i] > 1e-3*g_max:
            n1 = n[i]
            break

    for i in range(len(gn)):
        if gn[-i] > 1e-3*g_max:
            n2 = n[-i]
            break
    
    # Compute the stepsize for the modes
    dn = ceil((n2-n1+1)/n_lim)

    # Return n1, n2 and dn
    return n1, n2, dn


# Function to compute the initial semi-major axis (assuming low eccentricities)
def p_ini(m1,m2,e0,to):
    # Compute and return the semi-major axis
    return pow(6.4*pow(G,3)*m2*pow(m1,2)*(1+(73/24)*pow(e0,2)+(37/96)*pow(e0,4))*to/(pow(c,5)*pow(1-e0*e0,3.5)),0.25)


# The waveform generating function
def HABwave(m1,m2,S,e0,D,iota,delta,alpha0,gamma0,eta0,PHI0,fo_min,fo_max,n_lim,to):
    # Convert several quantities in SI units
    m1 = m1*M_sun
    m2 = m2*M_sun
    D = D*Mpc

    # Compute the initial semi-major axis
    p0 = p_ini(m1,m2,e0,to)

    # Compute the reduced mass, the chirp mass, and the initial orbital frequency
    mu = m2*m1/(m2+m1)
    mc = pow(m1*m2,0.6)/pow(m1+m2,0.2)
    f0 = sqrt(G*(m1+m2)/pow(p0,3))
    
    # Compute the time derivative of the frequency and compute the highest possible fundamental frequency
    f_dot = 96*pow(G*mc,5/3)*pow(f0,11/3)*(1+73*e0*e0/24+37*pow(e0,4)/96)/(5*pow(c,5)*pow(1-e0*e0,7/2))
    if f_dot*to < 1e-4:
        f_dot = 0
    f_max = f0 + f_dot*to

    # Compute the modes considered and generate an array of them
    n1, n2, dn = mmm(e0,f0,f_max,fo_min,fo_max,n_lim)
    n_tot = range(int(n1),int(n2+1),int(dn))


    # Determine the smallest sample time required and generate the final time steps (in s)
    t_sample = max(0.5/fo_max,0.5/(n_tot[-1]*f_max))
    t = np.arange(0,to,t_sample)

    # Determine the initial sample time and generate the initial adaptive time steps (in s)
    t_adap = max(0.5/(n_tot[0]*f_max),t_sample)
    ta = np.arange(0,to,t_adap)


    # Compute the time derivatives of the eccentricity and the precession angles
    e_dot = -304/15*e0*pow(G*mc,5/3)*pow(f0,8/3)*(1+121*e0*e0/304)/(pow(c,5)*pow(1-e0*e0,5/3))
    alpha_dot = 2*f0*S*(G*m1*f0/pow(c,3))/pow(1-e0*e0,1.5)
    gamma_dot = (3*f0*pow(G*m1*f0,2/3)/((c*c)*(1-e0*e0))*(1+0.25*pow(G*m1*f0,2/3)*(26-15*e0*e0)/((c*c)*(1-e0*e0))-6*f0*S*cos(iota)*pow(G*m1*f0,2/3)/(c*c)*pow(1-e0*e0,1.5)))

    # Decide if the time evolution is considered and use sparser points if necessary
    swt = False
    if -e_dot*to > 1e-3 or alpha_dot*to > 2e-2*pi or gamma_dot*to > 2e-2*pi:
        if len(t) > 100:
            th = t[::len(t)//100]
        else:
            th = t
        swt = True

    # Compute the evolution of the eccentricity
    if -e_dot*to > 1e-3:
        e = e0 + e_dot*th
    else:
        e = e0

    # Compute the evolution of the first precession angle
    if alpha_dot*to > 2e-2*pi:
        alpha = alpha0 + alpha_dot*th
    else:
        alpha = alpha0
    
    # Compute the evolution of the second precession angle
    if gamma_dot*to > 2e-2*pi:
        gamma = gamma0 + gamma_dot*th
    else:
        gamma = gamma0


    # Compute the angles
    costheta = sin(alpha)*sin(delta)*sin(iota)+cos(delta)*cos(iota)
    sinphi = ((-sin(gamma)*sin(alpha)*cos(iota)+cos(alpha)*cos(gamma))*sin(delta)+sin(iota)*cos(delta)*sin(gamma))/sqrt(1-costheta*costheta)
    cosphi = (-(sin(gamma)*cos(alpha)+sin(alpha)*cos(gamma)*cos(iota))*sin(delta)+sin(iota)*cos(gamma)*cos(delta))/sqrt(1-costheta*costheta)

    # Remove infinities if necessary    
    if hasattr(sinphi, "__len__"):
        sinphi[sinphi==np.inf] = 0
    if hasattr(cosphi, "__len__"):
        cosphi[cosphi==np.inf] = 1


    # Compute the intital frequency, the derivative of the frequency, and the frequency over time
    omega0 = f0
    if f_dot != 0:
        omega0_dot = f_dot
        omega = omega0 + omega0_dot*ta
    else:
        omega0_dot = 0
        omega = omega0

    # Compute the phase
    PHID = omega0*ta + 0.5*omega0_dot*ta*ta


    # Initiate the arrays for the polarizations and the previous adapted times steps
    hp = 0
    hc = 0
    taap = 0

    
    # Compute the amplitude of the wave and the coefficients for the amplitude
    A = -mu*pow(G,5/3)*pow((m1+m2)*omega,2/3)/(D*pow(c,4))
    A1 = cosphi*cosphi-sinphi*sinphi*costheta*costheta
    A2 = sinphi*sinphi-cosphi*cosphi*costheta*costheta
    A3 = 2*sinphi*cosphi*(1+costheta*costheta)
    A4 = 2*sinphi*cosphi*costheta
    A5 = 2*(cosphi*cosphi-sinphi*sinphi)*costheta


    # Compute the modes and sum them to the waveform
    for n in n_tot:
        # Generate the adaptive time steps
        taa = np.arange(0,to,n_tot[0]*t_adap/n)

        # Compute the "Bessel" functions
        an, bn, cn = bessel(e,n)

        # Compute the phase
        PHI = n*PHID + PHI0
        if n != n1:
            PHI = interp(taa,ta,PHI)

        # Compute the partial amplitudes and interpolate if necessary
        AA1 = A1*an + A2*bn
        if swt:
            AA1 = interp(taa,th,AA1)
        
        AA2 = A3*cn
        if swt:
            AA2 = interp(taa,th,AA2)
        
        AA3 = A4*(an-bn)
        if swt:
            AA3 = interp(taa,th,AA3)
        
        AA4 = A5*cn
        if swt:
            AA4 = interp(taa,th,AA4)


        # Compute the harmonics
        cp = cos(PHI)
        sp = sin(PHI)

        hpn = AA1*cp - AA2*sp
        hcn = AA3*cp + AA4*sp

        # Save the harmonics
        if n == n1:
            hp = hpn
            hc = hcn
        else:
            hp = interp(taa,taap,hp) + hpn
            hc = interp(taa,taap,hc) + hcn

        taap = taa
        

    # Mutiply by the amplitude of the wave
    hp = sqrt(dn)*interp(t,ta,A)*hp
    hc = sqrt(dn)*interp(t,ta,A)*hc

    # Give back the waveform and the phases of the harmonics
    return t, hp, hc
