""" Code to compute the SNR of EMRIs in TianQin and LISA """

# Import packages and functions need
import numpy as np
import csv
import HABwave_more as wave
import Detection

from numpy import pi, sqrt, log10

# Define a year
yr = 3.1536e7 # in s


### The parameters of the source: 
# m1: the mass of the big black hole in M_sun
# m2: the mass of the small black hole in M_sun
# S: the spin magnitude of the big black hole
# e0: the initial eccentricity of the orbit
# D: the distance of the source in Mpc
# iota: the angle between the spin and the orbital angular momentum
# delta: the angle between the angular momentum and the GW's propagation direction in rad
# col: the source's ecliptic colatitude in rad [-pi/2,pi/2]
# lon: the source's ecliptic longitude in rad [0,2pi]
# alpha0: the initial angle of the first precession in rad
# gamma0: the initial angle of the second precession in rad
# eta0: the initial angle of the detector's position in rad
# PHI0: the initial phase in rad
# f_min: the lowest frequency considered in Hz
# f_max: the highest frequency considered in Hz
# n_lim: the maximal number of modes considered
# to: the observation time for the waveform in s

### Set the fixed parameters:
m2 = 30
S = 0
e0 = 0.2
D = 20
iota = 0
delta = pi/6
alpha0 = 0
gamma0 = 0
eta0 = 0
PHI0 = 0
f_min = 1e-4
f_max = 10
n_lim = 100
to = 2*yr

# Define the array of the masses considered
M = np.logspace(5,7,21)

# Define the arrays of the sky localizations considered
col = pi*np.linspace(-85,85,18)/180
lon = pi*np.linspace(5,355,36)/180


# Create arrays to save the SNRs
SNR_TQ = np.zeros((len(col),len(lon)))
SNR_LISA = np.zeros((len(col),len(lon)))


# Compute the SNR for different masses at different sky localizations
# Consider the different masses
for m in M[20:]:
	# Call the waveform
	t, hp, hc = wave.HABwave(m,m2,S,e0,D,iota,delta,alpha0,gamma0,eta0,PHI0,f_min,f_max,n_lim,to)

	# Generate the on and off times for LISA
	OO_L = Detection.OO_LISA(t)

	# Consider the different sky positions
	for i in range(len(col)):
		for j in range(len(lon)):
			
			# Compute the signal detected by TianQin and Fourier transform it
			f, h = Detection.fourier_TQ(t, hp, hc, col[i], lon[j], f_min, f_max)

			# Compute TianQin's PSD
			psd_TQ = Detection.PSD_TQ(f)

			# Compute and safe the SNR in TianQin
			SNR_TQ[i][j] = sqrt(Detection.inner_product(h, h, f, psd_TQ))
			

			# Compute the signal detected by LISA and Fourier transform it
			f, h = Detection.fourier_LISA(t, hp, hc, col[i], lon[j], f_min, f_max, OO_L)

			# Compute LISA's PSD
			psd_LISA = Detection.PSD_LISA(f)

			# Compute and safe the SNR in LISA
			SNR_LISA[i][j] = sqrt(Detection.inner_product(h, h, f, psd_LISA))

	
	# Save the SNR in dat-files (create folders 'SNR_TianQin' and 'SNR_LISA' to save the files)
	# Save the SNR in TianQin
	with open('SNR_TianQin/SNR_TQ_{}.dat'.format(log10(m)), 'w', newline='') as f:
		for i in range(len(col)):
			writer = csv.writer(f, delimiter=' ')
			writer.writerow(SNR_TQ[i])
	

	# Save the SNR in LISA
	with open('SNR_LISA/SNR_LISA_{}.dat'.format(log10(m)), 'w', newline='') as f:
		for i in range(len(col)):
			writer = csv.writer(f, delimiter=' ')
			writer.writerow(SNR_LISA[i])

