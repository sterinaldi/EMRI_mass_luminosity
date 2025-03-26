""" Code to compute the detection accuracy of the mass, the sky localization, and the luminosity distance for EMRIs with TianQin and LISA using a Fisher matrix analysis """

# Import packages and functions need
import numpy as np
import csv
import masslum.fisher.HABwave_more as wave
import masslum.fisher.Detection as Detection

from numpy import pi, sqrt, log10
from numpy.linalg import inv

from pathlib import Path

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

# Set the step size for the differentiation
delt = 1e-10

def main():

    # Create folders to save files
    tianqin_folder = Path('DATA_TianQin')
    tianqin_folder.mkdir(exist_ok = True)
    LISA_folder = Path('DATA_LISA')
    LISA_folder.mkdir(exist_ok = True)
    # Define the array of the masses considered
    M = np.logspace(5,7,5)

    # Define the arrays of the sky localizations considered
    col = pi*np.linspace(-85,85,18)/180
    lon = pi*np.linspace(5,355,36)/180


    # Create arrays to save the sigmas
    sM_TQ = np.zeros((len(col),len(lon)))
    sD_TQ = np.zeros((len(col),len(lon)))
    scc_TQ = np.zeros((len(col),len(lon)))
    sll_TQ = np.zeros((len(col),len(lon)))
    scl_TQ = np.zeros((len(col),len(lon)))
    sM_LISA = np.zeros((len(col),len(lon)))
    sD_LISA = np.zeros((len(col),len(lon)))
    scc_LISA = np.zeros((len(col),len(lon)))
    sll_LISA = np.zeros((len(col),len(lon)))
    scl_LISA = np.zeros((len(col),len(lon)))


    # Compute the SNR for different masses at different sky localizations
    # Consider the different masses
    for m in M:
        # Compute the differentiation of the waveforms for the mass
	t, hp_Mp, hc_Mp = wave.HABwave(m*(1+delt),m2,S,e0,D,iota,delta,alpha0,gamma0,eta0,PHI0,f_min,f_max,n_lim,to)
	t, hp_Mm, hc_Mm = wave.HABwave(m*(1-delt),m2,S,e0,D,iota,delta,alpha0,gamma0,eta0,PHI0,f_min,f_max,n_lim,to)

	dhp_M = (hp_Mp - hp_Mm)/(2*m*delt)
	dhc_M = (hc_Mp - hc_Mm)/(2*m*delt)

	
	# Generate a waveform for the calculation of the other differentiation
	t, hp, hc = wave.HABwave(m,m2,S,e0,D,iota,delta,alpha0,gamma0,eta0,PHI0,f_min,f_max,n_lim,to)


	# Compute the differentiation of the waveforms for the distance
	dhp_D = -hp/D
	dhc_D = -hc/D


        # Set up an empty Fisher matrix
        F = np.zeros((4,4))

        # Consider the different sky positions
        for i in range(len(col)):
            for j in range(len(lon)):
                # Compute the signals detected by TianQin and Fourier transform them
                # For the mass and the luminosity distance
                f, dh_M = Detection.fourier_TQ(t, dhp_M, dhc_M, col[i], lon[j], f_min, f_max)
                f, dh_D = Detection.fourier_TQ(t, dhp_D, dhc_D, col[i], lon[j], f_min, f_max)


                # Compute the Fourier transformed differentiation over the sky positions for TianQin
                f, h_cp = Detection.fourier_TQ(t, hp, hc, col[i]*(1+delt), lon[j], f_min, f_max)
                f, h_cm = Detection.fourier_TQ(t, hp, hc, col[i]*(1-delt), lon[j], f_min, f_max)
                dh_c = (h_cp - h_cm)/(2*col[i]*delt)
                
                f, h_lp = Detection.fourier_TQ(t, hp, hc, col[i], lon[j]*(1+delt), f_min, f_max)
                f, h_lm = Detection.fourier_TQ(t, hp, hc, col[i], lon[j]*(1-delt), f_min, f_max)
                dh_l = (h_lp - h_lm)/(2*lon[i]*delt)


                # Compute TianQin's PSD
                psd_TQ = Detection.PSD_TQ(f)


                # Compute and safe the elements of the Fisher matrix for TianQin
                F[0][0] = Detection.inner_product(dh_M, dh_M, f, psd_TQ)
                F[0][1] = F[1][0] = Detection.inner_product(dh_M, dh_D, f, psd_TQ)
                F[0][2] = F[2][0] = Detection.inner_product(dh_M, dh_c, f, psd_TQ)
                F[0][3] = F[3][0] = Detection.inner_product(dh_M, dh_l, f, psd_TQ)
                F[1][1] = Detection.inner_product(dh_D, dh_D, f, psd_TQ)
                F[1][2] = F[2][1] = Detection.inner_product(dh_D, dh_c, f, psd_TQ)
                F[1][3] = F[3][1] = Detection.inner_product(dh_D, dh_l, f, psd_TQ)
                F[2][2] = Detection.inner_product(dh_c, dh_c, f, psd_TQ)
                F[2][3] = F[3][2] = Detection.inner_product(dh_c, dh_l, f, psd_TQ)
                F[3][3] = Detection.inner_product(dh_l, dh_l, f, psd_TQ)

                # Obtain the detection errors by inverting the Fisher matrix and save them
                Sig = inv(F)

                sM_TQ[i][j] = Sig[0][0]
                sD_TQ[i][j] = Sig[1][1]
                scc_TQ[i][j] = Sig[2][2]
                sll_TQ[i][j] = Sig[3][3]
                scl_TQ[i][j] = Sig[2][3]
                
                # Compute the signals detected by LISA and Fourier transform them
                # For the mass and the luminosity distance
                f, dh_M = Detection.fourier_LISA(t, dhp_M, dhc_M, col[i], lon[j], f_min, f_max)
                f, dh_D = Detection.fourier_LISA(t, dhp_D, dhc_D, col[i], lon[j], f_min, f_max)


                # Compute the Fourier transformed differentiation over the sky positions for TianQin
                f, h_cp = Detection.fourier_LISA(t, hp, hc, col[i]*(1+delt), lon[j], f_min, f_max)
                f, h_cm = Detection.fourier_LISA(t, hp, hc, col[i]*(1-delt), lon[j], f_min, f_max)
                dh_c = (h_cp - h_cm)/(2*col[i]*delt)

                f, h_lp = Detection.fourier_LISA(t, hp, hc, col[i], lon[j]*(1+delt), f_min, f_max)
                f, h_lm = Detection.fourier_LISA(t, hp, hc, col[i], lon[j]*(1-delt), f_min, f_max)
                dh_l = (h_lp - h_lm)/(2*lon[i]*delt)


                # Compute LISA's PSD
                psd_LISA = Detection.PSD_LISA(f)


                # Compute and safe the elements of the Fisher matrix for TianQin
                F[0][0] = Detection.inner_product(dh_M, dh_M, f, psd_LISA)
                F[0][1] = F[1][0] = Detection.inner_product(dh_M, dh_D, f, psd_LISA)
                F[0][2] = F[2][0] = Detection.inner_product(dh_M, dh_c, f, psd_LISA)
                F[0][3] = F[3][0] = Detection.inner_product(dh_M, dh_l, f, psd_LISA)
                F[1][1] = Detection.inner_product(dh_D, dh_D, f, psd_LISA)
                F[1][2] = F[2][1] = Detection.inner_product(dh_D, dh_c, f, psd_LISA)
                F[1][3] = F[3][1] = Detection.inner_product(dh_D, dh_l, f, psd_LISA)
                F[2][2] = Detection.inner_product(dh_c, dh_c, f, psd_LISA)
                F[2][3] = F[3][2] = Detection.inner_product(dh_c, dh_l, f, psd_LISA)
                F[3][3] = Detection.inner_product(dh_l, dh_l, f, psd_LISA)
                
                # Obtain the detection errors by inverting the Fisher matrix and save them
                Sig = inv(F)

                sMM_LISA[i][j] = Sig[0][0]
                sDD_LISA[i][j] = Sig[1][1]
                scc_LISA[i][j] = Sig[2][2]
                sll_LISA[i][j] = Sig[3][3]
                scl_LISA[i][j] = Sig[2][3]

        # Save the detection accuracy in dat-files
	    # Save sigma^2 for the mass in TianQin
        with open(Path(tianqin_folder,'sigMM_TQ_{}.dat'.format(log10(m))), 'w', newline='') as f:
	        for i in range(len(col)):
		        writer = csv.writer(f, delimiter=' ')
		        writer.writerow(sMM_TQ[i])

    	# Save sigma^2 for the distance in TianQin
        with open(Path(tianqin_folder,'sigDD_TQ_{}.dat'.format(log10(m))), 'w', newline='') as f:
		    for i in range(len(col)):
			    writer = csv.writer(f, delimiter=' ')
    			writer.writerow(sDD_TQ[i])

	    # Save sigma^2 for the colatitude in TianQin
        with open(Path(tianqin_folder,'sigcc_TQ_{}.dat'.format(log10(m))), 'w', newline='') as f:
	    	for i in range(len(col)):
		    	writer = csv.writer(f, delimiter=' ')
			    writer.writerow(scc_TQ[i])

    	# Save sigma^2 for the longitude in TianQin
        with open(Path(tianqin_folder,'sigll_TQ_{}.dat'.format(log10(m))), 'w', newline='') as f:
	    	for i in range(len(col)):
		    	writer = csv.writer(f, delimiter=' ')
			    writer.writerow(sll_TQ[i])

    	# Save diagonal element for colatitude and longitude in TianQin
        with open(Path(tianqin_folder,'sigcl_TQ_{}.dat'.format(log10(m))), 'w', newline='') as f:
	    	for i in range(len(col)):
		    	writer = csv.writer(f, delimiter=' ')
			    writer.writerow(scl_TQ[i])


    	# Save sigma^2 for the mass in LISA
        with open(Path(LISA_folder,'sigMM_LISA_{}.dat'.format(log10(m))), 'w', newline='') as f:
	    	for i in range(len(col)):
		    	writer = csv.writer(f, delimiter=' ')
			    writer.writerow(sMM_LISA[i])

    	# Save sigma^2 for the distance in LISA
        with open(Path(LISA_folder,'sigDD_LISA_{}.dat'.format(log10(m))), 'w', newline='') as f:
	    	for i in range(len(col)):
		    	writer = csv.writer(f, delimiter=' ')
			    writer.writerow(sDD_LISA[i])

    	# Save sigma^2 for the colatitude in LISA
        with open(Path(LISA_folder,'sigcc_LISA_{}.dat'.format(log10(m)), 'w', newline='') as f:
	    	for i in range(len(col)):
		    	writer = csv.writer(f, delimiter=' ')
			    writer.writerow(scc_LISA[i])

    	# Save sigma^2 for the longitude in LISA
            with open(Path(LISA_folder,'sigll_LISA_{}.dat'.format(log10(m))), 'w', newline='') as f:
    		for i in range(len(col)):
	    		writer = csv.writer(f, delimiter=' ')
		    	writer.writerow(sll_LISA[i])

    	# Save diagonal element for colatitude and longitude in LISA
        with open(Path(LISA_folder,'sigcl_LISA_{}.dat'.format(log10(m))), 'w', newline='') as f:
	    	for i in range(len(col)):
		    	writer = csv.writer(f, delimiter=' ')
			    writer.writerow(scl_LISA[i])

if __name__ == '__main__':
    main()
