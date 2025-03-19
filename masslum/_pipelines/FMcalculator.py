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
# dec: the source's declination in rad [-pi/2,pi/2]
# RA: the source's right ascenssion in rad [0,2pi]
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
    dec = pi*np.linspace(-85,85,18)/180
    RA = pi*np.linspace(5,355,36)/180


    # Create arrays to save the sigmas
    sM_TQ = np.zeros((len(dec),len(RA)))
    sD_TQ = np.zeros((len(dec),len(RA)))
    sdd_TQ = np.zeros((len(dec),len(RA)))
    sRR_TQ = np.zeros((len(dec),len(RA)))
    sdR_TQ = np.zeros((len(dec),len(RA)))
    sM_LISA = np.zeros((len(dec),len(RA)))
    sD_LISA = np.zeros((len(dec),len(RA)))
    sdd_LISA = np.zeros((len(dec),len(RA)))
    sRR_LISA = np.zeros((len(dec),len(RA)))
    sdR_LISA = np.zeros((len(dec),len(RA)))


    # Compute the SNR for different masses at different sky localizations
    # Consider the different masses
    for m in M:
        # Compute the differentiation of the waveforms
        # For the mass
        t, hp_Mp, hc_Mp = wave.HABwave(m*(1+delt),m2,S,e0,D,iota,delta,alpha0,gamma0,eta0,PHI0,f_min,f_max,n_lim,to)
        t, hp_Mm, hc_Mm = wave.HABwave(m*(1-delt),m2,S,e0,D,iota,delta,alpha0,gamma0,eta0,PHI0,f_min,f_max,n_lim,to)

        dhp_M = (hp_Mp - hp_Mm)/(2*m*delt)
        dhc_M = (hc_Mp - hc_Mm)/(2*m*delt)
        
        # For the luminosity distance
        t, hp_Dp, hc_Dp = wave.HABwave(m,m2,S,e0,D*(1+delt),iota,delta,alpha0,gamma0,eta0,PHI0,f_min,f_max,n_lim,to)
        t, hp_Dm, hc_Dm = wave.HABwave(m,m2,S,e0,D*(1-delt),iota,delta,alpha0,gamma0,eta0,PHI0,f_min,f_max,n_lim,to)

        dhp_D = (hp_Dp - hp_Dm)/(2*D*delt)
        dhc_D = (hc_Dp - hc_Dm)/(2*D*delt)


        # Generate a waveform for the calculation of sky localization
        t, hp, hc = wave.HABwave(m,m2,S,e0,D,iota,delta,alpha0,gamma0,eta0,PHI0,f_min,f_max,n_lim,to)


        # Set up an empty Fisher matrix
        F = np.zeros((4,4))

        # Consider the different sky positions
        for i in range(len(dec)):
            for j in range(len(RA)):
                # Compute the signals detected by TianQin and Fourier transform them
                # For the mass and the luminosity distance
                f, dh_M = Detection.fourier_TQ(t, dhp_M, dhc_M, dec[i], RA[j], f_min, f_max)
                f, dh_D = Detection.fourier_TQ(t, dhp_D, dhc_D, dec[i], RA[j], f_min, f_max)


                # Compute the Fourier transformed differentiation over the sky positions for TianQin
                f, h_dp = Detection.fourier_TQ(t, hp, hc, dec[i]*(1+delt), RA[j], f_min, f_max)
                f, h_dm = Detection.fourier_TQ(t, hp, hc, dec[i]*(1-delt), RA[j], f_min, f_max)
                dh_d = (h_dp - h_dm)/(2*dec[i]*delt)
                
                f, h_Rp = Detection.fourier_TQ(t, hp, hc, dec[i], RA[j]*(1+delt), f_min, f_max)
                f, h_Rm = Detection.fourier_TQ(t, hp, hc, dec[i], RA[j]*(1-delt), f_min, f_max)
                dh_R = (h_Rp - h_Rm)/(2*RA[i]*delt)


                # Compute TianQin's PSD
                psd_TQ = Detection.PSD_TQ(f)


                # Compute and safe the elements of the Fisher matrix for TianQin
                F[0][0] = Detection.inner_product(dh_M, dh_M, f, psd_TQ)
                F[0][1] = F[1][0] = Detection.inner_product(dh_M, dh_D, f, psd_TQ)
                F[0][2] = F[2][0] = Detection.inner_product(dh_M, dh_d, f, psd_TQ)
                F[0][3] = F[3][0] = Detection.inner_product(dh_M, dh_R, f, psd_TQ)
                F[1][1] = Detection.inner_product(dh_D, dh_D, f, psd_TQ)
                F[1][2] = F[2][1] = Detection.inner_product(dh_D, dh_d, f, psd_TQ)
                F[1][3] = F[3][1] = Detection.inner_product(dh_D, dh_R, f, psd_TQ)
                F[2][2] = Detection.inner_product(dh_d, dh_d, f, psd_TQ)
                F[2][3] = F[3][2] = Detection.inner_product(dh_d, dh_R, f, psd_TQ)
                F[3][3] = Detection.inner_product(dh_R, dh_R, f, psd_TQ)

                # Obtain the detection errors by inverting the Fisher matrix and save them
                Sig = inv(F)

                sM_TQ[i][j] = Sig[0][0]
                sD_TQ[i][j] = Sig[1][1]
                sdd_TQ[i][j] = Sig[2][2]
                sRR_TQ[i][j] = Sig[3][3]
                sdR_TQ[i][j] = Sig[2][3]
                
                # Compute the signals detected by LISA and Fourier transform them
                # For the mass and the luminosity distance
                f, dh_M = Detection.fourier_LISA(t, dhp_M, dhc_M, dec[i], RA[j], f_min, f_max)
                f, dh_D = Detection.fourier_LISA(t, dhp_D, dhc_D, dec[i], RA[j], f_min, f_max)


                # Compute the Fourier transformed differentiation over the sky positions for TianQin
                f, h_dp = Detection.fourier_LISA(t, hp, hc, dec[i]*(1+delt), RA[j], f_min, f_max)
                f, h_dm = Detection.fourier_LISA(t, hp, hc, dec[i]*(1-delt), RA[j], f_min, f_max)
                dh_d = (h_dp - h_dm)/(2*dec[i]*delt)

                f, h_Rp = Detection.fourier_LISA(t, hp, hc, dec[i], RA[j]*(1+delt), f_min, f_max)
                f, h_Rm = Detection.fourier_LISA(t, hp, hc, dec[i], RA[j]*(1-delt), f_min, f_max)
                dh_R = (h_Rp - h_Rm)/(2*RA[i]*delt)


                # Compute LISA's PSD
                psd_LISA = Detection.PSD_LISA(f)


                # Compute and safe the elements of the Fisher matrix for TianQin
                F[0][0] = Detection.inner_product(dh_M, dh_M, f, psd_LISA)
                F[0][1] = F[1][0] = Detection.inner_product(dh_M, dh_D, f, psd_LISA)
                F[0][2] = F[2][0] = Detection.inner_product(dh_M, dh_d, f, psd_LISA)
                F[0][3] = F[3][0] = Detection.inner_product(dh_M, dh_R, f, psd_LISA)
                F[1][1] = Detection.inner_product(dh_D, dh_D, f, psd_LISA)
                F[1][2] = F[2][1] = Detection.inner_product(dh_D, dh_d, f, psd_LISA)
                F[1][3] = F[3][1] = Detection.inner_product(dh_D, dh_R, f, psd_LISA)
                F[2][2] = Detection.inner_product(dh_d, dh_d, f, psd_LISA)
                F[2][3] = F[3][2] = Detection.inner_product(dh_d, dh_R, f, psd_LISA)
                F[3][3] = Detection.inner_product(dh_R, dh_R, f, psd_LISA)
                
                # Obtain the detection errors by inverting the Fisher matrix and save them
                Sig = inv(F)

                sMM_LISA[i][j] = Sig[0][0]
                sDD_LISA[i][j] = Sig[1][1]
                sdd_LISA[i][j] = Sig[2][2]
                sRR_LISA[i][j] = Sig[3][3]
                sdR_LISA[i][j] = Sig[2][3]

        # Save the detection accuracy in dat-files
	    # Save sigma^2 for the mass in TianQin
        with open(Path(tianqin_folder,'sigMM2_TQ_{}.dat'.format(log10(m))), 'w', newline='') as f:
	        for i in range(len(dec)):
		        writer = csv.writer(f, delimiter=' ')
		        writer.writerow(sMM_TQ[i])

    	# Save sigma^2 for the distance in TianQin
        with open(Path(tianqin_folder,'sigDD2_TQ_{}.dat'.format(log10(m))), 'w', newline='') as f:
		    for i in range(len(dec)):
			    writer = csv.writer(f, delimiter=' ')
    			writer.writerow(sDD_TQ[i])

	    # Save sigma^2 for the declination in TianQin
        with open(Path(tianqin_folder,'sigdd2_TQ_{}.dat'.format(log10(m))), 'w', newline='') as f:
	    	for i in range(len(dec)):
		    	writer = csv.writer(f, delimiter=' ')
			    writer.writerow(sdd_TQ[i])

    	# Save sigma^2 for the right ascension in TianQin
        with open(Path(tianqin_folder,'sigRR2_TQ_{}.dat'.format(log10(m))), 'w', newline='') as f:
	    	for i in range(len(dec)):
		    	writer = csv.writer(f, delimiter=' ')
			    writer.writerow(sRR_TQ[i])

    	# Save diagonal element for declination and right ascension in TianQin
        with open(Path(tianqin_folder,'sigdR_TQ_{}.dat'.format(log10(m))), 'w', newline='') as f:
	    	for i in range(len(dec)):
		    	writer = csv.writer(f, delimiter=' ')
			    writer.writerow(sdR_TQ[i])


    	# Save sigma^2 for the mass in LISA
        with open(Path(LISA_folder,'sigMM2_LISA_{}.dat'.format(log10(m))), 'w', newline='') as f:
	    	for i in range(len(dec)):
		    	writer = csv.writer(f, delimiter=' ')
			    writer.writerow(sMM_LISA[i])

    	# Save sigma^2 for the distance in LISA
        with open(Path(LISA_folder,'sigDD2_LISA_{}.dat'.format(log10(m))), 'w', newline='') as f:
	    	for i in range(len(dec)):
		    	writer = csv.writer(f, delimiter=' ')
			    writer.writerow(sDD_LISA[i])

    	# Save sigma^2 for the declination in LISA
        with open(Path(LISA_folder,'sigdd2_LISA_{}.dat'.format(log10(m)), 'w', newline='') as f:
	    	for i in range(len(dec)):
		    	writer = csv.writer(f, delimiter=' ')
			    writer.writerow(sdd_LISA[i])

    	# Save sigma^2 for the right ascension in LISA
            with open(Path(LISA_folder,'sigRR2_LISA_{}.dat'.format(log10(m))), 'w', newline='') as f:
    		for i in range(len(dec)):
	    		writer = csv.writer(f, delimiter=' ')
		    	writer.writerow(sRR_LISA[i])

    	# Save diagonal element for declination and right ascension in LISA
        with open(Path(LISA_folder,'sigdR_LISA_{}.dat'.format(log10(m))), 'w', newline='') as f:
	    	for i in range(len(dec)):
		    	writer = csv.writer(f, delimiter=' ')
			    writer.writerow(sdR_LISA[i])

if __name__ == '__main__':
    main()
