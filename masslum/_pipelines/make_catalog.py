import numpy as np
import matplotlib.pyplot as plt
import optparse
import os
import re
from pathlib import Path
from corner import corner
from scipy.interpolate import RegularGridInterpolator
from masslum.luminosity import gal_density, zmax, sample_catalog, mth, a_ding, b_ding, mass_luminosity_relation
from masslum.cosmology import Planck15 as omega

this_folder = Path(f'{os.path.dirname(__file__)}')

dec_grid = np.arange(-85,90,10)
ra_grid  = np.arange(5,360,10)

def get_number(s):
    return float(re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)[0])

def load_file(s):
    return RegularGridInterpolator((dec_grid, ra_grid), np.genfromtxt(s))

def compute_SNR(m1, ra, dec, DL):
    grids_LISA = {get_number(s.as_posix()): load_file(s) for s in Path(this_folder, 'SNR_LISA').glob('*.dat')}
    grids_TQ   = {get_number(s.as_posix()): load_file(s) for s in Path(this_folder, 'SNR_TianQin').glob('*.dat')}
    
    keys = np.array([k for k in grids_LISA.keys()])
    
    log10m  = np.log10(m1)
    ra_deg  = ra/np.pi * 180
    dec_deg = dec/np.pi * 180
    
    mmin    = np.min(keys)
    mmax    = np.max(keys)
    idx     = (log10m > mmin) & (log10m < mmax) & (ra_deg > 5) & (ra_deg < 355) & (dec_deg > -85) & (dec_deg < 85)
    mass_id = np.round(log10m, 1)
    
    # 0 for out of interpolation range, 20/DL gives the rescaling
    SNR_LISA = np.array([grids_LISA[mid]((deci, rai)) if id else 0. for mid, rai, deci, id in zip(mass_id, ra_deg, dec_deg, idx)]) * 20./DL
    SNR_TQ   = np.array([grids_TQ[mid]((deci, rai)) if id else 0. for mid, rai, deci, id in zip(mass_id, ra_deg, dec_deg, idx)]) * 20./DL
    
    return SNR_LISA, SNR_TQ

def compute_error(m1, ra, dec, DL):

    grids_sigMM_LISA = {get_number(s.as_posix()): load_file(s) for s in Path(this_folder, 'FM_LISA').glob('sigMM*.dat')}
    grids_sigDD_LISA = {get_number(s.as_posix()): load_file(s) for s in Path(this_folder, 'FM_LISA').glob('sigDD*.dat')}
    grids_sigcc_LISA = {get_number(s.as_posix()): load_file(s) for s in Path(this_folder, 'FM_LISA').glob('sigcc*.dat')}
    grids_sigll_LISA = {get_number(s.as_posix()): load_file(s) for s in Path(this_folder, 'FM_LISA').glob('sigll*.dat')}
    grids_sigcl_LISA = {get_number(s.as_posix()): load_file(s) for s in Path(this_folder, 'FM_LISA').glob('sigcl*.dat')}
    grids_sigMM_TQ   = {get_number(s.as_posix()): load_file(s) for s in Path(this_folder, 'FM_TianQin').glob('sigMM*.dat')}
    grids_sigDD_TQ   = {get_number(s.as_posix()): load_file(s) for s in Path(this_folder, 'FM_TianQin').glob('sigDD*.dat')}
    grids_sigcc_TQ   = {get_number(s.as_posix()): load_file(s) for s in Path(this_folder, 'FM_TianQin').glob('sigcc*.dat')}
    grids_sigll_TQ   = {get_number(s.as_posix()): load_file(s) for s in Path(this_folder, 'FM_TianQin').glob('sigll*.dat')}
    grids_sigcl_TQ   = {get_number(s.as_posix()): load_file(s) for s in Path(this_folder, 'FM_TianQin').glob('sigcl*.dat')}
    
    keys = np.array([k for k in grids_sigMM_LISA.keys()])
    
    log10m  = np.log10(m1)
    ra_deg  = ra/np.pi * 180
    dec_deg = dec/np.pi * 180
    
    mmin    = np.min(keys)
    mmax    = np.max(keys)
    idx     = (log10m > mmin) & (log10m < mmax) & (ra_deg > 5) & (ra_deg < 355) & (dec_deg > -85) & (dec_deg < 85)
    mass_id = np.round(log10m, 1)
    
    sigMM_LISA = np.array([grids_sigMM_LISA[mid]((deci, rai)) if id else 0. for mid, rai, deci, id in zip(mass_id, ra_deg, dec_deg, idx)]) * 20./DL
    sigDD_LISA = np.array([grids_sigDD_LISA[mid]((deci, rai)) if id else 0. for mid, rai, deci, id in zip(mass_id, ra_deg, dec_deg, idx)]) * 20./DL
    sigcc_LISA = np.array([grids_sigcc_LISA[mid]((deci, rai)) if id else 0. for mid, rai, deci, id in zip(mass_id, ra_deg, dec_deg, idx)]) * 20./DL
    sigll_LISA = np.array([grids_sigll_LISA[mid]((deci, rai)) if id else 0. for mid, rai, deci, id in zip(mass_id, ra_deg, dec_deg, idx)]) * 20./DL
    sigcl_LISA = np.array([grids_sigcl_LISA[mid]((deci, rai)) if id else 0. for mid, rai, deci, id in zip(mass_id, ra_deg, dec_deg, idx)]) * 20./DL
    sigMM_TQ   = np.array([grids_sigMM_TQ[mid]((deci, rai)) if id else 0. for mid, rai, deci, id in zip(mass_id, ra_deg, dec_deg, idx)]) * 20./DL
    sigDD_TQ   = np.array([grids_sigDD_TQ[mid]((deci, rai)) if id else 0. for mid, rai, deci, id in zip(mass_id, ra_deg, dec_deg, idx)]) * 20./DL
    sigcc_TQ   = np.array([grids_sigcc_TQ[mid]((deci, rai)) if id else 0. for mid, rai, deci, id in zip(mass_id, ra_deg, dec_deg, idx)]) * 20./DL
    sigll_TQ   = np.array([grids_sigll_TQ[mid]((deci, rai)) if id else 0. for mid, rai, deci, id in zip(mass_id, ra_deg, dec_deg, idx)]) * 20./DL
    sigcl_TQ   = np.array([grids_sigcl_TQ[mid]((deci, rai)) if id else 0. for mid, rai, deci, id in zip(mass_id, ra_deg, dec_deg, idx)]) * 20./DL
    
    return np.sqrt(sigMM_LISA), np.sqrt(sigDD_LISA), sigcc_LISA, sigll_LISA, sigcl_LISA, np.sqrt(sigMM_TQ), np.sqrt(sigDD_TQ), sigcc_TQ, sigll_TQ, sigcl_TQ

def main():

    parser = optparse.OptionParser(prog = 'masslum-catalog', description = 'Synthetic catalog generator')
    # Input/output
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder. Default: local directory", default = None)
    parser.add_option("--gal_density", type = "float", dest = "gal_density", help  = "Galaxy density", default = None)
    parser.add_option("--rates", type = "string", dest = "rates", help = "EMRI rate per Mpc^3 per year", default = "1e-7,1e-6,1e-5")
    (options, args) = parser.parse_args()
    
    if options.output is not None:
        options.output = Path(options.output).resolve()
        options.output.mkdir(exist_ok = True, parents = True)
    else:
        options.output = Path('.').resolve()

    if options.gal_density is None:
        options.gal_density = gal_density
    
    rates = [float(s) for s in options.rates.split(',')]
    
    ngal  = int(omega.ComovingVolume(zmax)*options.gal_density)
    print(f'Sampling {ngal} galaxies')
    cat   = sample_catalog(ngal)
    
    np.savetxt(Path(options.output, f'catalog_mth_{mth}.txt'), cat[cat[:,1]<mth], header = 'L m M ra dec z DL dz')
#    np.savetxt(Path(options.output, f'full_catalog.txt'), cat, header = 'L m M ra dec z DL dz')
    
    
    
    L   = cat[:,0]
    ra  = cat[:,3]
    dec = cat[:,4]
    DL  = cat[:,6]
    m1  = mass_luminosity_relation(L, a_ding, b_ding)
    
    for rate in rates:
        
        n_events = int(omega.ComovingVolume(zmax)*2*rate)
        
        print(f'Computing SNR, EMRI rate = {rate}')
        snr_LISA, snr_TQ = compute_SNR(m1[:n_events], ra[:n_events], dec[:n_events], DL[:n_events])
        print(f'Computing errors, EMRI rate = {rate}')
        std_m_LISA, std_dl_LISA, cov_dec_LISA, cov_ra_LISA, cor_radec_LISA, std_m_TQ, std_dl_TQ, cov_dec_TQ, cov_ra_TQ, cor_radec_TQ = compute_error(m1[:n_events], ra[:n_events], dec[:n_events], DL[:n_events])
        
        
        # Save to file
        np.savetxt(f'events_rate_{rate}.txt', np.array([m1[:n_events], DL[:n_events], ra[:n_events], dec[:n_events], snr_LISA, snr_TQ, std_m_LISA, std_dl_LISA, cov_dec_LISA, cov_ra_LISA, cor_radec_LISA, std_m_TQ, std_dl_TQ, cov_dec_TQ, cov_ra_TQ, cor_radec_TQ]).T, header = ' '.join(['m1','DL','ra','dec','snr_LISA', 'snr_TQ', 'std_m_LISA', 'std_dl_LISA', 'cov_dec_LISA', 'cov_ra_LISA', 'cor_radec_LISA', 'std_m_TQ', 'std_dl_TQ', 'cov_dec_TQ', 'cov_ra_TQ', 'cor_radec_TQ']))
        
        fig, ax = plt.subplots()
        c = ax.scatter(np.log10(m1[:n_events]), DL[:n_events], c = snr_LISA, marker = '.')
        idx = snr_LISA > 30
        ax.scatter(np.log10(m1[:n_events])[idx], DL[:n_events][idx], c = snr_LISA[idx], marker = '*')
        fig.colorbar(c, label = '$\\mathrm{SNR_{LISA}}$')
        ax.set_xlabel('$\\log_{10}(M)$')
        ax.set_ylabel('$D_L\ [\\mathrm{Mpc}]$')
        fig.savefig(f'snr_LISA_rate_{rate}.pdf', bbox_inches = 'tight')

        fig, ax = plt.subplots()
        c = ax.scatter(np.log10(m1[:n_events]), DL[:n_events], c = snr_TQ, marker = '.')
        idx = snr_TQ > 30
        ax.scatter(np.log10(m1[:n_events])[idx], DL[:n_events][idx], c = snr_TQ[idx], marker = '*')
        fig.colorbar(c, label = '$\\mathrm{SNR_{TQ}}$')
        ax.set_xlabel('$\\log_{10}(M)$')
        ax.set_ylabel('$D_L\ [\\mathrm{Mpc}]$')
        fig.savefig(f'snr_TQ_rate_{rate}.pdf', bbox_inches = 'tight')
    
if __name__ == '__main__':
    main()
