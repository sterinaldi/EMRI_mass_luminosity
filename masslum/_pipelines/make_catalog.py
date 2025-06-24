import numpy as np
import optparse
from corner import corner
from masslum.luminosity import gal_density, zmax, sample_catalog, mth
    
def main():

    parser = optparse.OptionParser(prog = 'masslum-catalog', description = 'Synthetic catalog generator')
    # Input/output
    parser.add_option("-o", "--output", type = "string", dest = "output", help = "Output folder. Default: local directory", default = None)
    parser.add_option("--gal_density", type = "float", dest = "gal_density", help  = "Galaxy density", default = None)

    if options.output is not None:
        options.output = Path(options.output).resolve()
        options.output.mkdir(exist_ok = True, parents = True)
    else:
        options.output = Path('.').resolve()

    if options.gal_density is None:
        options.gal_density = gal_density
    
    ngal  = int(omega.ComovingVolume(zmax)*options.gal_density)
    cat   = sample_catalog(ngal)
    
    np.savetxt(Path(options.output, f'catalog_mth_{mth}.txt'), cat[cat[:,1]<mth], header = 'L m M ra dec z DL dz')
    np.savetxt(Path(options.output, f'full_catalog.txt'), cat, header = 'L m M ra dec z DL dz')

if __name__ == '__main__':
    main()
