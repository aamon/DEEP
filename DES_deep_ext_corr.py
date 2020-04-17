import numpy as np
from astropy.table import Table
import astropy.io.fits as pyfits
from astropy import units as u
from astropy.coordinates import SkyCoord
import copy
import sys
import yaml
import  h5py as h
"""
The expected structure is for flat fits files with the photometry to be in the form of 2-D arrays, as are produced by Erin's model fitting code. 
"""

def read_cat(in_cat):
    try:
        print(in_cat)
        t = Table.read(in_cat)
    except:
        print('Error: catalogue ',in_cat,' not found or not a fits table!')
        data = h.File(in_cat,'`r')
        t = np.array(f['catalog/gold'])
        #sys.exit()
    return t


def match_cats(ra1, dec1, ra2, dec2):
    print("starting matching")
    # ra1, dec1 are the deep cat.
    c = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)  
    catalog = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)  
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    return idx

    
def deredden(t, config):
    try:
        print("finished matchign and dereddending")
        # get column names and extinction coeffs from config
        fcol = config['flux_col']
        efcol = config['flux_err_col']
        mcol = config['mag_col']
        coeffs = config['coeffs']

        # build extinction array
        extinct = np.array(t['EBV_SFD98'].quantity).repeat(len(coeffs)).reshape((-1,len(coeffs))) * coeffs

        # apply corrections
        t[fcol+'_dered'] = t[fcol] * 10.**(0.4*extinct)
        t[efcol+'_dered'] = t[efcol] * 10.**(0.4*extinct)
        t[mcol+'_dered'] = t[mcol] - extinct
        t[mcol+'_err_dered'] = 2.5*np.log10(1.+(t[efcol]/t[fcol]))

        # clean up inf and nan to flag values
        t[fcol+'_dered'][np.isnan(t[fcol+'_dered'])] = -9.999e9
        t[mcol+'_dered'][np.isnan(t[mcol+'_dered'])] = -9.999e9
        t[efcol+'_dered'][np.isnan(t[efcol+'_dered'])] = 9.999e9
        t[mcol+'_err_dered'][np.isnan(t[mcol+'_err_dered'])] = 9.999e9

        t[fcol+'_dered'][np.isinf(t[fcol+'_dered'])] = -9.999e9
        t[mcol+'_dered'][np.isinf(t[mcol+'_dered'])] = -9.999e9
        t[efcol+'_dered'][np.isinf(t[efcol+'_dered'])] = 9.999e9
        t[mcol+'_err_dered'][np.isinf(t[mcol+'_err_dered'])] = 9.999e9

    except:
        print('Error: catalogue columns missing or have unexpected names')
        sys.exit()

    # write out
    print(config['deep_cat'].split('.')[0]+'_extcorr.fits')
    t.write(config['deep_cat'].split('.')[0]+'_extcorr.fits', overwrite=True)
    
        

if __name__ == '__main__':

    # read in config
    if len(sys.argv) < 2:
        print('Use: python DES_deep_ext_corr.py dered.yaml')
        sys.exit()
    else:
        config = yaml.load(open(sys.argv[1]))
    print(config)

    deep = read_cat(config['deep_cat'])
    gold = read_cat(config['gold_cat'])
    
    print(deep)

    # match cats
    idx = match_cats(deep[config['coords'][0]], deep[config['coords'][1]], gold[config['coords'][2]], gold[config['coords'][3]])

    # add extinction column to deep cat
    deep[config['Ext_col']] = gold[config['Ext_col']][idx]

    # free this up if we can
    del(gold)
    
    deredden(deep, config)
