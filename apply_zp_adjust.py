import numpy as np
from astropy.table import Table
import astropy.io.fits as pyfits
from astropy import units as u
from astropy.coordinates import SkyCoord
import copy
import sys
import yaml
import glob

"""
The expected structure is for flat fits files with the photometry to be in the form of 2-D arrays, as are produced by Erin's model fitting code. 
"""

def read_cat(in_cat):
    try:
        t = Table.read(in_cat)
    except:
        print('Error: catalogue ',in_cat,' not found or not a fits table!')
        sys.exit()
    return t


def match_cats(ra1, dec1, ra2, dec2):
    # ra1, dec1 are the deep cat.
    c = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)  
    catalog = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)  
    idx, d2d, d3d = c.match_to_catalog_sky(catalog)
    return idx

    
def calibrate(t, config, nir=False):
    try:
        # get the correct set of column names
        if nir is True:
            bands = config['bands_nir']
        else:
            bands = config['bands_opt']

        print(bands)
        # Combine the adjustments into a single set
        # need to change to get the keys and iterate.
        # X3, C3, E2, COS
        adjust = {}
        for ikey in config['colour_adj'].keys():
            adj1 = config['colour_adj'][ikey]
            adj2 = config['gold_adj'][ikey]
            adj_sum = [adj1[j]+adj2[j] for j in range(4)]
            adjust.update({ikey:adj_sum})

        # define the four deep field subsets
        # change so that these column names are not hard coded.
        X3 = np.where((t['ra']<40.)&(t['dec']>-20.))[0]
        C3 = np.where((t['ra']>40.)&(t['dec']<-20.))[0]
        E2 = np.where((t['dec']<-40.))[0]
        COS = np.where((t['ra']>140.))[0]
        fields = [X3, C3, E2, COS]
        
        # mag cols (don't create new mag err cols, they don't change)
        for col_name in config['mag_cols']:
            # duplicate first
            t[col_name+config['col_ext']] = t[col_name]
            # run over bands
            for b,band in enumerate(bands):
                # and over field
                for f,field in enumerate(fields):
                    t[col_name+config['col_ext']][field,b] += adjust[band][f]
            # clean up nans and infs
            t[col_name+config['col_ext']][np.isnan(t[col_name+config['col_ext']])] = -9.999e9
            t[col_name+config['col_ext']][np.isinf(t[col_name+config['col_ext']])] = -9.999e9
            # finally clean up ~flag values to standard flags
            t[col_name+config['col_ext']][t[col_name+config['col_ext']]<-1.e9] = -9.999e9

        # flux cols
        for col_name in config['flux_cols']:
            # duplicate first
            t[col_name+config['col_ext']] = t[col_name]
            # run over bands
            for b,band in enumerate(bands):
                # and over field
                for f,field in enumerate(fields):
                    t[col_name+config['col_ext']][field,b] /= 10.**(0.4*adjust[band][f])                      # clean up nans and infs
            t[col_name+config['col_ext']][np.isnan(t[col_name+config['col_ext']])] = -9.999e9
            t[col_name+config['col_ext']][np.isinf(t[col_name+config['col_ext']])] = -9.999e9
            # finally clean up ~flag values to standard flags
            t[col_name+config['col_ext']][t[col_name+config['col_ext']]<-1.e9] = -9.999e9    

        # flux error cols
        for col_name in config['flux_err_cols']:
            # duplicate first
            t[col_name+config['col_ext']] = t[col_name]
            # run over bands
            for b,band in enumerate(bands):
                # and over field
                for f,field in enumerate(fields):
                    t[col_name+config['col_ext']][field,b] /= 10.**(0.4*adjust[band][f])                      # clean up nans and infs - error cols have +ve values
            t[col_name+config['col_ext']][np.isnan(t[col_name+config['col_ext']])] = 9.999e9
            t[col_name+config['col_ext']][np.isinf(t[col_name+config['col_ext']])] = 9.999e9
            # finally clean up ~flag values to standard flags
            t[col_name+config['col_ext']][t[col_name+config['col_ext']]>1.e9] = 9.999e9

        # correct the mistake from dereddening that bad PSF_mag were given 9.999e9, not -9.999e9
        t['psf_mag_dered'][t['psf_mag_dered']>1.e9] = -9.999e9
            
    except:
        print('Error: catalogue columns missing or have unexpected names')
        sys.exit()

    # write out
    if nir==True:
        print(config['nir_cat'].split('.')[0]+config['fname_ext']+'.fits')
        t.write(config['nir_cat'].split('.')[0]+config['fname_ext']+'.fits', overwrite=True)
    else:
        print(config['opt_cat'].split('.')[0]+config['fname_ext']+'.fits')
        t.write(config['opt_cat'].split('.')[0]+config['fname_ext']+'.fits', overwrite=True)

    return t


def make_db_file(opt, nir, config):
    # Build the catalogue that will be uploaded to DESDM.

    # sort the two catalogues by id, so that they are side-by-side matches.
    sort_opt = np.argsort(opt['id'])
    sort_nir = np.argsort(nir['id'])
    opt = opt[sort_opt]
    nir = nir[sort_nir]

    # build the new table
    t_new=Table()

    # take all the columns that are not photometry
    for col in config['db_cols']:
        if len(opt[col].shape)==1:
            t_new[col] = opt[col]
        else:
            for i in range(opt[col].shape[1]):
                t_new[col+'_'+str(i)] = opt[col][:,i]
    # nir cols
    for col in config['nir_extra_cols']:
        if len(opt[col].shape)==1:
            t_new[col+config['nir_ext']] = opt[col]
        else:
            for i in range(opt[col].shape[1]):
                t_new[col+config['nir_ext']+'_'+str(i)] = opt[col][:,i]
                
    # now do the photometry cols
    for col in config['photom_cols']:
        for b,band in enumerate(config['bands_opt']):
            t_new[col+'_'+band] = opt[col][:,b]
        for b,band in enumerate(config['bands_nir']):
            t_new[col+'_'+band] = nir[col][:,b]

    t_new.write('Y3_deep_fields_DB.cat.fits', overwrite=True)


if __name__ == '__main__':

    # read in config
    if len(sys.argv) < 2:
        print('Use: python apply_zp_adjust.py <yaml config file>')
        sys.exit()
    else:
        config = yaml.load(open(sys.argv[1]))

    opt = read_cat(config['opt_cat'])
    nir = read_cat(config['nir_cat'])
        
    opt_calib = calibrate(opt, config)
    nir_calib = calibrate(nir, config, nir=True)

    del opt,nir
    
    if config['make_db_file'] is True:
        # do some stuff
        make_db_file(opt_calib, nir_calib, config)
        

    

"""
Alt style if the columns have names:


# perform corrections and write table
try:
    for j,filt in enumerate(bands):
        fcol = 'bdf_flux_{}'.format(filt)
        efcol = 'bdf_flux_err_'.format(filt)
        mcol = 'bdf_mag_{}'.format(filt)
        emcol = 'bdf_mag_err_{}'.format(filt)
        #
        t[mcol+'_dered'] = t[mcol] - coeffs[j]*t['EBV_SFD98']
        t[emcol+'_dered'] = t[emcol]
        t[fcol+'_dered'] = t[fcol] * 10.**(0.4*coeffs[j]*t['EBV_SFD98'])
        t[efcol+'_dered'] = t[efcol] * 10.**(0.4*coeffs[j]*t['EBV_SFD98'])
    t.write(in_cat.split('.')[0]+'_extcorr.fits')
except:
    print('Error: catalogue columns missing or have unexpected names')
    sys.exit()

"""
