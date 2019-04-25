#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Make a catalogue with real and model PSFs + magnitude etc, for PSF testing script

#! /usr/bin/env python

get_ipython().system('jupyter nbconvert --to script MeasurePSFres_deep.ipynb')

from __future__ import print_function
import os
import numpy as np
#from read_psf_cats import read_data, band_combinations
import fitsio
import treecorr
import matplotlib
import matplotlib
matplotlib.use('Agg') # needs to be done before import pyplot
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table,join
import h5py as h

from __future__ import print_function
import os
import sys
import shutil
import logging
import datetime
import traceback
import numpy as np
import copy
import glob
import time
import fitsio
#import pixmappy
import pandas
import galsim
import galsim.des
#import piff
import ngmix
import wget


# In[2]:


#haven't edited or decided which bits of this cell I need
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Run PSFEx on a set of exposures')

    # Directory arguments
    parser.add_argument('--sex_dir', default='/astro/u/mjarvis/bin/',
                        help='location of sextrator executable')
    parser.add_argument('--piff_exe', default='/astro/u/mjarvis/.conda/envs/py2.7/bin/piffify',
                        help='location of piffify executable')
    parser.add_argument('--findstars_dir', default='/astro/u/mjarvis/bin',
                        help='location wl executables')
    parser.add_argument('--work', default='/astro/u/mjarvis/work/y3_piff',
                        help='location of intermediate outputs')
    parser.add_argument('--scratch', default='/data/mjarvis/y3_piff',
                        help='location of intermediate outputs')
    parser.add_argument('--pixmappy_dir', default='/astro/u/mjarvis/work/y3_piff/astro',
                        help='location of pixmappy astrometric solutions')
    parser.add_argument('--tag', default=None,
                        help='A version tag to add to the directory name')

    # Exposure inputs
    parser.add_argument('--base_exposures',
                        default='/astro/u/mjarvis/work/y3_piff/exposures-ccds-Y3A1_COADD.fits',
                        help='The base file with information about the DES exposures')
    parser.add_argument('--file', default='',
                        help='list of exposures (in lieu of separate exps)')
    parser.add_argument('--exps', default='', nargs='+',
                        help='list of exposures to run')

    # Configuration files
    parser.add_argument('--sex_config',
                        default='/astro/u/mjarvis/rmjarvis/DESWL/psf/y3.sex',
                        help='sextractor config file')
    parser.add_argument('--piff_config',
                        default='/astro/u/mjarvis/rmjarvis/DESWL/psf/piff.yaml',
                        help='piff config file')
    parser.add_argument('--findstars_config',
                        default='/astro/u/mjarvis/rmjarvis/DESWL/psf/y3.config',
                        help='findstars config file')
    parser.add_argument('--sex_params',
                        default='/astro/u/mjarvis/rmjarvis/DESWL/psf/sex.param_piff',
                        help='sextractor param file')
    parser.add_argument('--sex_filter',
                        default='/astro/u/mjarvis/rmjarvis/DESWL/psf/sex.conv',
                        help='name of sextractor filter file')
    parser.add_argument('--sex_nnw',
                        default='/astro/u/mjarvis/rmjarvis/DESWL/psf/sex.nnw',
                        help='name of sextractor star file')
    parser.add_argument('--tapebump_file',
                        default='/astro/u/mjarvis/rmjarvis/DESWL/psf/mask_ccdnum.txt',
                        help='name of tape bump file')
    parser.add_argument('--make_symlinks', default=0, type=int,
                        help='make symlinks in output dir, rather than move files')
    parser.add_argument('--noweight', default=False, action='store_const', const=True,
                        help='do not try to use a weight image.')


    # Options
    parser.add_argument('--clear_output', default=0, type=int,
                        help='should the output directory be cleared before writing new files?')
    parser.add_argument('--rm_files', default=1, type=int,
                        help='remove unpacked files after finished')
    parser.add_argument('--use_existing', default=0, type=int,
                        help='use previously downloaded files if they exist')
    parser.add_argument('--blacklist', default=1, type=int,
                        help='add failed CCDs to the blacklist')
    parser.add_argument('--run_piff', default=1, type=int,
                        help='run piff on files')
    parser.add_argument('--run_sextractor', default=1, type=int,
                        help='run sextractor to remake input catalog')
    parser.add_argument('--run_findstars', default=1, type=int,
                        help='force a run of findstars to get input star catalog')
    parser.add_argument('--mag_cut', default=-1, type=float,
                        help='remove the top mags using mag_auto')
    parser.add_argument('--min_mag', default=-1, type=float,
                        help='remove stars brighter than this mag')
    parser.add_argument('--nbright_stars', default=1, type=int,
                        help='use median of this many brightest stars for min mag')
    parser.add_argument('--max_mag', default=0, type=float,
                        help='only use stars brighter than this mag')
    parser.add_argument('--use_tapebumps', default=1, type=int,
                        help='avoid stars in or near tape bumps')
    parser.add_argument('--tapebump_extra', default=2, type=float,
                        help='How much extra room around tape bumps to exclude stars in units of FWHM')
    parser.add_argument('--single_ccd', default=0, type=int,
                        help='Only do the specified ccd (used for debugging)')
    parser.add_argument('--reserve', default=0, type=float,
                        help='Reserve some fraction of the good stars for testing')
    parser.add_argument('--get_psfex', default=False, action='store_const', const=True,
                        help='Download the PSFEx files along the way')
    parser.add_argument('--plot_fs', default=False, action='store_const', const=True,
                        help='Make a size-magnitude plot of the findstars output')
    parser.add_argument('--use_ngmix', default=False, action='store_const', const=True,
                        help='Use ngmix rather than hsm for the measurements')

    args = parser.parse_args()
    return args


# In[3]:


#read in list of stars made from Sextractor and PSFEx
star_file= "/global/homes/a/aamon/DES/DEStests/DEEP/deeppsfs/UltraVista/UVISTA_J_21_01_16_psfex-starlist.fits"

dat = fits.open(star_file)
cols = dat[2].columns
print(cols)

  # This has the following columns:
    # id: The original id from the SExtractor catalog
    # x: The x position
    # y: The y position
    # sky: The local sky value
    # noise: The estimated noise.  But these are all 0, so I think this isn't being calculated.
    # size_flags: Error flags that occurred when estimating the size
    # mag: The magnitude from SExtractor
    # sg: SExtractor's star/galaxy estimate.  Currently SPREAD_MODEL.  (Actually, currently none)
    # sigma0: The shapelet sigma that results in a b_11 = 0 shapelet parameter.
    # star_flag: 1 if findstars thought this was a star, 0 otherwise.


# In[4]:


def wget( url, file):
    full_file = os.path.join(url,file)
    if not os.path.isfile(full_file):
        # Sometimes this fails with an "http protocol error, bad status line".
        # Maybe from too many requests at once or something.  So we retry up to 5 times.
        nattempts = 5
        cmd = 'wget -q --no-check-certificate %s'%(full_file)
        for attempt in range(1,nattempts+1):
            if os.path.exists(full_file):
                break
    return full_file


# In[5]:


# Download the files we need. These files are 
#It looks to me that the image file is the full coadd, 
image_file = wget('ftp://ftp.star.ucl.ac.uk/whartley/ultraVISTA/','UVISTA_J_21_01_16_allpaw_skysub_015_dr3_rc_v5.fits.gz')

#row['root'] = root
#row['image_file'] = image_file

#usually weight is in image file but in this case, it's a separate file
weight_file = wget('ftp://ftp.star.ucl.ac.uk/whartley/ultraVISTA/','UVISTA_J_21_01_16_allpaw_skysub_015_dr3_rc_v5.weight.fits.gz')


# In[6]:


#Not sure this is necessary, but having this information might be useful for further tests

def read_image_header(row, img_file):
    """Read some information from the image header and write into the df row.
    """
    hdu = 0

    # Note: The next line usually works, but fitsio doesn't support CONTINUE lines, which DES
    #       image headers sometimes include.
    #h = fitsio.read_header(img_file, hdu)
    # I don't  care about any of the lines the sometimes use CONITNUE (e.g. OBSERVER), so I
    # just remove them and make the header with the rest of the entries.
    f = fitsio.FITS(img_file)
    header_list = f[hdu].read_header_list()
    header_list = [ d for d in header_list if 'CONTINUE' not in d['name'] ]
    h = fitsio.FITSHDR(header_list)
    try:
        date = h['DATE-OBS']
        date, time = date.strip().split('T',1)

        filter = h['FILTER']
        filter = filter.split()[0]

        sat = h['SATURATE']
        fwhm = h['FWHM']

        ccdnum = int(h['CCDNUM'])
        detpos = h['DETPOS'].strip()

        telra = h['TELRA']
        teldec = h['TELDEC']
        telha = h['HA']
        if galsim.__version__ >= '1.5.1':
            telra = galsim.Angle.from_hms(telra) / galsim.degrees
            teldec = galsim.Angle.from_dms(teldec) / galsim.degrees
            telha = galsim.Angle.from_hms(telha) / galsim.degrees
        else:
            telra = galsim.HMS_Angle(telra) / galsim.degrees
            teldec = galsim.DMS_Angle(teldec) / galsim.degrees
            telha = galsim.HMS_Angle(telha) / galsim.degrees

        airmass = float(h.get('AIRMASS',-999))
        sky = float(h.get('SKYBRITE',-999))
        sigsky = float(h.get('SKYSIGMA',-999))

        tiling = int(h.get('TILING',0))
        hex = int(h.get('HEX',0))

    except Exception as e:
        logger.info("Caught %s",e)
        logger.info("Cannot read header information from %s", img_file)
        raise

    row['date'] = date
    row['time'] = time
    row['sat'] = sat
    row['fits_filter'] = filter
    row['fits_fwhm'] = fwhm
    row['fits_ccdnum'] = ccdnum
    row['telra'] = telra
    row['teldec'] = teldec
    row['telha'] = telha
    row['airmass'] = airmass
    row['sky'] = sky
    row['sigsky'] = sigsky
    row['tiling'] = tiling
    row['hex'] = hex


# In[7]:


# Make the work directory if it does not exist yet.
#This chunk of code doens't work
args = parse_args()
work = os.path.expanduser(args.work)
try:
    if not os.path.exists(work):
        os.makedirs(work)
except OSError:
    if not os.path.exists(work): raise
scratch = os.path.expanduser(args.scratch)
logger.info('scratch dir = %s',scratch)
try:
    if not os.path.exists(scratch):
        os.makedirs(scratch)

except OSError:
    if not os.path.exists(scratch): raise
            
# A listing Erin made of all the exposures in Y3 used in meds files
all_exp = fitsio.read(args.base_exposures)
# Switch to native endians, so pandas doesn't complain.
all_exp = all_exp.astype(all_exp.dtype.newbyteorder('='))
            
row = pandas.DataFrame(info).iloc[0]
read_image_header(row, image_file)


# In[8]:


#put the stars data into a dataframe 

def read_findstars(star_file, img_file):
    """Read the findstars output file
    """
    if not os.path.exists(star_file):
        return None

    # Read the output and make a DataFrome with the contents *********something buggy here
    data = fitsio.read(star_file)
    data = data.astype(data.dtype.newbyteorder('='))
    print(data) 
    df = pandas.DataFrame(data)
    print(df)
    ntot = len(df)
    ######nstars = df['star_flag'].sum()
    #logger.info('   found %d stars',ntot)
    print('   found %d stars',ntot)

    #print('mag range = ',np.min(df['mag']), np.max(df['mag']))
    #####is_star = df['star_flag'] == 1
    #print('star mag range = ',np.min(df['mag'][is_star]), np.max(df['mag'][is_star]))
    #print('zero point = ',magzp)
    #####df['mag'] += magzp - 25.
    #print('star mag range => ',np.min(df['mag'][is_star]), np.max(df['mag'][is_star]))

    #Add on some extra information from the sextractor catalog
    #INSTEAD I'LL USE THE WCS AND THE X,Y TO GET RA AND DEC.
    image = galsim.fits.read(img_file)
    wcs = image.wcs
    world = w.wcs_pix2world((x,y))
    print(world)
    df['ra'] = world[:,0]
    df['dec'] = world[:,1]
    print(df)
    return df


# In[9]:


df= read_findstars(star_file,image_file)


# In[ ]:


#read in psf model file

psfex_file= "/global/homes/a/aamon/DES/DEStests/deeppsfs/UltraVista/UVISTA_J_21_01_16_psfcat.psf"
dat = fits.open(psf_file)
print(dat.info()) 
print(dat[1].header)
data= dat[1].data

#if args.get_psfex:
#  if not (args.use_existing and os.path.exists(psfex_File)):
#   psfex_file = wget(url_base, base_path + '/psf/', wdir, root + '_psfexcat.psf', logger)
#    logger.info('psfex_file = %s',psfex_file)
#    row['psfex_file'] = psfex_file
#    keep_files.append(psfex_file)


# In[ ]:


#neither my starlist nor Mike's has an obs_flux in the starlist?

def measure_psfex_shapes(df, psfex_file, image_file, noweight, wcs, fwhm): #, logger):
    """Measure shapes of the PSFEx solution at each location.
    """
    #logger.info('Read in PSFEx file: %s',psfex_file)

    #ignore fact that I have no star_file for now
    ind = df.index[df] 
    #ind = df.index[df['star_flag'] == 1]
    #logger.info('ind = %s',ind)
    #n_psf = len(ind)
    #logger.info('n_psf = %s',n_psf)

    df['psfex_dx'] = [ -999. ] * len(df)
    df['psfex_dy'] = [ -999. ] * len(df)
    df['psfex_e1'] = [ -999. ] * len(df)
    df['psfex_e2'] = [ -999. ] * len(df)
    df['psfex_T'] = [ -999. ] * len(df)
    df['psfex_flux'] = [ -999. ] * len(df)
    df['psfex_flag'] = [ NOT_STAR ] * len(df)
    df.loc[ind, 'psfex_flag'] = 0

    full_image = galsim.fits.read(image_file, hdu=0)

    if wcs is not None:
        full_image.wcs = wcs

    if not noweight:
        print("I'm using a weight)")
        full_weight = galsim.fits.read(image_file, hdu=0)
        full_weight.array[full_weight.array < 0] = 0.

    stamp_size = 48

    for i in ind:
        x = df['X_IMAGE'].iloc[i]
        y = df['Y_IMAGE'].iloc[i]
        
        #print('Measure PSFEx model shape at ',x,y)
        image_pos = galsim.PositionD(x,y)
        psf_i = psf.getPSF(image_pos)

        b = galsim.BoundsI(int(x)-stamp_size/2, int(x)+stamp_size/2,
                           int(y)-stamp_size/2, int(y)+stamp_size/2)
        b = b & full_image.bounds
        im = full_image[b]

        im = psf_i.drawImage(image=im, method='no_pixel')
        im *= df['obs_flux'].iloc[i]    

        if noweight:
            wt = None
        else:
            wt = full_weight[b]
            var = wt.copy()
            var.invertSelf()
            im.addNoise(galsim.VariableGaussianNoise(rng, var))
        
        dx, dy, e1, e2, T, flux, flag = ngmix_fit(im, wt, fwhm, x, y, logger)
        
        if np.any(np.isnan([dx,dy,e1,e2,T,flux])):
            logger.info(' *** NaN detected (%f,%f,%f,%f,%f,%f).',dx,dy,e1,e2,T,flux)
            flag |= BAD_MEASUREMENT
        else:
            df.loc[i, 'psfex_dx'] = dx
            df.loc[i, 'psfex_dy'] = dy
            df.loc[i, 'psfex_e1'] = e1
            df.loc[i, 'psfex_e2'] = e2
            df.loc[i, 'psfex_T'] = T
            df.loc[i, 'psfex_flux'] = flux
        df.loc[i, 'psfex_flag'] |= flag
    logger.info('final psfex_flag = %s',df['psfex_flag'][ind].values)
    #print('df[ind] = ',df.loc[ind].describe())
    flag_outliers(df, ind, 'psfex', 4., logger)


# In[ ]:


measure_psfex_shapes(df, psfex_file, image_file, noweight=False, wcs, fwhm) #, logger)   


# In[ ]:


def measure_star_shapes(df, image_file, noweight, wcs, fwhm): #, logger):
    """Measure shapes of the raw stellar images at each location.
    """
    #logger.info('Read in stars in file: %s',image_file)

    ind = df.index[df] #['star_flag'] == 1]
    #logger.info('ind = %s',ind)
    n_psf = len(ind)
    #logger.info('n_psf = %s',n_psf) #ignore logger for now
    print('n_psf = %s',n_psf)

    df['obs_dx'] = [ -999. ] * len(df)
    df['obs_dy'] = [ -999. ] * len(df)
    df['obs_e1'] = [ -999. ] * len(df)
    df['obs_e2'] = [ -999. ] * len(df)
    df['obs_T'] = [ -999. ] * len(df)
    df['obs_flux'] = [ -999. ] * len(df)
    df['obs_flag'] = [ NOT_STAR ] * len(df)
    df.loc[ind, 'obs_flag'] = 0

    full_image = galsim.fits.read(image_file, hdu=0)

    if wcs is not None:
        full_image.wcs = wcs

    if not noweight:
        full_weight = galsim.fits.read(image_file, hdu=2)
        full_weight.array[full_weight.array < 0] = 0.

    stamp_size = 48

    for i in ind:
        x = df['x'].iloc[i]
        y = df['y'].iloc[i]

        #print('Measure shape for star at ',x,y)
        b = galsim.BoundsI(int(x)-stamp_size/2, int(x)+stamp_size/2,
                           int(y)-stamp_size/2, int(y)+stamp_size/2)
        b = b & full_image.bounds
        im = full_image[b]

        if noweight:
            wt = None
        else:
            wt = full_weight[b]

        
        dx, dy, e1, e2, T, flux, flag = ngmix_fit(im, wt, fwhm, x, y, logger)

        #logger.info('ngmix measurement: (%f,%f,%f,%f,%f,%f).',dx,dy,e1,e2,T,flux)
        if np.any(np.isnan([dx,dy,e1,e2,T,flux])):
            logger.info(' *** NaN detected (%f,%f,%f,%f,%f,%f).',dx,dy,e1,e2,T,flux)
            flag |= BAD_MEASUREMENT
        else:
            df.loc[i, 'obs_dx'] = dx
            df.loc[i, 'obs_dy'] = dy
            df.loc[i, 'obs_e1'] = e1
            df.loc[i, 'obs_e2'] = e2
            df.loc[i, 'obs_T'] = T
            df.loc[i, 'obs_flux'] = flux
        df.loc[i, 'obs_flag'] |= flag
    logger.info('final obs_flag = %s',df['obs_flag'][ind].values)
    #print('df[ind] = ',df.loc[ind].describe())
    flag_outliers(df, ind, 'obs', 4., logger)

    # Any stars that weren't measurable here, don't use for PSF fitting.
    df.loc[df['obs_flag']!=0, 'use'] = False
    
measure_star_shapes(df, image_file, noweight, wcs, fwhm, logger)


# In[ ]:


exp_cat_file = os.path.join(wdir, 'exp_psf_cat_%d.fits'%exp)
        with fitsio.FITS(exp_cat_file,'rw',clobber=True) as f:
            f.write_table(exp_stars_df.to_records(index=False), extname='stars')
            f.write_table(exp_info_df.to_records(index=False), extname='info')


# In[ ]:





# In[ ]:





# In[ ]:




