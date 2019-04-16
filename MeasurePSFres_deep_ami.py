#! /usr/bin/env python
# Run PSFEx for a set of exposures, including making any necessarily input files.
# It also logs errors into a psf blacklist file.
# Functions stolen from: 
# https://github.com/rmjarvis/DESWL/blob/master/psf/run_piff.py
# Currently NO RESERVE STARS
# Probably lots of extraneous parameters and flags as well...
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
import pixmappy
import pandas
import galsim
import galsim.des
import piff
import ngmix

import matplotlib
#matplotlib.use('Agg') # needs to be done before import pyplot
import matplotlib.pyplot as plt

# Don't skip columns in describe output  (default is 20, which is a bit too small)
pandas.options.display.max_columns = 200

# Define the parameters for the blacklist
# AC note:  a lot of these settings are not really used for the deep fields
# currently...!
# How many stars are too few or too many?
FEW_STARS = 25
MANY_STARS_FRAC = 0.3
# How high is a high FWHM?  3.6 arcsec / 0.26 arcsec/pixel = 13.8 pixels
#HIGH_FWHM = 13.8
HIGH_FWHM = 3.6  # (We switched to measuring this in arcsec)
NSIG_T_OUTLIER = 4   # How many sigma for a chip to be an outlier in <T>.

# Not copying flag vals for blacklist and psf catalog...
rng = galsim.BaseDeviate(1234)
MAX_CENTROID_SHIFT = 1.0
NOT_USED = 1
BAD_MEASUREMENT = 2
CENTROID_SHIFT = 4
OUTLIER = 8
FAILURE = 32
RESERVED = 64
NOT_STAR = 128
BLACK_FLAG_FACTOR = 512 # blacklist flags are this times the original exposure blacklist flag
# blacklist flags go up to 64,


# Not copying array to convert ccdnum to detpos (this may or may not
# be important)

def read_psfex_stars(star_file, cat_file, magzp, logger):
    """Read the PSFEx star selection
    """
    if not os.path.exists(star_file):
        return None

    # Read the output and make a DataFrome with the contents
    data = fitsio.read(star_file, ext=2)
    data = data.astype(data.dtype.newbyteorder('='))
    flags_psf = data['FLAGS_PSF']
    source_id = data['SOURCE_NUMBER']
    x_im = data['X_IMAGE']
    y_im = data['Y_IMAGE']
    df = pandas.DataFrame(data={'SOURCE_NUMBER':source_id, 'X_IMAGE':x_im,
                          'Y_IMAGE':y_im, 'FLAGS_PSF':flags_psf})

    ntot = len(df)
    nstars = df['FLAGS_PSF'].sum()
    logger.info('   found %d stars',nstars)
    is_star = df['FLAGS_PSF'] == 1

    # Add on some extra information from the sextractor catalog
    sdata = fitsio.read(cat_file, 2)
    assert len(data) == len(sdata)
    df['mag_aper'] = sdata['MAG_APER'][:,0]
    df['flux_radius'] = sdata['FLUX_RADIUS']

    use = df['FLAGS_PSF'] == 1
    df['use'] = use  # Just using all of the stars currently
    return df


def hsm(im, wt, logger):
    #print('im stats: ',im.array.min(),im.array.max(),im.array.mean(),np.median(im.array))
    #print('wt = ',wt)
    #if wt:
        #print('im stats: ',wt.array.min(),wt.array.max(),wt.array.mean(),np.median(wt.array))
    flag = 0
    try:
        shape_data = im.FindAdaptiveMom(weight=wt, strict=False)
        #print('shape_data = ',shape_data)
    except Exception as e:
        logger.info(e)
        logger.info(' *** Bad measurement (caught exception).  Mask this one.')
        flag |= BAD_MEASUREMENT

    if shape_data.moments_status != 0:
        logger.info('status = %s',shape_data.moments_status)
        logger.info(' *** Bad measurement (hsm status).  Mask this one.')
        flag |= BAD_MEASUREMENT

    if galsim.__version__ >= '1.5.1':
        dx = shape_data.moments_centroid.x - im.true_center.x
        dy = shape_data.moments_centroid.y - im.true_center.y
    else:
        dx = shape_data.moments_centroid.x - im.trueCenter().x
        dy = shape_data.moments_centroid.y - im.trueCenter().y
    #print('dx, dy = ',dx,dy)
    if dx**2 + dy**2 > MAX_CENTROID_SHIFT**2:
        logger.info(' *** Centroid shifted by %f,%f in hsm.  Mask this one.',dx,dy)
        flag |= CENTROID_SHIFT

    flux = shape_data.moments_amp
    #print('flux = ',flux)

    # Account for the image wcs
    if im.wcs.isPixelScale():
        g1 = shape_data.observed_shape.g1
        g2 = shape_data.observed_shape.g2
        T = 2 * shape_data.moments_sigma**2 * im.scale**2
        #print('simple shape = ',g1,g2,T)
    else:
        e1 = shape_data.observed_shape.e1
        e2 = shape_data.observed_shape.e2
        s = shape_data.moments_sigma
        #print('simple shape = ',e1,e2,s)

        if galsim.__version__ >= '1.5.1':
            jac = im.wcs.jacobian(im.true_center)
        else:
            jac = im.wcs.jacobian(im.trueCenter())
        M = np.matrix( [[ 1 + e1, e2 ], [ e2, 1 - e1 ]] ) * s*s
        J = jac.getMatrix()
        M = J * M * J.T

        e1 = (M[0,0] - M[1,1]) / (M[0,0] + M[1,1])
        e2 = (2.*M[0,1]) / (M[0,0] + M[1,1])
        T = M[0,0] + M[1,1]

        shear = galsim.Shear(e1=e1, e2=e2)
        g1 = shear.g1
        g2 = shear.g2
        #print('distorted shape = ',g1,g2,T)

    return dx, dy, g1, g2, T, flux, flag

def measure_star_shapes(df, image_file, noweight, wcs, logger):
    """Measure shapes of the raw stellar images at each location.
    """
    logger.info('Read in stars in file: %s',image_file)

    ind = df.index[df['FLAGS_PSF'] == 0]
    logger.info('ind = %s',ind)
    n_psf = len(ind)
    logger.info('n_psf = %s',n_psf)

    df['obs_dx'] = [ -999. ] * len(df)
    df['obs_dy'] = [ -999. ] * len(df)
    df['obs_e1'] = [ -999. ] * len(df)
    df['obs_e2'] = [ -999. ] * len(df)
    df['obs_T'] = [ -999. ] * len(df)
    df['obs_flux'] = [ -999. ] * len(df)
    df['obs_flag'] = [ NOT_STAR ] * len(df)
    df.loc[ind, 'obs_flag'] = 0

    if 'reserve' in df:
        df.loc[df['reserve'], 'obs_flag'] |= RESERVED
        df.loc[~df['use'] & ~df['reserve'], 'obs_flag'] |= NOT_USED
    else:
        df.loc[~df['use'], 'obs_flag'] |= NOT_USED

    full_image = galsim.fits.read(image_file, hdu=0)
    #full_image = galsim.fits.read(image_file)
    if wcs is not None:
        full_image.wcs = wcs

    if not noweight:
        weight_file = image_file.replace(".fits", ".weight.fits")
        full_weight = galsim.fits.read(weight_file, hdu=0)
        full_weight.array[full_weight.array < 0] = 0.

    stamp_size = 48

    for i in ind:
        x = df['X_IMAGE'].iloc[i]
        y = df['Y_IMAGE'].iloc[i]

        #print('Measure shape for star at ',x,y)
        b = galsim.BoundsI(int(x)-stamp_size/2, int(x)+stamp_size/2,
                           int(y)-stamp_size/2, int(y)+stamp_size/2)
        b = b & full_image.bounds
        im = full_image[b]

        if noweight:
            wt = None
        else:
            wt = full_weight[b]

            
        dx, dy, e1, e2, T, flux, flag = hsm(im, wt, logger)
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
    #flag_outliers(df, ind, 'obs', 4., logger) # This needs to be ported...

    # Any stars that weren't measurable here, don't use for PSF fitting.
    df.loc[df['obs_flag']!=0, 'use'] = False

def measure_psfex_shapes(df, psfex_file, image_file, noweight, wcs, logger):
    """Measure shapes of the PSFEx solution at each location.
    """
    logger.info('Read in PSFEx file: %s',psfex_file)

    ind = df.index[df['FLAGS_PSF'] == 0]
    logger.info('ind = %s',ind)
    n_psf = len(ind)
    logger.info('n_psf = %s',n_psf)

    df['psfex_dx'] = [ -999. ] * len(df)
    df['psfex_dy'] = [ -999. ] * len(df)
    df['psfex_e1'] = [ -999. ] * len(df)
    df['psfex_e2'] = [ -999. ] * len(df)
    df['psfex_T'] = [ -999. ] * len(df)
    df['psfex_flux'] = [ -999. ] * len(df)
    df['psfex_flag'] = [ NOT_STAR ] * len(df)
    df.loc[ind, 'psfex_flag'] = 0

    if 'reserve' in df:
        df.loc[df['reserve'], 'psfex_flag'] |= RESERVED
    df.loc[~df['use'], 'psfex_flag'] |= NOT_USED

    try:
        psf = galsim.des.DES_PSFEx(psfex_file, image_file)
    except Exception as e:
        logger.info('Caught %s',e)
        df.loc[ind, 'psfex_flag'] = FAILURE
        return

    full_image = galsim.fits.read(image_file, hdu=0)

    if wcs is not None:
        full_image.wcs = wcs

    if not noweight:
        weight_file = image_file.replace(".fits", ".weight.fits")
        full_weight = galsim.fits.read(weight_file, hdu=0)
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

        dx, dy, e1, e2, T, flux, flag = hsm(im, wt, logger)
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
    #flag_outliers(df, ind, 'psfex', 4., logger)

# Change locations to yours
cdir = '/fs/scratch/cond0080/UltraVISTA/'
pf = '%s/psf/UVISTA_J_21_01_16_psfcat.psf'%cdir
sf = '%s/psf/UVISTA_J_21_01_16_psfex-starlist.fits'%cdir
cf = '%s/cat/UVISTA_J_21_01_16_psfcat.fits'%cdir
im_f = '%s/UVISTA_J_21_01_16_allpaw_skysub_015_dr3_rc_v5.fits'%cdir
# Currently if noweight is False, assumed the weight file is the image file
# but with .weight.fits
#wt_f = '%s/UVISTA_J_21_01_16_allpaw_skysub_015_dr3_rc_v5.weight.fits'%cdir
magzp = 30.0
logging_level = logging.INFO
logger = logging.getLogger('size_residual')
# Read in some useful values, such as position
df = read_psfex_stars(sf, cf, magzp, logger)
# Measure the hsm shapes on the stars in the actual image
measure_star_shapes(
    df,im_f,noweight=False,wcs=None,logger=logger)
# Measure 
measure_psfex_shapes(
    df,pf,im_f,noweight=False,wcs=None,logger=logger)


# Now plot some results:

psf_t = df['psfex_T'].values
star_t = df['obs_T'].values
good = (psf_t!=-999)&(star_t!=-999)
resid_T = psf_t[good]-star_t[good]

# Plotting the distribution of residuals
plt.hist(resid_T, 30)
plt.xlabel('PSF T - obstar T', fontsize='x-large')
plt.ylabel('#',fontsize='x-large')
plt.savefig('UltraVISTA_J_resid.png',bbox_inches='tight')

plt.show()


"""
Currently no functions to:
 - remove bad stars (e.g. I didn't take this function from Mike's script)
 - also didn't take flag_outliers function from his script, and this might
   be useful(?)
"""
