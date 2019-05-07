#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#! /usr/bin/env python                                                                                            
# Run PSFEx for a set of exposures, including making any necessarily input files.                                 
# It also logs errors into a psf blacklist file.                                                                  
# Functions stolen from:                                                                                          
# https://github.com/rmjarvis/DESWL/blob/master/psf/run_piff.py                                                                                                                                        
# Probably lots of extraneous parameters and flags as well...   

get_ipython().system('jupyter nbconvert --to script MeasurePSFres_new.ipynb')

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
import piff
import ngmix
from astropy.io import fits
import wget
from astropy.wcs import WCS
from ngmix import priors, joint_prior

import matplotlib
#matplotlib.use('Agg') # needs to be done before import pyplot                                                    
import matplotlib.pyplot as plt


# In[2]:


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


# In[31]:


#put the stars data into a dataframe 

def read_psfex_stars(star_file, cat_file, logger): #combination of read findstars and read_image_header in MJ script
    """Read the PSFEx star selection                                                                              
    """
    if not os.path.exists(star_file):
        return None
    #dat = fits.open(star_file)
    #print(dat[2].columns)
    #dat = fits.open(cat_file)
    #print(dat[2].columns)
    
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
    #print('   found %d stars',ntot,nstars)
    logger.info('   found %d stars',nstars)
    #is_star = df['FLAGS_PSF'] == 1
    #print
    #print('   found %d good stars', len(is_star))

    # Add on some extra information from the sextractor catalog                                                   
    sdata = fitsio.read(cat_file, 2)
    #print(data['X_IMAGE'])
    #print(sdata['X_IMAGE'])
    assert len(data) == len(sdata)
    #print("magaper")
    #print(sdata['MAG_APER'])
    #print(sdata['MAG_APER'].shape)
    df['mag_aper'] = sdata['MAG_APER'][:,8]
    df['flux_radius'] = sdata['FLUX_RADIUS']

    #df = df[df.FLAGS_PSF == 0] #this line doesn't work!!!!!!!
    #print('   found %d good stars', len(df))
    plt.scatter(sdata['MAG_APER'][:,8], sdata['FLUX_RADIUS'],c='blue',label='FLAGS_PSF!=0 : %d' % (len(sdata['FLUX_RADIUS'])), marker='.',s=4) # , colormap='viridis')
    plt.scatter(sdata['MAG_APER'][np.where(data['FLAGS_PSF'] == 0),8], sdata['FLUX_RADIUS'][np.where(data['FLAGS_PSF'] == 0)],c='red',label='FLAGS_PSF==0 :%d' % (len(sdata['FLUX_RADIUS'][np.where(data['FLAGS_PSF'] == 0)])), marker='.',s=4) # , colormap='viridis')
    #plt.scatter(df['mag_aper'], df['flux_radius'], c='red',label='FLAGS_PSF==0', marker='.',s=4) # , colormap='viridis')
    plt.xlim((10,28))
    plt.ylim(0,10)
    #axs2[i].legend(sexstar['FLAGS_PSF'])
    plt.ylabel('FLUX_RADIUS')
    plt.xlabel('MAG_APER[:,8]')
    plt.title('%s'% band)
    plt.legend()
    plt.show()

    return df


# In[4]:


#this is just a dublicate from the end in order to test why my mag_apers for "good" galaxies were higher than expected

# Change locations to yours      
cdir = '/global/cscratch1/sd/aamon/DEEP/UVista'
cdir2= '/global/cscratch1/sd/amichoi/UltraVISTA'                                            

#band =  "J" #H, Ks, Y
bands=["J" ]#, "H", "Ks", "Y"]

for band in bands:
    print(band)
    
    pf = '%s/psf/UVISTA_%s_21_01_16_psfcat.psf' % (cdir2, band) # PSFEx image
    sf= '%s/psf/UVISTA_%s_21_01_16_psfex-starlist.fits' % (cdir2, band) #list of stars made from Sextractor and PSFEx
    cf = '%s/cat/UVISTA_%s_21_01_16_psfcat.fits' % (cdir2, band) #the output from extractor 
    im_f = '%s/UVISTA_%s_21_01_16_allpaw_skysub_015_dr3_rc_v5.fits' % (cdir, band)  #VIDEO_H_10_34.31_-4.80.cleaned.fits
    wt_f = '%s/UVISTA_%s_21_01_16_allpaw_skysub_015_dr3_rc_v5.weight.fits'%(cdir, band)   

    full_image = galsim.fits.read(im_f, hdu=0)
    wcs = full_image.wcs
    hdu = 0
    f = fitsio.FITS(im_f)
    header_list = f[hdu].read_header_list()
    header_list = [ d for d in header_list if 'CONTINUE' not in d['name'] ]
    h = fitsio.FITSHDR(header_list)
    #print(h)
    fwhm = h['PSF_FWHM']
    
    magzp = 30.0     
    mmlogging_level = logging.INFO
    logger = logging.getLogger('size_residual')
    df = read_psfex_stars(sf, cf, magzp, logger)


# In[5]:


def make_ngmix_prior(T, pixel_scale):
    # centroid is 1 pixel gaussian in each direction
    cen_prior=priors.CenPrior(0.0, 0.0, pixel_scale, pixel_scale)

    # g is Bernstein & Armstrong prior with sigma = 0.1
    gprior=priors.GPriorBA(0.1)

    # T is log normal with width 0.2
    Tprior=priors.LogNormal(T, 0.2)

    # flux is the only uninformative prior
    Fprior=priors.FlatPrior(-10.0, 1.e10)

    prior=joint_prior.PriorSimpleSep(cen_prior, gprior, Tprior, Fprior)
    return prior


def ngmix_fit(im, wt, fwhm, x, y, logger, psfflag):     
    flag = 0
    dx, dy, g1, g2, flux = 0., 0., 0., 0., 0.
    T_guess = (fwhm / 2.35482)**2 * 2.
    T = T_guess
    #print('fwhm = %s, T_guess = %s'%(fwhm, T_guess))
    if psfflag==0:
        
            #hsm_dx,hsm_dy,hsm_g1,hsm_g2,hsm_T,hsm_flux,hsm_flag = hsm(im, None, logger)
            #logger.info('hsm: %s, %s, %s, %s, %s, %s, %s',hsm_dx,hsm_dy,hsm_g1,hsm_g2,hsm_T,hsm_flux,hsm_flag)
            #if hsm_flag != 0:
                #print('hsm: ',g1,g2,T,flux,hsm_flag)
                #print('Bad hsm measurement.  Reverting to g=(0,0) and T=T_guess = %s'%(T_guess))
                #T = T_guess
            #elif np.abs(np.log(T/T_guess)) > 0.5:
                #print('hsm: ',g1,g2,T,flux,hsm_flag)
                #print('T = %s is not near T_guess = %s.  Reverting to T_guess'%(T,T_guess))
            #T = T_guess
        #print("before wcs.local")
        #print(im.wcs.local)
        #print("before im.center")
        #print(im.true_center)
        #print("this line ", im.wcs.local(im.true_center))
        wcs = im.wcs.local(im.true_center)
        #print(wcs)
        try:
            #print("going to make prior", T,wcs.minLinearScale())
            prior = make_ngmix_prior(T, wcs.minLinearScale())
            #print("prior", prior)

            if galsim.__version__ >= '1.5.1':
                cen = im.true_center - im.origin
            else:
                cen = im.trueCenter() - im.origin()
            jac = ngmix.Jacobian(wcs=wcs, x=cen.x + x - int(x+0.5), y=cen.y + y - int(y+0.5))
            if wt is None:
                obs = ngmix.Observation(image=im.array, jacobian=jac)
            else:
                obs = ngmix.Observation(image=im.array, weight=wt.array, jacobian=jac)

            lm_pars = {'maxfev':4000}
            runner=ngmix.bootstrap.PSFRunner(obs, 'gauss', T, lm_pars, prior=prior)
            runner.go(ntry=3)

            ngmix_flag = runner.fitter.get_result()['flags']
            gmix = runner.fitter.get_gmix()
        except Exception as e:
            logger.info(e)
            logger.info(' *** Bad measurement (caught exception).  Mask this one.')
            print(' *** Bad measurement (caught exception).  Mask this one.')
            flag |= BAD_MEASUREMENT
            return dx,dy,g1,g2,T,flux,flag

        if ngmix_flag != 0:
            logger.info(' *** Bad measurement (ngmix flag = %d).  Mask this one.',ngmix_flag)
            flag |= BAD_MEASUREMENT
            print(' *** Bad measurement (ngmix flag = %d).  Mask this one.',ngmix_flag)

        dx, dy = gmix.get_cen()
        if dx**2 + dy**2 > MAX_CENTROID_SHIFT**2:
            logger.info(' *** Centroid shifted by %f,%f in ngmix.  Mask this one.',dx,dy)
            flag |= CENTROID_SHIFT
            print(' *** Centroid shifted by %f,%f in ngmix.  Mask this one.',dx,dy)

        g1, g2, T = gmix.get_g1g2T()
        if abs(g1) > 0.5 or abs(g2) > 0.5:
            logger.info(' *** Bad shape measurement (%f,%f).  Mask this one.',g1,g2)
            flag |= BAD_MEASUREMENT

        flux = gmix.get_flux() / wcs.pixelArea()  # flux is in ADU.  Should ~ match sum of pixels
    #logger.info('ngmix: %s %s %s %s %s %s %s',dx,dy,g1,g2,T,flux,flag)
    return dx, dy, g1, g2, T, flux, flag
    
#measure_star_shapes(df,im_f,noweight=False,wcs=wcs,use_ngmix=True, fwhm=FWHM,logger=logger)


# In[87]:


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


# In[88]:


#"Measure shapes of the raw stellar images at each location.      
def measure_star_shapes(df, image_file, weight_file,noweight, wcs, use_ngmix, fwhm, logger):
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
        print("finding reserve but I want to ignore this")
        df.loc[df['reserve'], 'obs_flag'] |= RESERVED
        df.loc[~df['use'] & ~df['reserve'], 'obs_flag'] |= NOT_USED
    #else:
        #df.loc[~df['use'], 'obs_flag'] |= NOT_USED

    #df['ra'] = sdata['ALPHAWIN_J2000']
    #df['dec'] = sdata['DELTAWIN_J2000']
    #INSTEAD I'LL USE THE WCS AND THE X,Y TO GET RA AND DEC
    full_image = galsim.fits.read(image_file, hdu=0)
    w = WCS(image_file)
    xall = df['X_IMAGE']
    yall = df['Y_IMAGE']
    
    #print(xall,yall)
    world = w.wcs_pix2world(xall,yall,1)
    #print(world)
    #print(world[0])
    df['ra'] = world[0]
    df['dec'] = world[1]
 
    if wcs is not None:
        full_image.wcs = wcs

    if not noweight:
        print("want weights! ", weight_file)
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
            
        if use_ngmix:
            #print("using ngmix")
            #print(df['FLAGS_PSF'][i])
            dx, dy, e1, e2, T, flux, flag = ngmix_fit(im, wt, fwhm, x, y, logger,df['FLAGS_PSF'][i])
            
        else:
            dx, dy, e1, e2, T, flux, flag = hsm(im, wt, logger)
            
        #dx, dy, e1, e2, T, flux, flag = hsm(im, wt, logger)
        #print(dx, dy, e1, e2, T, flux, flag)
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


# In[89]:


def measure_psfex_shapes(df, psfex_file, image_file, weight_file, noweight, wcs, use_ngmix, fwhm, logger):
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
    #df.loc[~df['use'], 'psfex_flag'] |= NOT_USED

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
        print("want weights! ", weight_file)
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
            
        if use_ngmix:
            dx, dy, e1, e2, T, flux, flag = ngmix_fit(im, wt, fwhm, x, y, logger, df['FLAGS_PSF'][i])
        else:
            dx, dy, e1, e2, T, flux, flag = hsm(im, wt, logger)

        #dx, dy, e1, e2, T, flux, flag = hsm(im, wt, logger)
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
    #print('final psfex_flag = %s',df['psfex_flag'][ind].values)
    logger.info('final psfex_flag = %s',df['psfex_flag'][ind].values)
    #print('df[ind] = ',df.loc[ind].describe())                                                                          
    #flag_outliers(df, ind, 'psfex', 4., logger)                    


# In[9]:


#not working
def wget( url, file):
    full_file = os.path.join(url,file)
    print(full_file)
    if not os.path.isfile(full_file):
        # Sometimes this fails with an "http protocol error, bad status line".
        # Maybe from too many requests at once or something.  So we retry up to 5 times.
        nattempts = 5
        cmd = 'wget -q --no-check-certificate %s'%(full_file)
        for attempt in range(1,nattempts+1):
            if os.path.exists(full_file):
                break
    return full_file


# In[81]:


#want psf vs mag- brighter vs fatter
def bin_by_mag(m, dT, dTfrac, min_mused, band):
    min_mag = min(m) #13.5
    max_mag = max(m) #21
    print("min and max mag: ", min_mag, max_mag)
     
    mag_bins = np.linspace(min_mag,max_mag,31)
    print('mag_bins = ',mag_bins)
    index = np.digitize(m, mag_bins)
    #print('len(index) = ',len(index))

    bin_dT = [dT[index == i].mean() for i in range(1, len(mag_bins))]
    print('bin_dT = ',bin_dT)
    bin_dTfrac = [dTfrac[index == i].mean() for i in range(1, len(mag_bins))]
    #for i in range(1, len(mag_bins)):
        #print(len(dT[index == i]))
    bin_dT_err = [ np.sqrt(dT[index == i].var() / len(dT[index == i])) for i in range(1, len(mag_bins)) ]
    #print('dTfrac = ',len(dTfrac[index == i]))
    
    bin_dTfrac_err = [ np.sqrt(dTfrac[index == i].var() / len(dTfrac[index == i]))
                    for i in range(1, len(mag_bins)) ]
    #print('bin_dTfrac_err = ',bin_dTfrac_err)
    # Fix up nans
    for i in range(1,len(mag_bins)):
        if i not in index:
            bin_dT[i-1] = 0.
            bin_dTfrac[i-1] = 0.
            bin_dT_err[i-1] = 0.
            bin_dTfrac_err[i-1] = 0.
    fig, axes = plt.subplots(2,1, sharex=True)
    
    ax = axes[0]
    #ax.set_ylim(-0.02,0.02)
    ax.plot([min_mag,max_mag], [0,0], color='black')
    ax.plot([min_mused,min_mused],[-1,1], color='Grey')
    #ax.fill( [min_mag,min_mag,min_mused,min_mused], [-1,1,1,-1], fill=True, color='Grey',alpha=0.3)
    ax.fill( [18.3,18.3,max_mag,max_mag], [0.003,-0.001,-0.001,0.003], fill=True, color='grey',alpha=0.3)
    t_line = ax.errorbar(mag_bins[:-1], bin_dT, yerr=bin_dT_err, color='darkturquoise', fmt='o')
    #ax.axhline(y=0.003, linewidth=4, color='grey')
    #ax.legend([t_line], [r'$\delta T$'])
    ax.set_ylabel(r'$(T_{\rm PSF} - T_{\rm model}) \quad({\rm arcsec}^2)$', fontsize='x-large')

    ax = axes[1]
    #ax.set_ylim(-0.05,0.05)
    ax.plot([min_mag,max_mag], [0,0], color='black')
    ax.plot([min_mused,min_mused],[-1,1], color='Grey')
    #ax.fill( [min_mag,min_mag,min_mused,min_mused], [-1,1,1,-1], fill=True, color='Grey',alpha=0.3)
    t_line = ax.errorbar(mag_bins[:-1], bin_dTfrac, yerr=bin_dTfrac_err, color='darkturquoise', fmt='o')
    #ax.legend([t_line], [r'$\delta T$'])
    ax.set_ylabel(r'$(T_{\rm PSF} - T_{\rm model})/ T_{\rm PSF}$', fontsize='x-large')

    ax.set_xlim(min_mag,max_mag)
    ax.set_xlabel('%s-magnitude'% (band), fontsize='x-large')

    fig.set_size_inches(7.0,10.0)
    plt.tight_layout()
    name='dpsf_%s_DEEP.pdf' % (band)
    #plt.savefig(name)
    plt.show()


# In[91]:


# Change locations    
cdir = '/global/cscratch1/sd/aamon/DEEP/UVista'
cdir2= '/global/cscratch1/sd/amichoi/UltraVISTA'                                            

#band =  "J" #H, Ks, Y
bands=["J" ]#, "H", "Ks", "Y"]

for band in bands:
    print(band)
    
    pf = '%s/psf/UVISTA_%s_21_01_16_allpaw_skysub_015_dr3_rc_v5_psfcat.psf' % (cdir2, band) # PSFEx image
    sf= '%s/psf/UVISTA_%s_21_01_16_allpaw_skysub_015_dr3_rc_v5_psfex-starlist.fits' % (cdir2, band) #list of stars made from Sextractor and PSFEx
    cf = '%s/cat/UVISTA_%s_21_01_16_allpaw_skysub_015_dr3_rc_v5_psfcat.fits' % (cdir2, band) #the output from extractor 
    im_f = '%s/UVISTA_%s_21_01_16_allpaw_skysub_015_dr3_rc_v5.fits' % (cdir, band)  #VIDEO_H_10_34.31_-4.80.cleaned.fits
    wt_f = '%s/UVISTA_%s_21_01_16_allpaw_skysub_015_dr3_rc_v5.weight.fits'%(cdir, band) 
    #dat = fits.open(sf)
    #print(dat[2].columns)
    
    #VIDEO
    """
    cdir= '/global/cscratch1/sd/amichoi/VIDEO' 
    cdir2= '/global/cscratch1/sd/amichoi/VIDEO/CDFS'  
    pf = '%s/psf/VIDEO_%s_6_52.80_-27.71_psfcat.psf' % (cdir2, band) # PSFEx image
    sf= '%s/psf/VIDEO_%s_6_52.80_-27.71_psfex-starlist.fits' % (cdir2, band) #list of stars made from Sextractor and PSFEx
    cf = '%s/cat/VIDEO_%s_6_52.80_-27.71_psfcat.fits' % (cdir2, band) #the output from extractor 
    im_f = '%s/VIDEO_%s_6_52.80_-27.71.cleaned.fits' % (cdir, band)  #VIDEO_H_10_34.31_-4.80.cleaned.fits
    wt_f = '%s/VIDEO_%s_6_52.80_-27.71.weight.fits'%(cdir, band) 
    #get wcs and fwhm from image file
    full_image = galsim.fits.read(im_f, hdu=0)
    dat = fitsio.read(sf, ext=2)
    FWHM = dat['FWHM_PSF'] #video doesn't have fwhm in image header! 
    """
    wcs = full_image.wcs
    f = fitsio.FITS(im_f)
    hdu=0
    header_list = f[hdu].read_header_list()
    header_list = [ d for d in header_list if 'CONTINUE' not in d['name'] ]
    h = fitsio.FITSHDR(header_list)
    #print(h)
    FWHM = h['PSF_FWHM'] #video doesn't have fwhm in image header! 


    magzp = 30.0
    mmlogging_level = logging.INFO
    logger = logging.getLogger('size_residual')
    # Read in some useful values, such as position                                                                           
    df = read_psfex_stars(sf, cf, logger)
    # Measure the hsm shapes on the stars in the actual image                                                                
    measure_star_shapes(
        df,im_f,wt_f, noweight=False,wcs=wcs,use_ngmix=False, fwhm=FWHM,logger=logger)
    # Measure      
    #print(list(df))
    measure_psfex_shapes(
        df,pf,im_f, wt_f,noweight=False,wcs=wcs,use_ngmix=False, fwhm=FWHM, logger=logger)

    print(list(df))
    print("All objs: ", len(df))
    df = df[df.FLAGS_PSF == 0]
    from astropy.table import Table
    t = Table.from_pandas(df)
    print(t)
    name='PSFres_VIDEO_%s_ngmix.fits' % (band)
    t.write(name, overwrite=True, format='ascii')  
    #print(t)
    
    ####################################################
    #CUTS
    good = (df['psfex_T'].values!=-999)&(df['obs_T'].values!=-999)
    df=df[good]
    print("Good objs: ", len(df))
    ####################################################
    #PLOT
    def compute_res(d):
        dt =  (d['obs_T']-d[prefix+'_T'])
        dtfrac = dt/d['obs_T']
        print('mean dt = ',np.mean(dt))
        return dtfrac, dt 

    prefix="psfex"
    fracsizeres, sizeres=compute_res(df)

    # Plotting the distribution of residuals                                                                                 
    plt.hist(sizeres, 30)
    plt.xlabel('T_res = PSF - obs', fontsize='x-large')
    plt.ylabel('Num good objects',fontsize='x-large')
    plt.title('%s'% band)
    #plt.savefig('UltraVISTA_J_resid.png',bbox_inches='tight')
    plt.figure(figsize=(12,12))
    plt.show()

    fig, ax = plt.subplots()
    hb=ax.hexbin(df['mag_aper'],sizeres ,bins='log')#, marker='.', facecolors='lightblue', color='blue',alpha=0.2)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('log10(N)')
    plt.xlabel('%s mag_aper' %(band), fontsize='x-large')
    plt.ylabel('T_res', fontsize='x-large')
    #plt.savefig('UltraVISTA_J_resid.png',bbox_inches='tight')

    bin_by_mag(df['mag_aper'], sizeres, fracsizeres, 15, band)


"""                                                                                                                      
Currently no functions to:     
- ngmix instead of hsm
 - remove bad stars (e.g. I didn't take this function from Mike's script)                                                
 - also didn't take flag_outliers function from his script, and this might                                               
   be useful(?)           
- is weight file working?
"""


# In[12]:


# Starfile has the following columns:
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
  
  #psfcat has the following columns:
  # name = 'VIGNET'; format = '3969E'; unit = 'count'; disp = 'G12.7'; dim = '(63, 63)'
  # name = 'XWIN_IMAGE'; format = '1D'; unit = 'pixel'; disp = 'F11.4'
  # name = 'YWIN_IMAGE'; format = '1D'; unit = 'pixel'; disp = 'F11.4'
  # name = 'X_IMAGE'; format = '1E'; unit = 'pixel'; disp = 'F11.4'
  # name = 'Y_IMAGE'; format = '1E'; unit = 'pixel'; disp = 'F11.4'
  # name = 'FLUX_APER'; format = '12E'; unit = 'count'; disp = 'G12.7'
  # name = 'FLUXERR_APER'; format = '12E'; unit = 'count'; disp = 'G12.7'
  # name = 'FLUX_MAX'; format = '1E'; unit = 'count'; disp = 'G12.7'
  # name = 'MAG_APER'; format = '12E'; unit = 'mag'; disp = 'F8.4'
  # name = 'MAGERR_APER'; format = '12E'; unit = 'mag'; disp = 'F8.4'
  # name = 'FLUX_RADIUS'; format = '1E'; unit = 'pixel'; disp = 'F10.3'
  # name = 'ELONGATION'; format = '1E'; disp = 'F8.3'
  # name = 'FLAGS'; format = '1I'; disp = 'I3'
  # name = 'SNR_WIN'; format = '1E'; disp = 'G10.4'
  # name = 'ERRAWIN_WORLD'; format = '1E'; unit = 'deg'; disp = 'G12.7'
  # name = 'ERRBWIN_WORLD'; format = '1E'; unit = 'deg'; disp = 'G12.7'
  # name = 'ERRTHETAWIN_J2000'; format = '1E'; unit = 'deg'; disp = 'F6.2'


# In[13]:


# Now plot some results:                                                                                                 
print("All objs: ", len(df))
#good = (df['psfex_T'].values!=-999)&(df['obs_T'].values!=-999)
#df=df[good]
print("Good objs: ", len(df))
#resid_T = psf_t-star_t
#dtfrac = psf_t/star_t
#mag=df['mag_aper']#[good]

def compute_res(d):
    dt =  (d['obs_T']-d[prefix+'_T'])
    dtfrac = dt/d['obs_T']
    #print('mean dt = ',np.mean(dt))
    return dtfrac, dt 

prefix="psfex"
fracsizeres, sizeres=compute_res(df)

# Plotting the distribution of residuals                                                                                 
plt.hist(sizeres, 30)
plt.xlabel('T_res = PSF - obs', fontsize='x-large')
plt.ylabel('Num good objects',fontsize='x-large')
#plt.savefig('UltraVISTA_J_resid.png',bbox_inches='tight')
plt.figure(figsize=(12,12))
plt.show()

fig, ax = plt.subplots()
hb=ax.hexbin(df['mag_aper'],sizeres ,bins='log')#, marker='.', facecolors='lightblue', color='blue',alpha=0.2)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('log10(N)')
plt.xlabel('mag_aper', fontsize='x-large')
plt.ylabel('T_res', fontsize='x-large')
#plt.savefig('UltraVISTA_J_resid.png',bbox_inches='tight')

bin_by_mag(df['mag_aper'], sizeres, fracsizeres, 15, band)


# In[ ]:




