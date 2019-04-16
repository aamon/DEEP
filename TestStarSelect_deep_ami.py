#!/usr/bin/env python

# Make a catalogue with real and model PSFs + magnitude etc, for PSF testing script
# Sex file and star file in des read_files() needs to be altered

from __future__ import print_function
import os
import numpy as np
#from read_psf_cats import read_data, band_combinations
import fitsio
import matplotlib
#matplotlib.use('Agg') # needs to be done before import pyplot
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table,join,hstack
import h5py as h
import sys
import shutil
import logging
import datetime
import traceback
import copy
import glob
import time
import pandas


def read_files(filter, basedir):
    """ Read in the files for a given filter and base directory.
    The structure of the directory is such that within this directory there
    are cat/ and psf/ sub-directories containing the relevant catalogs

    Note: Do not include the final slash!
    e.g. on NERSC the basedir looks like either
    /global/cscratch1/sd/amichoi/UltraVISTA
    or
    /global/cscratch1/sd/amichoi/VIDEO
    """
    survey = os.path.split(basedir)[1]
    print(survey)

    # First read in the SExtractor for PSFEx catalog
    if survey=='UltraVISTA':
        sex_file=os.path.join(
            basedir, 'cat/UVISTA_%s_21_01_16_psfcat.fits' %filter)
    elif survey=='VIDEO':
        sex_file=os.path.join(
              basedir, 'cat/VIDEO_%s_10_36.80_-5.01_psfcat.fits' %filter)
        
    dat = fits.open(sex_file)
    cols = dat[2].columns
    #print(cols)
    sex=Table(dat[2].data)
    print("Length of sex file: ", len(sex))

    #read in catalog containing list of stars made from Sextractor and PSFEx
    if survey=='UltraVISTA':
        star_file=os.path.join(
            basedir, 'psf_minsn500/UVISTA_%s_21_01_16_psfex-starlist.fits' %filter)
    elif survey=='VIDEO':
        star_file=os.path.join(
              basedir, 'psf/VIDEO_%s_10_36.80_-5.01_psfex-starlist.fits' %(
                filter))

    dat = fits.open(star_file)
    cols = dat[2].columns
    #print(cols)
    star=Table(dat[2].data)
    print("Length of star file: ", len(star))
    
    sex['X_IMAGE']=sex['X_IMAGE'].astype(int)
    star['X_IMAGE']=star['X_IMAGE'].astype(int)
    sex['Y_IMAGE']=sex['Y_IMAGE'].astype(int)
    star['Y_IMAGE']=star['Y_IMAGE'].astype(int)
    sexstarmerge = join(sex, star, keys=['X_IMAGE','Y_IMAGE'],  join_type='inner')

    #sexstarmerge = join(sex, star, join_type='inner')
    print("length of merged cat: ", len(sexstarmerge))
 
    cols = tuple(name for name in sexstarmerge.colnames if len(sexstarmerge[name].shape) <= 1)
    t2 = sexstarmerge[cols]
    sexstardf = t2.to_pandas()

    return sexstarmerge, sex, star
    #return sex, star


#filter=['H', 'J', 'Ks']#, 'Y']
filter=['J','H','Ks']
minsn = '500'
numfilts=len(filter)

#make figure for fwhm-snr
"""
fig, axs = plt.subplots(3, figsize=(4, 18), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .1, wspace=.5)
fig2, axs2 = plt.subplots(3, figsize=(4, 18), facecolor='w', edgecolor='k')
fig2.subplots_adjust(hspace = .3, wspace=.5)
fig3, axs3 = plt.subplots(3, figsize=(4, 18), facecolor='w', edgecolor='k')
fig3.subplots_adjust(hspace = .3, wspace=.5)
"""
fig, axs = plt.subplots(ncols=3, figsize=(18, 4), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .1, wspace=.5)
fig2, axs2 = plt.subplots(ncols=3, figsize=(18, 4), facecolor='w', edgecolor='k')
fig2.subplots_adjust(hspace = .3, wspace=.5)
fig3, axs3 = plt.subplots(ncols=3, figsize=(18, 4), facecolor='w', edgecolor='k')
fig3.subplots_adjust(hspace = .3, wspace=.5)
#make figure for flux-size
for i in range(numfilts):
    print(i)
    filt=filter[i]
    print(filt)
    
    sexstar, sex, star =read_files(filt, '/fs/scratch/cond0080/UltraVISTA')
    flags_psf = sexstar['FLAGS_PSF']
    fwhm_psf = sexstar['FWHM_PSF']
    snr_psf = sexstar['SNR_PSF']
    flux_rad = sexstar['FLUX_RADIUS']
    flux_aper = sexstar['FLUX_APER'][:,3]
    fluxerr_aper = sexstar['FLUXERR_APER'][:,3]
    mag_aper = sexstar['MAG_APER'][:,3]
    notstar = np.where(flags_psf!=0)
    star = np.where(flags_psf==0)
    #print(ssdf)
    #print(len(sexstar['FLAGS_PSF'][np.where(sexstar['FLAGS_PSF']!=0)]))
    print(len(flags_psf[notstar]))
    #match data so can use FLAGS_PSF
    #print("length of merged cat: ", len(sexstar))
    print("length of non-stars: ",len(flags_psf[notstar]))
    print("length of stars: ",len(flags_psf[star]))
    #print("length of non-stars: ",len(sexstar['FLAGS_PSF'][np.where(sexstar['FLAGS_PSF']!=0)]))
    #print("length of stars: ",len(sexstar['FLAGS_PSF'][np.where(sexstar['FLAGS_PSF']==0)]))
    
    #add to axes    
    #axs[i].scatter(x=star['FWHM_PSF'],y=star['SNR_PSF'],c='red',label='FLAGS_PSF=0', marker='.',s=4)##,ax=axs[i]) # use this is conconcered about matching
    axs[i].plot(fwhm_psf[notstar], snr_psf[notstar], 'b,',label='FLAGS_PSF>0')
    axs[i].plot(fwhm_psf[star],snr_psf[star], 'r,',label='FLAGS_PSF=0')

    axs[i].text(2, 5E6, "%d/%d FLAGS_PSF=0/FLAGS_PSF!=0"%(
            len(flags_psf[star]), len(flags_psf[notstar])))
    #axs[i].set_xscale('log')
    axs[i].set_yscale('log')
    axs[i].set_xlim(2,12)
    axs[i].set_ylim(10,10**7 )
    axs[i].set_xlabel('FWHM_PSF')
    axs[i].set_ylabel('SNR_PSF')
    axs[i].set_title('%s (SAMPLE_SNMIN=%s)'%(filt, minsn))
    #axs[i].legend(sexsexstar['FLAGS_PSF'])
    
    #axs2[i].scatter(x=sexstar['FLUX_RADIUS'],y=sexstar['FLUX_APER'][:,3],c='blue',label='FLAGS_PSF=0', marker='.',s=4) use this is conconcered about matching
                    
    axs2[i].plot(flux_rad[notstar], flux_aper[notstar]/fluxerr_aper[notstar],'b,',label='FLAGS_PSF>0')
    axs2[i].plot(flux_rad[star],flux_aper[star]/fluxerr_aper[star],'r,',label='FLAGS_PSF=0')

    axs2[i].text(1,5E1, "%d/%d FLAGS_PSF=0/FLAGS_PSF!=0"%(
                len(flags_psf[star]), len(flags_psf[notstar])))
    ##plt.xscale('log')
    axs2[i].set_yscale('log')
    axs2[i].set_ylim(1,10**6)
    axs2[i].set_xlim(1,10)
    #axs2[i].legend(sexsexstar['FLAGS_PSF'])
    axs2[i].set_xlabel('FLUX_RADIUS')
    axs2[i].set_ylabel('FLUX_APER/FLUXERR_APER')
    axs2[i].set_title(filt)

    axs3[i].text(10,1, "%d/%d FLAGS_PSF=0/FLAGS_PSF!=0"%(
                len(flags_psf[star]), len(flags_psf[notstar])))
    axs3[i].plot(mag_aper[notstar],flux_rad[notstar],'b,',label='FLAGS_PSF>0')
    axs3[i].plot(mag_aper[star],flux_rad[star],'r,',label='FLAGS_PSF=0')
    axs3[i].set_ylim(0,10)
    axs3[i].set_xlim(10,28)
    #axs2[i].legend(sexsexstar['FLAGS_PSF'])
    axs3[i].set_ylabel('FLUX_RADIUS')
    axs3[i].set_xlabel('MAG_APER')
    axs3[i].set_title(filt)

#plt.show()

fig.savefig("Figs/UltraVISTA_SNR_psffwhm_minsn%s.png"%minsn)
fig2.savefig("Figs/UltraVISTA_SN_size_minsn%s.png"%minsn)       
fig3.savefig("Figs/UltraVISTA_size_mag_minsn%s.png"%minsn)
