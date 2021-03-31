#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import numpy as np
import pickle
import json
import scipy
import matplotlib.pyplot as plt
import shutil
from skimage.measure import regionprops
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import sys
import string
from PIL import Image
from scipy import signal
from scipy.interpolate import interp1d
from skimage import measure
import seaborn as sns
from scipy import stats

# Program specific
PORE_STATS_BASE_DIRECTORY = 'C://Users//codyt//Documents//repos//pore_stats//pore_stats//oi_3'
sys.path.append(PORE_STATS_BASE_DIRECTORY)
import optical_imaging as oi
import oi_file
import image_processing 



#Calculate avg velocity in m/s from backward and forward frame
#Returns np.array with first and last element missing from x_pos
def calc_v(x_pos,fps=11103):
    v_avg_array = []
    for i in range(1,len(x_pos)-1):
        
        xi = x_pos[i]
        xim1 = x_pos[i-1]
        xip1 = x_pos[i+1]
        
        delta_x = ((xi-xim1)+(xip1-xi))/2
        delta_t = 1/fps
        v_avg = delta_x/delta_t
        
        v_avg_array.append(v_avg)
    
    return np.asarray(v_avg_array)/10**6

def region1(xpos,asp):

  asp1_max =  np.nanmax(asp[(xpos>0)&(xpos<50)])
  asp1_arg = np.where(asp==asp1_max)[0][0]

  asp2_min =  np.nanmin(asp[(xpos>50)&(xpos<100)])
  asp2_arg = np.where(asp==asp2_min)[0][0]

  return [xpos[asp1_arg:asp2_arg+1],asp[asp1_arg:asp2_arg+1]]

def region2(xpos,asp):

  asp1_min =  np.nanmin(asp[(xpos>50)&(xpos<100)])
  asp1_arg = np.where(asp==asp1_min)[0][0]

  asp2_max =  np.nanmax(asp[(xpos>100)&(xpos<150)])
  asp2_arg = np.where(asp==asp2_max)[0][0]

  return [xpos[asp1_arg:asp2_arg+1],asp[asp1_arg:asp2_arg+1]]

def region3(xpos,asp):

  asp1_max =  np.nanmax(asp[(xpos>100)&(xpos<150)])
  asp1_arg = np.where(asp==asp1_max)[0][0]

  asp2_min =  np.nanmin(asp[(xpos>150)&(xpos<170)])
  asp2_arg = np.where(asp==asp2_min)[0][0]

  return [xpos[asp1_arg:asp2_arg+1],asp[asp1_arg:asp2_arg+1]]

def fit_poly(data,degree=1):
  
  fit = np.polyfit(data[0],data[1],degree)

  return fit

def calc_features(df):

  df['r_idx'] = df.apply(lambda a: np.argmin(np.abs(a.aspect-1)),axis=1)
  df['nar1_idx'] = df.apply(lambda a: (a['xcm_um']>0)&(a['xcm_um']<50),axis=1)
  df['nar2_idx'] = df.apply(lambda a: (a['xcm_um']>100)&(a['xcm_um']<150),axis=1)
  df['cav_idx'] = df.apply(lambda a: (a['xcm_um']>50)&(a['xcm_um']<100),axis=1)
  df['out1_idx'] = df.apply(lambda a: (a['xcm_um']<0),axis=1)
  df['out2_idx'] = df.apply(lambda a: (a['xcm_um']>150),axis=1)


  df['rad'] = df.apply(lambda a: np.nanmean(a.r_um[a.r_idx]),axis=1)
  df['nar1_def'] = df.apply(lambda a: np.nanmax(a.deform_cx[a.nar1_idx]),axis=1)
  df['nar2_def'] = df.apply(lambda a: np.nanmax(a.deform_cx[a.nar2_idx]),axis=1)
  df['cav1_def'] = df.apply(lambda a: np.nanmin(a.deform_cx[a.cav_idx]),axis=1)

  df['r_el'] = df.apply(lambda a: np.nanmean(a.r_um_el[a.r_idx]),axis=1)
  df['nar1_asp'] = df.apply(lambda a: np.nanmax(a.aspect[a.nar1_idx]),axis=1)
  df['nar2_asp'] = df.apply(lambda a: np.nanmax(a.aspect[a.nar2_idx]),axis=1)
  df['cav1_asp'] = df.apply(lambda a: np.nanmin(a.aspect[a.cav_idx]),axis=1)

  df['nar1_max_arg'] = df.apply(lambda a: np.where(a.aspect == a.nar1_asp)[0][0],axis=1)
  df['nar2_max_arg'] = df.apply(lambda a: np.where(a.aspect == a.nar2_asp)[0][0],axis=1)
  df['cav1_min_arg'] = df.apply(lambda a: np.where(a.aspect == a.cav1_asp)[0][0],axis=1)


  #df['t_poly1'] = df.apply(lambda a: fit_poly(region1(a.tf,a.aspect)),axis=1)
  #df['t_poly2'] = df.apply(lambda a: fit_poly(region2(a.tf,a.aspect)),axis=1)

  df['x_poly1'] = df.apply(lambda a: fit_poly(region1(a.xcm_um,a.aspect)),axis=1)
  df['x_poly2'] = df.apply(lambda a: fit_poly(region2(a.xcm_um,a.aspect)),axis=1)
  df['x_poly3'] = df.apply(lambda a: fit_poly(region3(a.xcm_um,a.aspect)),axis=1)

  df['region1_dx'] = df.apply(lambda a: np.abs(a.xcm_um[a.nar1_max_arg]-a.xcm_um[a.cav1_min_arg]) ,axis=1)
  df['region1_dt'] = df.apply(lambda a: np.abs(a.tf[a.nar1_max_arg]-a.tf[a.cav1_min_arg]) ,axis=1)

  df['region1_dasp'] = df['nar1_asp'] - df['cav1_asp']

  df['delta_asp'] = df.nar1_asp - df.nar2_asp


  df['v_avg'] = df.apply(lambda a: calc_v(a.xcm_um),axis=1)

  # OTHER FEATURES
  df['mean_area'] = df.apply(lambda a: np.nanmean(a['area']), axis=1)
  df['mean_perimeter'] = df.apply(lambda a: np.nanmean(a['perimeter']), axis=1)
  df['mean_aspect'] = df.apply(lambda a: np.nanmean(a['aspect']), axis=1)

  return df 









