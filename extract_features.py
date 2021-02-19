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
import cv2 as cv
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import sys
from sklearn import svm
import string
from PIL import Image
from skimage import measure
import seaborn as sns
from scipy import stats
from itertools import compress
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Load dataframe from pkl
df = pd.read_pickle('/home/dan/Documents/siwylab/AWS/Full_filt_101_cx_el.pkl')

# Filter data for going through channel

enter_exit = []
for index, row in df.iterrows():
    enter = row.xcm_um < 0
    inside = (row.xcm_um > 0) & (row.xcm_um < 150)
    exit = row.xcm_um > 150
    enter_exit.append(np.any(enter)&(np.any(inside))&(np.any(exit)))
df['ch_filt'] = enter_exit
df = df[df.ch_filt]
df['y'] = df.apply(lambda a: int(a['cell'] == 'hl60'), axis=1)

df['r_idx'] = df.apply(lambda a: np.argmin(np.abs(a.aspect-1)),axis=1)
df['nar1_idx'] = df.apply(lambda a: (a['xcm_um']>0)&(a['xcm_um']<50),axis=1)
df['nar2_idx'] = df.apply(lambda a: (a['xcm_um']>100)&(a['xcm_um']<150),axis=1)
df['cav_idx'] = df.apply(lambda a: (a['xcm_um']>50)&(a['xcm_um']<100),axis=1)
df['cav2_idx'] = df.apply(lambda a: (a['xcm_um']>150)&(a['xcm_um']<200),axis=1)

df['rad'] = df.apply(lambda a: np.nanmean(a.r_um[a.r_idx]),axis=1)
df['nar1_def'] = df.apply(lambda a: np.nanmean(a.deform_cx[a.nar1_idx]),axis=1)
df['nar2_def'] = df.apply(lambda a: np.nanmean(a.deform_cx[a.nar2_idx]),axis=1)
df['cav1_def'] = df.apply(lambda a: np.nanmean(a.deform_cx[a.cav_idx]),axis=1)

df['r_el'] = df.apply(lambda a: np.nanmean(a.r_um_el[a.r_idx]),axis=1)
df['nar1_asp'] = df.apply(lambda a: np.nanmax(a.aspect[a.nar1_idx]),axis=1)
df['nar2_asp'] = df.apply(lambda a: np.nanmax(a.aspect[a.nar2_idx]),axis=1)
df['cav1_asp'] = df.apply(lambda a: np.nanmean(a.aspect[a.cav_idx]),axis=1)
df['cav2_asp'] = df.apply(lambda a: np.nanmean(a.aspect[a.cav2_idx]),axis=1)


# Find max of first region, min of cavity, delta between them in TIME, fit curve for aspect ratio in that space
df['nar1_tf'] = df.apply(lambda a: a.tf[a.nar1_idx], axis=1)
df['nar2_tf'] = df.apply(lambda a: a.tf[a.nar2_idx], axis=1)

df['cav_tf'] = df.apply(lambda a: a.tf[a.cav_idx], axis=1)
df['cav2_tf'] = df.apply(lambda a: a.tf[a.cav2_idx], axis=1)

df['nar1_asp_full'] = df.apply(lambda a: a.aspect[a.nar1_idx], axis=1)
df['nar2_asp_full'] = df.apply(lambda a: a.aspect[a.nar2_idx], axis=1)

df['cav_asp_full'] = df.apply(lambda a: a.aspect[a.cav_idx], axis=1)
df['cav2_asp_full'] = df.apply(lambda a: a.aspect[a.cav2_idx], axis=1)

# Get times for max/min of a given region
df['nar1_time_at_max'] = df.apply(lambda a: a.nar1_tf[np.argmax(a.aspect[a.nar1_idx])],axis=1)
df['nar2_time_at_max'] = df.apply(lambda a: a.nar2_tf[np.argmax(a.aspect[a.nar2_idx])],axis=1)

df['cav_time_at_min'] = df.apply(lambda a: a.cav_tf[np.argmin(np.abs(a.aspect[a.cav_idx]))],axis=1)
df['cav2_time_at_min'] = df.apply(lambda a: a.cav2_tf[np.argmin(np.abs(a.aspect[a.cav2_idx]))],axis=1)

df['nar1_asp_at_max'] = df.apply(lambda a: a.nar1_asp_full[np.argmax(a.aspect[a.nar1_idx])],axis=1)
df['nar2_asp_at_max'] = df.apply(lambda a: a.nar2_asp_full[np.argmax(a.aspect[a.nar2_idx])],axis=1)

df['cav_asp_at_min'] = df.apply(lambda a: a.cav_asp_full[np.argmin(np.abs(a.aspect[a.cav_idx]))],axis=1)
df['cav2_asp_at_min'] = df.apply(lambda a: a.cav2_asp_full[np.argmin(np.abs(a.aspect[a.cav2_idx]))],axis=1)

# Find difference in time, aspect ratio (dt, da) for each pair of regions
df['dt1'] = df['cav_time_at_min']-df['nar1_time_at_max']
df['dasp1'] = df['nar1_asp_at_max']-df['cav_asp_at_min']

df['dt2'] = df['cav2_time_at_min']-df['nar2_time_at_max']
df['dasp2'] = df['nar2_asp_at_max']-df['cav2_asp_at_min']

# OTHER FEATURES
df['mean_area'] = df.apply(lambda a: np.nanmean(a['area']), axis=1)
df['mean_perimeter'] = df.apply(lambda a: np.nanmean(a['perimeter']), axis=1)
df['mean_aspect'] = df.apply(lambda a: np.nanmean(a['aspect']), axis=1)

fitp0 = []
fitp1 = []
lfitr0p0 = []
lfitr0p1 = []
lfitr1p0 = []
lfitr1p1 = []
# Need to first set stop + start, then
for index, row in df.iterrows():
    append_data_r0 = False
    append_data_r1 = False
    inner_data = []
    inner_x = []
    inner_t = []
    c = 0
    c2 = 0
    for i, t in enumerate(row['tf']):
        if t == row['nar1_time_at_max']:
            append_data_r0 = True
            
        if t == row['nar2_time_at_max']:
            append_data_r1 = True
            
        # REGION 1 - Append data to inner list
        if append_data_r0:
            c += 1  # Start counting frames only when we start appending data
            aspect = np.nan_to_num(row['aspect'][i])
            inner_data.append(aspect)
            inner_x.append(row['xcm_um'][i])
            inner_t.append(c)
        
        # REGION 2 - Append data to inner list
        if append_data_r1:
            c2 += 1  # Start counting frames only when we start appending data
            aspect = np.nan_to_num(row['aspect'][i])
            inner_data.append(aspect)
            inner_x.append(row['xcm_um'][i])
            inner_t.append(c2)
        
        # REGION 1 - Append data to outer list
        if t == row['cav_time_at_min']:
            fitp1.append(np.polyfit(inner_x, inner_data, 1)[1])
            fitp0.append(np.polyfit(inner_x, inner_data, 1)[0])
            lfitr0p0.append(np.polyfit(inner_t, inner_data, 1)[0])
            lfitr0p1.append(np.polyfit(inner_t, inner_data, 1)[1])
            
            # Reset lists for next region
            inner_data = []
            inner_x = []
            inner_t = []
        
        # REGION 2 - Append data to outer list
        if t == row['cav2_time_at_min']:
            fitp1.append(np.polyfit(inner_x, inner_data, 1)[1])
            fitp0.append(np.polyfit(inner_x, inner_data, 1)[0])
            lfitr1p0.append(np.polyfit(inner_t, inner_data, 1)[0])
            lfitr1p1.append(np.polyfit(inner_t, inner_data, 1)[1])
        
        
# P0 is slope
# P1 is intercept
# df['fitp0'] = fitp0
df['lfitr0p0'] = lfitr0p0
df['lfitr0p1'] = lfitr0p1
df['lfitr1p0'] = lfitr1p0
df['lfitr1p1'] = lfitr1p1
# Peak to peak
df['peak_to_peak'] = df['lfitr0p1'] - df['lfitr1p1']

