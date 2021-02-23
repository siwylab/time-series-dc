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


# Program specific
PORE_STATS_BASE_DIRECTORY = 'C://Users//codyt//Documents//repos//pore_stats//pore_stats//oi_3'

sys.path.append(PORE_STATS_BASE_DIRECTORY)
import optical_imaging as oi
import oi_file
import image_processing 

# Open video
res_x = 880
res_y = 140
fps = 11103
exp = 1.5
camera = 0

#Define Videos for calculations
data_base_path = 'D:/'
date_ar = ['10-8-20']*6 + ['11-3-20']*5+['11-5-20']*5
particle_type_ar = ['hl60']*3+['hl60d']*3+['hl60']+['hl60d']*4+['hl60']*4+['hl60d']
channel_type = '/25_50_25x150'
file_index_ar = ['0','1','2']*2+['0']+['0','1','2','3']+['0','1','2','3']+['2']

assert len(date_ar)==len(particle_type_ar)==len(file_index_ar)

df_list = []
for i in range(len(date_ar)):

    date = date_ar[i]
    particle_type = particle_type_ar[i]
    file_index = file_index_ar[i]

    df_file_path = data_base_path + date +'/'+ particle_type + channel_type + '/oi/'+ file_index + '/df/calcs101x56_2'

    temp=pd.read_pickle(df_file_path)
    df_list.append(temp)
    
full_df = pd.concat(df_list)

#filter data for going through channel

enter_exit = []
for index, row in full_df.iterrows():
    
    enter = row.xcm_um<0
    inside = (row.xcm_um>0)&(row.xcm_um<150)
    exit = row.xcm_um>150
    enter_exit.append(np.any(enter)&(np.any(inside))&(np.any(exit)))
    
    
filt_df = full_df[enter_exit]
print(len(full_df))

filt_df.to_pickle('D:/full_data101x56_2')
