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






#Pass df to filter out based on y position, area to convex hull ratio, and radius
#DF must have full calculations complete
def filter_df(df,ymax=5,max_ar = 1.05,radius_std = 3):
	
	y_hist =[]
	for idx, row in df.iterrows():
	    y = row.yc_um_el
	    y_cav = y[row.cav_idx]
	    y_avg = y_cav.mean()
	    y_hist.append(y_avg)

	df = df[np.abs(df.y_hist)<ymax]

	#filter by area ratio
	area_ratio =[]
	for idx, row in df.iterrows():
    	area = row.area
    	area_cx = row.area_cx
    	area_ratio.append(np.all(area_cx/area<max_ar))
	df = df[area_ratio]

	#filter by radius
	r_var = np.var(df.rad)
	r_mean = np.mean(df.rad)
	df = df[(df.rad>r_mean-radius_std*r_var)&(df.rad<r_mean+radius_std*r_var)]

	return df

#filter data for going through channel
def filter_enter_exit(df):


	enter_exit = []
	for index, row in df.iterrows():
    
	    enter = row.xcm_um<0
	    inside = (row.xcm_um>0)&(row.xcm_um<150)
	    exit = row.xcm_um>150
	    enter_exit.append(np.any(enter)&(np.any(inside))&(np.any(exit)))
    
    
	return df[enter_exit]



	


