import pandas as pd
import os
import numpy as np
import pickle
import json
import scipy
import matplotlib.pyplot as plt
import shutil
from skimage.measure import regionprops
import cv2 
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
from skimage.draw import rectangle
import importlib
import splitfolders
from scipy.stats import gaussian_kde
from skimage import data, color, io, img_as_float


import plots
import df_utils
import features

def cell_sequence(df,output_file='D://',save=False):

	#filtered df of dates before calling
	row = df[df.cell=='hl60d'].iloc[193]
	file_path = 'D:/' + row.date + '/' + row.cell + '/25_50_25x150/oi/' + row.run + '/bin/test_camera_0.raw'
	tf = row.tf

	background = df_utils.raw_frame(file_path,100)
	img1 = np.zeros(background.shape)
	temp = np.zeros((len(row.tf),140,880))
	temp_mask = np.zeros((len(row.tf),140,880))
	tf_list = [2,7,11,14,17,21,25,28,31,35]

	for i in tf_list:
	    
	    temp[i,:,:] = df_utils.raw_frame(file_path,row.tf[i])*row['mask'][i] - background*row['mask'][i]
	    temp_mask[i,:,:] = row['mask'][i]
	    
	img_mask = np.sum(temp_mask,axis=0).astype(np.int)*0.5
	img = np.sum(temp,axis=0)+cv2.blur(background,(3,3))

	plt.figure(figsize=(10,10))
	plt.imshow(img,cmap='gray')
	plt.axis('off')
	plt.tight_layout()
	if save:
		plt.savefig(output_file)
	else:
		plt.show()