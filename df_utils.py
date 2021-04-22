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
	print('Length prefilter: ' +str(len(df)))
	y_hist =[]
	for idx, row in df.iterrows():
		y = row.yc_um_el
		y_cav = y[row.cav_idx]
		y_avg = y_cav.mean()
		y_hist.append(y_avg)

	df = df[np.abs(y_hist)<ymax]

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


	print('Length postfilter: ' +str(len(df)))
	
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

def extract_bboxes(mask):

    """Compute bounding boxes from masks. Taken from MASK RCNN
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """

    boxes = np.zeros([1, 4], dtype=np.int32)
    
    m = mask
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        x2 += 1
        y2 += 1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    boxes = np.array([y1-20, x1-20, y2+20, x2+20])

    return boxes.astype(np.int32)

def extract_bboxes_uniform(mask,pad = 40):

	"""Compute bounding boxes from masks. Taken from MASK RCNN
	mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

	Returns: bbox array [num_instances, (y1, x1, y2, x2)].
	"""

	boxes = np.zeros([1, 4], dtype=np.int32)

	m = mask
	# Bounding box.
	horizontal_indicies = np.where(np.any(m, axis=0))[0]
	vertical_indicies = np.where(np.any(m, axis=1))[0]

	x_avg = horizontal_indicies.mean()
	y_avg = vertical_indicies.mean()

	x1 = x_avg - pad
	x2 = x_avg + pad
	y1 = y_avg - pad
	y2 = y_avg + pad


	boxes = np.array([y1, x1, y2, x2])

	return boxes.astype(np.int32)

def raw_frame(file_path,tf,width = 880,height = 140):
    
    with open(file_path,'rb') as f:
        f.seek(np.int64(tf*2*height*width))
        image_array = np.fromfile(f,dtype = np.uint16, count = int(height*width))
        frame = image_array.reshape((height,width)).astype(np.float32)
        
    return frame


def raw_bbox(row,back_sub = False,tf_back=0,cav=True,idx=0):

	file_path = 'D:/' + row.date + '/' + row.cell + '/25_50_25x150/oi/' + row.run + '/bin/test_camera_0.raw'

	if cav:
		idx = row.cav1_min_arg
		
	tf = row.tf[idx]
	
	frame = raw_frame(file_path,tf)
	mask = row['mask'][idx]

	if back_sub:
		temp_frame = raw_frame(file_path,tf_back)
		frame = frame - temp_frame

	ystart,xstart,yend,xend= extract_bboxes_uniform(mask)

	return frame[ystart:yend,xstart:xend]

def masked_raw_bbox(row):

	file_path = 'D:/' + row.date + '/' + row.cell + '/25_50_25x150/oi/' + row.run + '/bin/test_camera_0.raw'

	idx = row.cav1_min_arg
	tf = row.tf[idx]

	frame = raw_frame(file_path,tf)
	mask = row['mask'][idx]

	frame = frame*mask

	ystart,xstart,yend,xend = extract_bboxes_uniform(mask)

	return frame[ystart:yend,xstart:xend]

def mask_bbox(row):

	file_path = 'D:/' + row.date + '/' + row.cell + '/25_50_25x150/oi/' + row.run + '/bin/test_camera_0.raw'

	idx = row.cav1_min_arg
	tf = row.tf[idx]

	mask = row['mask'][idx]

	ystart,xstart,yend,xend = extract_bboxes_uniform(mask)

	return mask[ystart:yend,xstart:xend]

def save_cell_images(df,mask=True,root_path = 'D://'):

	cell_types = df.cell.unique()

	for c in cell_types:
		i = 0
		for idx,row in df[df.cell==c].iterrows():

			if mask:
				image = masked_raw_bbox(row)
			else:
				image = raw_bbox(row)

			plt.imsave(root_path+ '/' + c +'/' + str(c) + ' ' + str(i) + '.png',image,vmin=0,vmax=4095,cmap='gray')
			i = i+1

def save_mask_images(df,root_path = 'D://'):

	cell_types = df.cell.unique()

	for c in cell_types:
		i = 0
		for idx,row in df[df.cell==c].iterrows():

			image = mask_bbox(row)


			plt.imsave(root_path+ '/' + c +'/' + str(c) + ' ' + str(i) + '.png',image,cmap='gray')
			i = i+1

def read_feats():
	with open('./sklearn_models/feature_list.txt', 'r') as file:
		return json.load(file)
"""
def save_cells_hdf5(df):



     Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    

    image_array = np.empty([len(df),140,880,3])
    label_array = np.empty([len(df),1])

    cell_types = df.cell.unique()

    for c in cell_types:

		i = 0

		for idx,row in df[df.cell==c].iterrows():

			if mask:
				image_array[i,:,:,:] = np.stack((masked_raw_bbox(row),)*3, axis=-1)
				label_array[i,:] = c
			else:
				image_array[i,:,:,:] = np.stack((raw_bbox(row),)*3, axis=-1)
				label_array[i,:] = c
			i = i+1


    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()

"""