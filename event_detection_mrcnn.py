#!/usr/bin/env python
# coding: utf-8

# !git clone https://github.com/tomgross/Mask_RCNN.git --branch tensorflow-2.0
import os
import pickle
import zipfile
import shutil
import os
import sys
import tarfile
import random

#Numeric computing
import math
import re
import time
import numpy as np

# ML/CV
import cv2
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import normalize 

# plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm

# other
from collections import Counter
from datetime import datetime
# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn_56 import utils
from mrcnn_56 import visualize
from mrcnn_56.visualize import display_images
import mrcnn_56.model as modellib
from mrcnn_56.model import log
from cells import cells56 as cells

PORE_STATS_BASE_DIRECTORY = 'C://Users//codyt//Documents//repos//pore_stats//pore_stats'

sys.path.append(ROOT_DIR+'/oi')
import optical_imaging as oi
import oi_file
import image_processing 

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


config = cells.CellConfig()
CELL_DIR = os.path.join(ROOT_DIR, '/cells/')


# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Directory to save logs and trained model

MODEL_DIR = DEFAULT_LOGS_DIR  

CELL_WEIGHTS_PATH = ROOT_DIR + '/TRAINED_MODEL_WEIGHTS_CONFIG/2-11-101x56.h5'
# Create model in inference mode

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,config=config)


# Load weights
print("Loading weights ", CELL_WEIGHTS_PATH)
model.load_weights(CELL_WEIGHTS_PATH, by_name=True)


def get_distance(events, x_list):
    # Match data to events
    # Set initial distances to greater than channel length
    dist_matrix = np.ones((len(list(events)), len(x_list)))*900 
    bool_matrix = np.zeros((len(list(events)), len(x_list)), bool)
    for i, key in enumerate(events):
        for v, x_pos in enumerate(x_list):
            # If we have new event, no previous x positions to compare to
            if not len(events[key]['xpos']):
                dist_matrix[i, v] = np.nan
            else:
                dist = x_pos - events[key]['xpos'][-1]
                if dist < -15:
                    dist = 900
                dist_matrix[i, v] = dist
    return dist_matrix, bool_matrix

def extract_fits (oi_vid, fname, frange):
    start = datetime.now()
    upper_thresh = 2500
    lower_thresh = 600
    prev_x = 0
    prev_tf = 0
    score_thresh = 0.97
    dt_threshold = 2
    dx_threshold = 60
    image_inner = []  # list of np arrays
    time_inner = []
    active_events = {0:{'mask':[], 'time':[], 'xpos':[]}}
    deactivated_events = {}
    event_counter = 0
    pickle_filename = fname

    # Save version of fit_ellispes, run parameters, video to log

    # Need to create a list of active cells
    # for tf in range(int(oi_vid._total_frames)):
    # for tif in range(1*10**4, 3*10**4):

    for tif in frange:
        deactive_indices = []
        # Get frame and apply mask
        frame = oi_vid.get_frame(tif, camera=camera).astype(np.float32)
        frame = (255*(frame - np.min(frame))/np.ptp(frame)).astype(np.float32)
        results = model.detect([frame[..., np.newaxis]], verbose=0)
        r = results[0]
        mask_list = []
        # If no image is detected, close off any active events outside temporal threshold
        if r['rois'].shape[0] == 0:
            continue
        x_pos_list = []
        y_pos_list = []
        # If there are multiple cells, sort masks based on x-position
        if r['rois'].shape[0] > 1:
            mask_list_unfiltered = []
            # Loop over all cells found, get x, y position
            for instance, box in enumerate(r['rois']):
                x_pos = np.abs(box[1] + box[3])/2
                y_pos = np.abs(box[0] + box[2])/2
                # If the cell is too large or too small, skip it
                area = np.sum(r['masks'][:,:,instance])
                if area > upper_thresh or area < lower_thresh or y_pos > 150:
                    continue
                x_pos_list.append(x_pos)
                y_pos_list.append(y_pos)
                mask_list_unfiltered.append(r['masks'][:,:,instance])
                # Need to deal with case where the nn finds two masks for the same x
            if not x_pos_list:
                continue
            # Find duplicates
            duplicates = [item for item, count in Counter(x_pos_list).items() if count > 1]
            # If there are duplicates, delete the one farthest from the central axis
            if len(duplicates):
                dy = []
                index = []
                for i, (x, y) in enumerate(zip(x_pos_list, y_pos_list)):
                    if x in duplicates:
                        dy.append(np.abs(res_y/2-y))
                        index.append(i)
                sorted_y = [x for _,x in sorted(zip(dy, index))]
                sorted_y.pop(0) # Keep the closest element
                x_pos_list = [i for j, i in enumerate(x_pos_list) if j not in sorted_y]
                mask_list_unfiltered = [i for j, i in enumerate(mask_list_unfiltered) if j not in sorted_y]
            mask_list = [x for _,x in sorted(zip(x_pos_list, mask_list_unfiltered), reverse=True)]  # sort mask list right to left
            x_pos_list.sort(reverse=True)
        else:
            for box in r['rois']:
                x_pos = np.abs(box[1] + box[3])/2
                y_pos = np.abs(box[0] + box[2])/2
                # If the cell is too large or too small, skip it
                area = np.sum(r['masks'])
            if area > upper_thresh or area < lower_thresh:
                continue
            x_pos_list.append(x_pos)
            mask_list = [r['masks']]

        # if there are more x_positions than events, we have a new event
        if len(active_events.keys()) < len(x_pos_list):
            event_counter += 1
            active_events[event_counter] = {'mask':[], 'time':[], 'xpos':[]}

        dist_matrix, _ = get_distance(active_events, x_pos_list)

        # Screen active events, delete inactive events, if there are multiple events, calculate relevant dx
        for i, key in enumerate(active_events):
            # If there is a new event
            if not len(active_events[key]['mask']):
                dx = 1
                dt = 1
            else:
                dx = np.min(dist_matrix[i, :])
                dt = tif - active_events[key]['time'][-1]

            # If event is not new, see if dx and dt are within thresholds
            if dx > dx_threshold or dx < 5 or dt > dt_threshold:
                # Only save events with at least 20 frames
                if len(active_events[key]['time']) > 20:
                    deactivated_events[key] = active_events[key]
                active_events.pop(key, None)
                event_counter += 1
                active_events[event_counter] = {'mask':[], 'time':[], 'xpos':[]}

        dist_matrix, bool_matrix = get_distance(active_events, x_pos_list)

        #     Match x positions to active events
        for i, key in enumerate(active_events):
            for v, x_pos in enumerate(x_pos_list):
                if np.sum(np.isnan(dist_matrix[i, :])): # If we have a nan in this set of distances, save it for later
                    continue
                best_matches = dist_matrix[i, :].argsort()
                for choice in best_matches: 
                    if not bool_matrix[i, choice] and active_events[key]['time'][-1] != tif:
                        bool_matrix[:, choice] = True # x position has been selected, cannot be used twice
                        active_events[key]['xpos'].append(x_pos_list[choice])
                        active_events[key]['mask'].append(mask_list[choice])
                        active_events[key]['time'].append(tif)
                        break

        # Need to check if all x positions have been assigned
        if np.sum(np.invert(bool_matrix)):
            unmatched_exists = True
        else:
            unmatched_exists = False

        while unmatched_exists:
            for i, key in enumerate(active_events):
                # Now we only want to look at the events without distances
                if not np.sum(np.isnan(dist_matrix[i, :])): 
                    continue
                for v, x_pos in enumerate(x_pos_list):
                    if not bool_matrix[i, v]:
                        bool_matrix[:, v] = True # x position has been selected, cannot be used twice
                        active_events[key]['xpos'].append(x_pos_list[v])
                        active_events[key]['mask'].append(mask_list[v])
                        active_events[key]['time'].append(tif)
                        break
            if np.sum(bool_matrix):
                unmatched_exists = False

        # Delete closed events
        prev_tf = tif
        prev_x = x_pos
    for key in active_events:
        deactivated_events[key] = active_events[key]
    
    with open(pickle_filename + '.pkl', 'wb') as f:
        pickle.dump(deactivated_events, f, pickle.HIGHEST_PROTOCOL)

    end = datetime.now()
    dif = end-start
    seconds = dif.total_seconds()
    print('Total time: ', seconds)
    print('Time per frame: ', seconds/len(frange))


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

#### Define file path
#data_base_path = oi_file.data_base_path
for i in range(len(file_index_ar)):


    date = date_ar[i]
    particle_type = particle_type_ar[i]
    file_index = file_index_ar[i]

    file_path = data_base_path + date +'/' + particle_type + channel_type + '/oi/'+ file_index+ '/bin/test_camera_0.raw'


    # Set output file path
    event_file_path = data_base_path + date + '/'+particle_type + channel_type + '/oi/' + file_index + '/events/'

    # Open video
    res_x = 880
    res_y = 140
    fps = 11103
    exp = 1.5
    camera = 0

    ## camera = 0 is raw 16bpp chronos
    oi_vid = oi_file.Video(file_path, res_x, res_y, fps, exp,camera=camera)

    # Load events
    oi_events = pd.read_csv(event_file_path + 'test_camera_0_events.txt').to_numpy().flatten()
    output_file = event_file_path + 'cell_events_101x56_2'

    print('loaded', len(oi_events), 'oi events')

    print('Extracting ', len(oi_events), ' cells')
    extract_fits(oi_vid, output_file, oi_events)