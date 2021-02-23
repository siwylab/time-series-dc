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
from skimage.morphology import convex_hull

# Program specific
PORE_STATS_BASE_DIRECTORY = 'C://Users//codyt//Documents//repos//pore_stats//pore_stats//oi_3'
sys.path.append(PORE_STATS_BASE_DIRECTORY)
import optical_imaging as oi
import oi_file
import image_processing 

#Calculate postion and deformation from mask. Input is dataframe (.pkl) from event_detection_mrcnn and stage file
def calculate_df(event_path,oi_stage):

    
    #initialize lists to hold data across events
    mask_array = []
    time_array = []
    perimeter_array = []
    area_array = []
    circ_array = []
    deform_array = []
    r_um_array = []
    xcm_um_array, ycm_um_array = [],[]
    tf_array = []
    event = []
    
    perimeter_array_cx = []
    area_array_cx = []
    circ_array_cx = []
    deform_array_cx = []
    r_um_array_cx = []
    
    ellipse_xc_um_array = []
    ellipse_yc_um_array = []
    ellipse_a_array, ellipse_b_array = [],[]
    aspect_array = []
    r_um_elps_array = []
    
    df_raw = pd.read_pickle(event_path)   
    
    for j in df_raw.keys():

        num = len(df_raw[j]['mask'])
        xcm_um = np.empty(num)
        ycm_um = np.empty(num)
        area = np.empty(num)
        per = np.empty(num)
        circ = np.empty(num)
        deform = np.empty(num)
        r_um = np.empty(num)
        tf = np.empty(num)
        masks = []
        
        area_cx = np.empty(num)
        per_cx = np.empty(num)
        circ_cx = np.empty(num)
        deform_cx = np.empty(num)
        r_um_cx = np.empty(num)
        
        ellipse_xc_um = np.empty(num)
        ellipse_yc_um = np.empty(num)
        ellipse_a = np.empty(num)
        ellipse_b = np.empty(num)
        aspect = np.empty(num)
        r_um_elps = np.empty(num)
        
        
        for i in range(num):

            frame = np.reshape(df_raw[j]['mask'][i],(140,880))
            masks.append(frame)
            
            hull = convex_hull.convex_hull_image(frame)
            
            
            xpix = np.where(frame==1)[1]
            ypix  = np.where(frame==1)[0]
            xc_pix ,yc_pix  = oi_stage.get_channel_coordinates(xpix, ypix)

            xc_um  = oi_stage.pixels_to_meters(xc_pix)
            yc_um  = oi_stage.pixels_to_meters(yc_pix)
            xcm_um[i]  = np.mean(xc_um)
            ycm_um[i]  = np.mean(yc_um)

            area[i]  = np.sum(frame)
            per[i]  = measure.perimeter(frame)
            circ[i]  = 2*np.sqrt(np.pi)*np.sqrt(area[i])/per[i]
            deform[i]  = 1 - circ[i]
            r_um[i]  = (oi_stage.pixels_to_meters(np.sqrt(area[i]/np.pi))+oi_stage.pixels_to_meters(per[i]/(2*np.pi)))/2
            tf[i] = df_raw[j]['time'][i]
            
            area_cx[i]  = np.sum(hull)
            per_cx[i]  = measure.perimeter(hull)
            circ_cx[i]  = 2*np.sqrt(np.pi)*np.sqrt(area[i])/per[i]
            deform_cx[i]  = 1 - circ[i]
            r_um_cx[i]  = (oi_stage.pixels_to_meters(np.sqrt(area[i]/np.pi))+oi_stage.pixels_to_meters(per[i]/(2*np.pi)))/2
    
            
            #ellipse_pixels = np.where(processed_frame == 1)
            ellipse = oi.fit_ellipse_image_aligned(xpix, ypix)
            ellipse_x, ellipse_y = oi.get_ellipse_center(ellipse)
            ellipse_xc,ellipse_yc = oi_stage.get_channel_coordinates(ellipse_x,ellipse_y)
            ellipse_xc_um[i] = oi_stage.pixels_to_meters(ellipse_xc)
            ellipse_yc_um[i] = oi_stage.pixels_to_meters(ellipse_yc)
            ellipse_a[i], ellipse_b[i] = oi.get_ellipse_axes_lengths(ellipse)
            aspect[i] = ellipse_a[i]/ellipse_b[i]
            r_um_elps[i] = np.sqrt(ellipse_a[i]*ellipse_b[i])
            
            
        event.append(j)
        mask_array.append(masks)
        time_array.append(tf)
        perimeter_array.append(per)
        area_array.append(area)
        circ_array.append(circ)
        deform_array.append(deform)
        r_um_array.append(r_um)
        xcm_um_array.append(xcm_um) 
        ycm_um_array.append(ycm_um)
        
        
        perimeter_array_cx.append(per_cx)
        area_array_cx.append(area_cx)
        circ_array_cx.append(circ_cx)
        deform_array_cx.append(deform_cx)
        r_um_array_cx.append(r_um_cx)

        ellipse_xc_um_array.append(ellipse_xc_um)
        ellipse_yc_um_array.append(ellipse_yc_um)
        ellipse_a_array.append(ellipse_a)
        ellipse_b_array.append(ellipse_b)
        aspect_array.append(aspect)
        r_um_elps_array.append(r_um_elps)

    d = {'event':event,'tf':time_array,'mask':mask_array,'perimeter':perimeter_array,'area':area_array,
         'circ':circ_array,'deform':deform_array,'r_um':r_um_array,'xcm_um':xcm_um_array,'yc_um':ycm_um_array,
        'perimeter_cx':perimeter_array_cx,'area_cx':area_array_cx,'circ_cx':circ_array_cx,'deform_cx':deform_array_cx,
         'r_um_cx':r_um_array_cx,'xc_um_el':ellipse_xc_um_array,'yc_um_el':ellipse_yc_um_array,'a':ellipse_a_array,
         'b':ellipse_b_array,'aspect':aspect_array,'r_um_el':r_um_elps_array}   

    df = pd.DataFrame(data=d)
    df['cell'] = particle_type
    df['date'] = date
    df['run'] = file_index
    
    #Remove events with empty time series. Artifact of event detection
    df = df[df.tf.map(lambda a: a.size!=0)]
    
    return df

# Open video
res_x = 880
res_y = 140
fps = 11103
exp = 1.5
camera = 0


#Define Videos for calculations
data_base_path = 'D:/'
date_ar =  ['10-8-20']*6 + ['11-3-20']*5+['11-5-20']*5
particle_type_ar = ['hl60']*3+['hl60d']*3+['hl60']+['hl60d']*4+['hl60']*4+['hl60d']
channel_type = '/25_50_25x150'
file_index_ar = ['0','1','2']*2+['0']+['0','1','2','3']+['0','1','2','3']+['2']

assert len(date_ar)==len(particle_type_ar)==len(file_index_ar)


for i in range(len(date_ar)):

    date = date_ar[i]
    particle_type = particle_type_ar[i]
    file_index = file_index_ar[i]


    vid_file_path = data_base_path + date + '/'+particle_type + channel_type + '/oi/'+ file_index+ '/bin/test_camera_0.raw'
    stage_file_path = data_base_path + date +'/'+ particle_type + channel_type + '/oi/'+ file_index+'/stage/stage_0.txt'
    event_file_path = data_base_path + date +'/'+ particle_type + channel_type + '/oi/'+ file_index+'/events/cell_events_101x56_2.pkl'
    output_file_path = data_base_path + date +'/'+ particle_type + channel_type + '/oi/'+ file_index + '/df/'
    
    if os.path.isfile(output_file_path + 'calcs101x56_2'):
        continue
        
    oi_vid = oi_file.Video(vid_file_path, res_x, res_y, fps, exp,camera=camera)
    
    template_frame = oi_vid.get_frame(0)
    cs = oi.load_stage_file(stage_file_path)
    

    oi_stage = oi.Stage(template_frame, cs[0], cs[1], cs[2], cs[3])
    
    df = calculate_df(event_file_path,oi_stage)
    
    if not os.path.isdir(output_file_path):
        os.mkdir(output_file_path)
    df.to_pickle(output_file_path+'calcs101x56_2')


