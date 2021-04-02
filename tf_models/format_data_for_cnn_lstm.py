import pandas as pd
import os
import numpy as np
import pickle
import json
import scipy
import matplotlib.pyplot as plt
import shutil
from skimage.measure import regionprops
from skimage.morphology import convex_hull_image
import cv2 as cv
import tensorflow as tf
import skimage
from itertools import compress
from sklearn.model_selection import train_test_split


# Load data
df = pd.read_pickle('dataframe.pkl')

#  Extract minimasks and filter sequences
# - Since the masks are boolean arrays, our information is limited to the bounding box around our cell.
# We'll slice out that bounding box, pad to a size larger than any cell, and center the cells
# - Additionally, we need to pad the sequence of masks to all be the same length.
# While some sequences are longer than the max length, we'll remove those later.
# - We also want to use convex hull from skimage to smooth out some of our fits


def bbox(row):
    mask_list = []
    max_len = 40
    x_min = -20
    x_max = 180
    x_start = np.argmin(np.abs(row['xcm_um']-x_min))
    x_stop = np.argmin(np.abs(row['xcm_um']-x_max))
    image = row['mask'][x_start:x_stop]
    for i, mask in enumerate(image):
        x_min = np.min(np.where(mask)[0])
        x_max = np.max(np.where(mask)[0])
        y_min = np.min(np.where(mask)[1])
        y_max = np.max(np.where(mask)[1])
        width = x_max - x_min
        height = y_max - y_min
        # Apply convex hull to avoid introducing bias and improve fit
        chull = convex_hull_image(mask)
        screened = chull[x_min:x_max, y_min:y_max]
        screened = tf.expand_dims(tf.convert_to_tensor(screened), axis=-1)
        screened = tf.image.pad_to_bounding_box(screened, int((90-width)/2), int((90-height)/2), 90, 90).numpy()
        
        rr, cc = skimage.draw.ellipse(r=45, c=45, r_radius=row['b'][i]*2.0, c_radius=row['a'][i]*2.0, shape=(90, 90))
        ellipse = np.zeros((90, 90, 1))
        ellipse[rr, cc] = 1
        mask_list.append(np.stack((screened, ellipse), axis=2)[:, :, :, 0])
    if len(mask_list) < max_len:
        for _ in range(max_len-len(mask_list)):
            mask_list.append(np.zeros((90, 90, 2), dtype=bool))
    return np.stack(mask_list)


# Apply function defined above
df['minimask'] = df.apply(bbox, axis=1)

# Extract x,y from dataframe
x = df['minimask'].to_numpy()
y = np.array(df['cell'] == 'hl60', dtype=int)


# Reformat x, y and remove sequences that are erroneously long
x_list = []
y_list = []
for i, yi in zip(range(x.shape[0]), y):
    if x[i].shape != (40, 90, 90, 2):
        continue
    x_list.append(x[i])
    y_list.append(yi)
x_list = np.stack(x_list)
y_list = np.stack(y_list)


np.save('x_data', x_list)
np.save('y_data', y_list)