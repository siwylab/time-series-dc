
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
from scipy import signal,stats
from scipy.interpolate import interp1d
from skimage import measure
import df_utils
import features
import seaborn as sns

# Program specific
PORE_STATS_BASE_DIRECTORY = 'C://Users//codyt//Documents//repos//pore_stats//pore_stats//oi_3'

sys.path.append(PORE_STATS_BASE_DIRECTORY)
import optical_imaging as oi
import oi_file
import image_processing


def plot_dates(df,features,save=False):

	for feat in features:
	    ax = sns.violinplot(x="date", y=feat, hue="cell",
	                    data=df, palette="muted")
	    if save:

	    	plt.savefig('./figures/violin_days/' + feat)
	    else:
	    	plt.show()

def plot_cells(df,features,save=False):

	for feat in features:

		sns.catplot(x="date", y=feat, hue="run",col = 'cell',data=df,kind='violin', palette="muted",legend=False)
		plt.legend(loc='upper right')
		plt.tight_layout()

	if save:

		plt.savefig('./figures/violin_days/' + feat)

	else:
		plt.show()

def plot_pop(df,feature,save=False):

	num_cells = df.cell.nunique()
	str_cells = df.cell.unique()

	x = []
	y = []
	e = []

	for i in range(num_cells):

		idx = df.cell == str_cells[i]

		x.append(i+1)

		y.append(df[feature][idx].mean())

		e.append(stats.sem(df[feature][idx]))

	temp_df = pd.DataFrame({'cell' : x, 'aspect' : y, 'sd_err' : e})

	g = sns.FacetGrid(data=temp_df, aspect=1, height=6)        
	g.map(plt.errorbar, "cell", "aspect", "sd_err", marker="o",linestyle='',markersize='6', capsize=4, elinewidth=2)


	# statistical annotation
	#x1, x2 = 1, 2   
	#y1, h, col = 1.45, .01, 'k'
	#plt.plot([x1, x1, x2, x2], [y1, y1+h, y1+h, y1], lw=1.5, c=col)
	#plt.text((x1+x2)*.5, y1+h, "***", ha='center', va='bottom', color=col)


	# place the ticks at center by widening the plot
	plt.xlim((np.min(x)-1, np.max(x)+1))
	# fix ticks at the number encoding for each class
	g.fig.axes[0].xaxis.set_ticks(x)
	# name the numbers
	g.fig.axes[0].xaxis.set_ticklabels(str_cells)
	plt.ylabel(feature,size=20)
	plt.xlabel('Cell',size=20)
	#plt.ylim(1.2,1.5)
	plt.show()
	#plt.savefig('./figures/101x56_2/aspect1_ratio')