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
from skimage.draw import rectangle
import importlib
import splitfolders
from scipy.stats import gaussian_kde

import plots
import df_utils
import features




def aspect_dvdx_x(df,output_path,event_num=193,save = False):



	df_cm = pd.read_csv('C://Users//codyt//Desktop//comsol//sheath_particle//bare_channel//velocity.csv',skiprows=8)[0:-2]
	df_cm = df_cm.drop(df_cm.columns[-1],axis=1)
	df_cm = df_cm.rename(columns={df_cm.columns[4]: 'u',df_cm.columns[5]: 'x', df_cm.columns[6]: "dudx",df_cm.columns[3]:'sr'})
	df_cm = df_cm.sort_values(by ='x' )

	df_cm['v_avg'] = df_cm.u.rolling(5).mean()
	df_cm['dv_avg'] = df_cm.dudx.rolling(5).mean()
	df_cm['sr_avg'] = df_cm.sr.rolling(5).mean()

	fig, axs = plt.subplots(2,sharex=False,sharey=False,figsize=(20,10))
	#fig.suptitle('Vertically stacked subplots')

	row = df[df.cell=='hl60d'].iloc[event_num]

	axs[0].scatter(row.xcm_um,row.aspect,color='black',s=32)
	axs[0].set_xlim([-20,170])
	axs[0].set_ylim([0.8,1.6])

	axs[0].set_ylabel('Aspect Ratio',size=20)


	# trend = df.iloc[event_num].x_poly1
	# trendpoly = np.poly1d(trend) 
	# xs = df.iloc[event_num].xcm_um[df.iloc[event_num].nar1_max_arg-1:df.iloc[event_num].cav1_min_arg+1]
	# axs[0].plot(xs,trendpoly(xs))

	fig.text(0.5, 0.04, 'X position ($\mu$m)', ha='center',size=20)


	xposition = [0,150]
	for xc in xposition:
	    axs[0].axvline(x=xc, color='r', linestyle='--',lw=3)
	    axs[1].axvline(x=xc, color='r', linestyle='--',lw=3)

	axs[1].plot(df_cm.x*1000,df_cm.dv_avg/1000,color='black',linewidth=2)
	axs[1].set_xlim([-20,170])
	axs[1].set_ylim([-210,210])

	axs[1].set_ylabel('dv/dx (1/s)',size=20)


	axs[0].axvspan(-20, 15, alpha=0.1, color='red')
	axs[1].axvspan(-20, 15, alpha=0.1, color='red')

	axs[0].axvspan(15, 75, alpha=0.1, color='green')
	axs[1].axvspan(15, 75, alpha=0.1, color='green')

	axs[0].axvspan(75, 125, alpha=0.1, color='yellow')
	axs[1].axvspan(75, 125, alpha=0.1, color='yellow')

	axs[0].axvspan(125, 170, alpha=0.1, color='blue')
	axs[1].axvspan(125, 170, alpha=0.1, color='blue')

	if save:
		plt.savefig(output_path)
	else:
		plt.show()

def aspect_shear_x(df,output_path,event_num=193,save=False):


	df_cm = pd.read_csv('C://Users//codyt//Desktop//comsol//sheath_particle//bare_channel//velocity.csv',skiprows=8)[0:-2]
	df_cm = df_cm.drop(df_cm.columns[-1],axis=1)
	df_cm = df_cm.rename(columns={df_cm.columns[4]: 'u',df_cm.columns[5]: 'x', df_cm.columns[6]: "dudx",df_cm.columns[3]:'sr'})
	df_cm = df_cm.sort_values(by ='x' )

	df_cm['v_avg'] = df_cm.u.rolling(5).mean()
	df_cm['dv_avg'] = df_cm.dudx.rolling(5).mean()
	df_cm['sr_avg'] = df_cm.sr.rolling(5).mean()

	fig, axs = plt.subplots(2,sharex=False,sharey=False,figsize=(20,10))
	#fig.suptitle('Vertically stacked subplots')

	row = df[df.cell=='hl60d'].iloc[193]

	axs[0].scatter(row.xcm_um,row.aspect)
	axs[0].set_xlim([-20,170])
	axs[0].set_ylim([0.8,1.6])

	axs[0].set_ylabel('Aspect Ratio',size=20)

	fig.text(0.5, 0.04, 'X position ($\mu$m)', ha='center',size=20)


	xposition = [0,150]
	for xc in xposition:
	    axs[0].axvline(x=xc, color='r', linestyle='--',lw=3)
	    axs[1].axvline(x=xc, color='r', linestyle='--',lw=3)

	axs[1].plot(df_cm.x*1000,df_cm.sr_avg/1000,color='black',linewidth=2)
	axs[1].set_xlim([-20,170])
	axs[1].set_ylim([0,400])

	axs[1].set_ylabel('shear rate (1/s)',size=20)

	plt.show()


	axs[0].axvspan(-20, 15, alpha=0.1, color='red')
	axs[1].axvspan(-20, 15, alpha=0.1, color='red')

	axs[0].axvspan(15, 75, alpha=0.1, color='green')
	axs[1].axvspan(15, 75, alpha=0.1, color='green')

	axs[0].axvspan(75, 125, alpha=0.1, color='yellow')
	axs[1].axvspan(75, 125, alpha=0.1, color='yellow')

	axs[0].axvspan(125, 170, alpha=0.1, color='blue')
	axs[1].axvspan(125, 170, alpha=0.1, color='blue')

	if save:
		plt.savefig(output_path)
	else:
		plt.show()

def feat_x(df,output_path,cell,y_label,feature='aspect',save=False):
	
	axis_size = 18
	for i,row in df[df.cell==cell].iterrows():
	    plt.scatter(row.xcm_um,row[feature],color='black',alpha=0.2,s=16)
	xposition = [0,150]
	for xc in xposition:
	    plt.axvline(x=xc, color='r', linestyle='--',lw=3)
	    
	plt.xlabel('x$_{c}$ ($\mu$m)',size=axis_size)
	plt.ylabel(y_label,size=axis_size)

	plt.tick_params(labelsize=14)
	plt.xlim((-20,170))
	#plt.ylim((.75,2))
	plt.tight_layout()

	if save:
		plt.imsave(output_path)
	else:
		plt.show()

def pop_means_sem(df,y_label,feat,output_path='D://',save=False):
	
	hlidx = df.cell == 'hl60'
	hldidx = df.cell == 'hl60d'

	x = [1,2]
	y = [df[hlidx][feat].mean(),df[hldidx][feat].mean()]
	e = [stats.sem(df[feat][hlidx]),stats.sem(df[feat][hldidx])]

	temp_df = pd.DataFrame({'cell' : x, 'aspect' : y, 'sd_err' : e})

	g = sns.FacetGrid(data=temp_df, aspect=1, height=6)        
	g.map(plt.errorbar, "cell", "aspect", "sd_err", marker="o",linestyle='',markersize='6', capsize=4, elinewidth=2)


	# statistical annotation
	#x1, x2 = 1, 2   
	#y1, h, col = 1.45, .01, 'k'
	#plt.plot([x1, x1, x2, x2], [y1, y1+h, y1+h, y1], lw=1.5, c=col)
	#plt.text((x1+x2)*.5, y1+h, "***", ha='center', va='bottom', color=col)


	# place the ticks at center by widening the plot
	plt.xlim((0, 3))
	# fix ticks at the number encoding for each class
	g.fig.axes[0].xaxis.set_ticks([1, 2])
	# name the numbers
	g.fig.axes[0].xaxis.set_ticklabels(['hl60', 'hl60d'])
	plt.ylabel(y_label,size=20)
	plt.xlabel('Cell',size=20)

	#plt.ylim(1.2,1.5)
	#plt.show()
	if save:
		plt.savefig(output_path)
	else:
		plt.show()

def temp_sem(df,y_label,feat,output_path='D://',save=False):
	
	hlidx = df.cell == 'hl60'
	hldidx = df.cell == 'hl60d'

	x = ['HL 60','HL 60d']
	y = [df[hlidx][feat].mean(),df[hldidx][feat].mean()]
	e = [stats.sem(df[feat][hlidx]),stats.sem(df[feat][hldidx])]

	temp_df = pd.DataFrame({'cell' : x, 'aspect' : y, 'sd_err' : e})

	       
	plt.errorbar([1,2], y, e, marker="o",linestyle='',markersize='6', capsize=4)




	# place the ticks at center by widening the plot
	plt.xlim((0, 3))
	# fix ticks at the number encoding for each class
	plt.xticks([1,2],['hl60','hl60d'])
	# name the numbers
	
	plt.ylabel(y_label,size=20)
	plt.xlabel('Cell',size=20)

	#plt.ylim(1.2,1.5)
	#plt.show()
	if save:
		plt.savefig(output_path)
	else:
		plt.show()


def pop_violin(df,output_path, feat, save=False):

	ax = sns.violinplot(x="cell", y=feat, hue="cell",
	                data=df, palette="muted")
	plt.xlabel(fontsize=10)

	if save:
		plt.savefig(output_path)
	else:
		plt.show()

def pop_hist(df,output_path,feat,save=False):

	hlidx = df.cell == 'hl60'
	hldidx = df.cell == 'hl60d'


	sns.distplot(df[hlidx][feat],label = 'hl60')
	sns.distplot(df[hldidx][feat],label='hl60d')

	plt.legend()

	if save:
		plt.savefig(output_path)
	else:
		plt.show()
	

def heatmat_pop(df,xfeat,yfeat,ylabel,xlabel,output_path='D://',save=False):

	fig, (ax1,ax2) = plt.subplots(1,2,sharex=True,sharey=True,figsize=(20,10))
	#fig.suptitle('Vertically stacked subplots')

	x = df[df.cell=='hl60'][xfeat].to_numpy()
	y = df[df.cell=='hl60'][yfeat].to_numpy()

	# Calculate the point density
	xy = np.vstack([x,y])
	z = gaussian_kde(xy)(xy)

	ax1.scatter(x, y, c=z, s=100, edgecolor=None)
	ax1.set_xlabel(xlabel,fontsize=20)
	#axs[0].set_xlim([-20,170])
	#axs[0].set_ylim([0.8,1.6])

	ax1.set_ylabel(ylabel,fontsize=20)
	ax1.set_title('HL 60', fontsize=20)

	x = df[df.cell=='hl60d'][xfeat].to_numpy()
	y = df[df.cell=='hl60d'][yfeat].to_numpy()

	# Calculate the point density
	xy = np.vstack([x,y])
	z = gaussian_kde(xy)(xy)

	ax2.scatter(x, y, c=z, s=100, edgecolor=None)
	ax2.set_xlabel(xlabel,fontsize=20)
	ax2.set_title('HL 60d',fontsize=20)

	#ax2.set_ylim([.8,2.2])
	ax2.set_xlim([5,8])


	ax1.annotate('n='+str(len(df[df.cell=='hl60'])), xy=(.95, .9), xycoords='axes fraction', fontsize=16,
	            xytext=(-5, 5), textcoords='offset points',
	            ha='right', va='bottom')

	ax2.annotate('n='+str(len(df[df.cell=='hl60d'])), xy=(.95, .9), xycoords='axes fraction', fontsize=16,
	            xytext=(-5, 5), textcoords='offset points',
	            ha='right', va='bottom')

	if save:
		plt.imsave(output_path)
	else:
		plt.show()

def heatmap_pop_full(df,xfeat,yfeat,ylabel,xlabel,output_path='D://',save=False):

	#fig, ax = plt.subplots(nrows=len(yfeat), ncols=3,sharex='none',sharey='row',figsize=(20,30))

	fig = plt.figure(figsize=(20,20))


	for row in range(len(yfeat)):

		ax1 = fig.add_subplot(len(yfeat),3,(1+row*3))
		ax2 = fig.add_subplot(len(yfeat),3,(2+row*3) ,sharey = ax1)
		ax3 = fig.add_subplot(len(yfeat),3,(3+row*3))

		x = df[df.cell=='hl60'][xfeat].to_numpy()
		y = df[df.cell=='hl60'][yfeat[row]].to_numpy()

		# Calculate the point density
		xy = np.vstack([x,y])
		z = gaussian_kde(xy)(xy)
		ax1.scatter(x, y, c=z, s=100, edgecolor=None)
		ax1.set_xlabel(xlabel)
		
		
		ax1.set_ylabel(ylabel[row],fontsize=20)


		ax1.annotate('n='+str(len(df[df.cell=='hl60'])), xy=(.95, .9), xycoords='axes fraction', fontsize=16, 
			xytext=(-5, 5), textcoords='offset points',
            ha='right', va='bottom')

		x = df[df.cell=='hl60d'][xfeat].to_numpy()
		y = df[df.cell=='hl60d'][yfeat[row]].to_numpy()

		# Calculate the point density
		xy = np.vstack([x,y])
		z = gaussian_kde(xy)(xy)
		ax2.scatter(x, y, c=z, s=100, edgecolor=None)
		ax2.set_xlabel(xlabel)
		#ax2.set_ylabel(ylabel,fontsize=20)


		ax2.annotate('n='+str(len(df[df.cell=='hl60d'])), xy=(.95, .9), xycoords='axes fraction', fontsize=16, 
			xytext=(-5, 5), textcoords='offset points',
            ha='right', va='bottom')


		hlidx = df.cell == 'hl60'
		hldidx = df.cell == 'hl60d'

		x = ['HL 60','HL 60d']
		y = [df[hlidx][yfeat[row]].mean(),df[hldidx][yfeat[row]].mean()]
		e = [stats.sem(df[hlidx][yfeat[row]]),stats.sem(df[hldidx][yfeat[row]])]
     
		ax3.errorbar([1,2], y, e, marker="o",linestyle='',markersize='6', capsize=4)

		# place the ticks at center by widening the plot
		ax3.set_xlim((0, 3))
		# fix ticks at the number encoding for each class
		ax3.set_xticks([1,2])
		ax3.set_xticklabels(['hl60','hl60d'],fontsize=20)
		# name the numbers
		
		#ax3.set_ylabel(y_label,size=20)
		#ax3.set_xlabel('Cell',fontsize=20)

		if row==0:
			ax1.set_title('HL 60', fontsize=20)
			ax2.set_title('HL 60d', fontsize=20)

		ax2.set_xlabel(xlabel,fontsize=20)
		ax1.set_xlabel(xlabel,fontsize=20)

	if save:
		plt.savefig(output_path)
	else:
		plt.show()


def Figure2(df,output_path='D://',save=False):


	fig, axes = plt.subplots(3,3,sharex=True,sharey=True,figsize=(20,10))
