#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import numpy as np
import os
from numpy import array
from keras.preprocessing.image import load_img, img_to_array, array_to_img #,flip_axis
import csv


class TestCNN(object):

  def __init__(self, r, s):
    self.dir = r
    self.data_dir = r + 'input/'
    self.labels_dir = r + 'labels/'  


    if os.path.isdir( r + 'costmaps/' ): 
		self.costmaps_dir = r + 'costmaps/'  
    self.save_dir = s
    np.set_printoptions(precision=3, threshold=10000, linewidth=10000)
    
  #--------------------------------------------------------------------------
  def preprocess_image_grey(self, image_path):
    img = load_img(image_path, grayscale=True)
    img = img_to_array(img)

    img[img <=15] = 0     #free: 0
    img[img >=235] = 1    #path: 255
    img[img >=184] = 0.8  #obstacles: 204
    img[img >=133] = 0.4  #people front: 153
    img[img >=82] = 0.2   #people back: 102
    img[img >=43] = 0.6   # goal: 63
    
    img = np.expand_dims(img, axis=0) 
    return img
    
    
    

  def load_image_grey(self, image_path):
    img = load_img(image_path, grayscale=True)
    img = img_to_array(img)/255.0
    
    img = np.expand_dims(img, axis=0) 
    return img


  def load_costmap_csv(self, image_path):
	  
    img = load_img(image_path, grayscale=True)
    img = img_to_array(img)/255.0
    
    img = np.expand_dims(img, axis=0) 
    return img




  #--------------------------------------------------------------------------
  def load_data(self, load_npy=False, save_npy=False, norm=False, six=False):

    if load_npy==True:
      self.data = np.load((self.dir+ 'data.npy'))
      self.labels = np.load((self.dir+ 'labels.npy'))
      if os.path.isfile(self.dir+ 'costmaps.npy'):
        self.costmaps = np.load((self.dir+ 'costmaps.npy'))
        
        
    else:
      infiles = [f for f in os.listdir(self.data_dir) if os.path.isfile(self.data_dir + f)]
      i = 1
      for f in infiles:
        infile = self.data_dir + f
        label = self.labels_dir + f
        image_in = self.preprocess_image_grey(infile)
        image_label = self.load_image_grey(label)
        
        if i == 1:
          self.data = np.array(image_in)
          self.labels = np.array(image_label)       
          
        else: 
          self.data = np.append(self.data, image_in, axis=0)
          self.labels = np.append(self.labels, image_label, axis=0)

	
        if hasattr(self,'costmaps_dir'):			
			costmap_file =  self.costmaps_dir + f[:-4]+".csv"  
			csv_costmap = np.genfromtxt(costmap_file, delimiter=';')     
			csv_costmap =  np.expand_dims(csv_costmap[:,:-1], axis=0)   
			if i == 1:
				self.costmaps = np.array(csv_costmap)    
			else:
				self.costmaps = np.append(self.costmaps, csv_costmap, axis=0) 	  
        i = i+1
        
        if i%50 == 0:
          print("i = ",i)
        
    if(save_npy==True):
      file_data = self.dir + 'data.npy'
      np.save(file_data, self.data)
      file_labels = self.dir + 'labels.npy'
      np.save(file_labels, self.labels)
      if hasattr(self, 'costmaps'): 
        file_costmaps = self.dir + 'costmaps.npy'
        np.save(file_costmaps, self.costmaps)        
 
  def remove_objective(self):
    self.data_obj = np.copy(self.data)
    self.data_obj[self.data_obj ==0.6] = 0.0
    
    
def remove_objective(img):
    data_obj = np.copy(img)
    data_obj[data_obj ==0.6] = 0.0   
    return data_obj
