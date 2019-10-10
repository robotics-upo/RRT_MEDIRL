#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy import array
import sys, random, math, pygame, csv
from math import sqrt,cos,sin,atan2
import scipy.misc

import tensorflow as tf
from tensorflow.contrib.image import rotate 
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.estimator.export import export
from tensorflow.python.saved_model import builder as saved_model_builder
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

import cv2

import database_management as db_management



tf.logging.set_verbosity(tf.logging.INFO)

rrt_star_module = tf.load_op_library('./bin/rrt_star.so')
metric_path_module = tf.load_op_library('./bin/metric_path.so')

@ops.RegisterGradient("RRTStar")
def _rrt_star_grad(op, grad):
  label = op.inputs[2]
  result = op.outputs[0]
  grad_in = label - result +sum_eps
  return grad, grad*grad_in, grad

@ops.RegisterGradient("MetricPath")
def _metric_path_grad(op, grad):
  return op.inputs[0]*grad,op.inputs[1]*grad,op.inputs[2]*grad


img = tf.placeholder("float")
label = tf.placeholder("float")
cost = tf.placeholder("float")


img = tf.reshape(img,[1,200,200])
cost = tf.reshape(cost,[1,200,200])
label = tf.reshape(cost,[1,200,200])

path = rrt_star_module.rrt_star(img,cost,label,0) 
dist_paths_rrt, dissimilarity =  metric_path_module.metric_path( path, label, img,0)

init = tf.global_variables_initializer()  


data = db_management.TestCNN("","")
map_costmap = data.load_image_grey("resources/costmap.png")
map_image = data.preprocess_image_grey("resources/img_map.png")
map_label = data.load_image_grey("resources/label.png")
map_costmap = (map_costmap[:,:,:,0])#.reshape(1,200,200)
map_image = (map_image[:,:,:,0])#.reshape(1,200,200)
map_label = (map_label[:,:,:,0])#.reshape(1,200,200)


print(dist_paths_rrt, dissimilarity)


with tf.contrib.tfprof.ProfileContext('/tmp/train_dir') as pctx:
	with tf.Session() as sess:  
		sess.run(init)	

		result = sess.run(path, feed_dict={img: map_image, cost: map_costmap, label: map_label})
		print(result)
		result2 = np.squeeze(result[0,:,:])*0.5 + map_image
		scipy.misc.imsave('resources/MapAndPath.jpg', np.squeeze(result2))
		scipy.misc.imsave('resources/Path.jpg', np.squeeze(np.squeeze(result[0,:,:])))

