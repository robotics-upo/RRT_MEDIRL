#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy import array
import os
from os import listdir
from os.path import isfile, join
import sys, random, math, pygame
from math import sqrt,cos,sin,atan2
import csv
import time
from keras.preprocessing.image import load_img, img_to_array, array_to_img, flip_axis
import scipy.misc
import shutil

import tensorflow as tf
from tensorflow.contrib.image import rotate 
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.estimator.export import export
from tensorflow.python.saved_model import builder as saved_model_builder


learning_rate = 0.00001
num_steps = 10000
batch_size = 20
N_REP_NET = 10

dropout = 0.2

tf.logging.set_verbosity(tf.logging.INFO)

sess = tf.Session()

rrt_star_module = tf.load_op_library('./rrt_star.so')
metric_path_module = tf.load_op_library('./metric_path.so')

@ops.RegisterGradient("RRTStar")
def _rrt_star_grad(op, grad):
  label = op.inputs[2]
  result = op.outputs[0]
  grad_in = (label-result)/tf.reduce_sum(label) - (1 - label - result)/tf.reduce_sum(1.0-label)
  return grad, grad*grad_in,grad, op.inputs[3]

@ops.RegisterGradient("MetricPath")
def _metric_path_grad(op, grad):
  return op.inputs[0]*grad,op.inputs[1]*grad,op.inputs[2]*grad


class TestCNN(object):

  def __init__(self, r, s):
    self.dir = r
    self.data_dir = r + 'input/'
    self.labels_dir = r + 'labels/'  
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

  #--------------------------------------------------------------------------
  def load_data(self, load_npy=False, save_npy=False, norm=False, six=False):

    if load_npy==True:
      self.data = np.load((self.dir+ 'data.npy'))
      self.labels = np.load((self.dir+ 'labels.npy'))

    else:
      infiles = [f for f in listdir(self.data_dir) if isfile(join(self.data_dir, f))]
      i = 1
      for f in infiles:
        infile = self.data_dir + f
        label = self.labels_dir + f
        image_in = self.preprocess_image_grey(infile)
        image_label = self.preprocess_image_grey(label)
        
        if i == 1:
          self.data = np.array(image_in)
          self.labels = np.array(image_label)       
          
        else: 
          self.data = np.append(self.data, image_in, axis=0)
          self.labels = np.append(self.labels, image_label, axis=0)
        i = i+1
        
        if i%50 == 0:
			print("i = ",i)
        
    if(save_npy==True):
      file_data = self.dir + 'data.npy'
      np.save(file_data, self.data)
      file_labels = self.dir + 'labels.npy'
      np.save(file_labels, self.labels)    

  def remove_objective(self):
    self.data_obj = np.copy(self.data)
    self.data_obj[self.data_obj ==0.6] = 0.0

def conv_net(x_dict, y_dict, dropout, reuse, is_training):

  with tf.variable_scope('ConvNet', reuse=reuse):

      img = tf.reshape(x_dict['images'], shape=[-1, 200, 200, 1])
      img_obj = tf.reshape(x_dict['images_obj'], shape=[-1, 200, 200, 1])
            
      x = tf.layers.conv2d(img_obj, 16, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.layers.conv2d(x, 14, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   

      x = tf.layers.conv2d(x, 12, 7, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.layers.conv2d(x, 10, 7, activation=tf.nn.relu, padding='same')

      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      
      x = tf.layers.conv2d(x, 8, 5, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.layers.conv2d(x, 6, 5, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)  
       
      x = tf.layers.conv2d(x, 4, 3, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)
      x = tf.layers.conv2d(x, 2, 3, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)            
      
      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)      
      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')  
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.layers.conv2d(x, 1, 1, activation=tf.nn.relu, padding='same')
      
      x += 0.000000000001
      x = tf.minimum( x, 10000000)
      x = tf.nn.l2_normalize(x,0)
      x += 0.000000000001

      cost_label = tf.reshape(x, [-1,200, 200])
  return cost_label



def training(features, labels, mode): 
      
  dist_min = 9999999.99
  dist_max = 0.0  
  dist = []
     
  cost_train = conv_net(features, labels, dropout, reuse=False, is_training=True)
  cost_test = conv_net(features, labels, dropout, reuse=True, is_training=False)
  
  logits_test = rrt_star_module.rrt_star(features['images'],cost_test,features['labels'], 0) 
  dist_paths_rrt =  metric_path_module.metric_path(logits_test,features['labels'], features['images'],0)
  mse_rrt = tf.losses.mean_squared_error(logits_test,features['labels'])
  dist.append(dist_paths_rrt)   
     
  dist_min = tf.cond(dist_paths_rrt<dist_min,lambda: dist_paths_rrt, lambda: dist_min )
  dist_max = tf.cond(dist_paths_rrt>dist_max,lambda: dist_paths_rrt, lambda: dist_max )
       
  for i in range(N_REP_NET-1):
    logits_test_i = rrt_star_module.rrt_star(features['images'],cost_test,features['labels'], i+1)   
    dist_paths_rrt_aux = metric_path_module.metric_path(logits_test_i,features['labels'],features['images'],0)
    mse_rrt += tf.losses.mean_squared_error(logits_test_i,features['labels'])
    logits_test += logits_test_i     
    dist.append(dist_paths_rrt_aux)
    dist_paths_rrt += dist_paths_rrt_aux
    dist_min = tf.cond(dist_paths_rrt_aux<dist_min,lambda: dist_paths_rrt_aux, lambda: dist_min )
    dist_max = tf.cond(dist_paths_rrt_aux>dist_max,lambda: dist_paths_rrt_aux, lambda: dist_max )


  logits_test /= N_REP_NET
  dist_paths_rrt /= N_REP_NET
  mse_rrt /= N_REP_NET
  
  tf_dist = tf.stack(dist)
  
  size_dist = float(N_REP_NET)


  mse_cost = tf.losses.mean_squared_error(cost_test,1.0-features['labels'])
  log_likelihood = tf.reduce_sum(tf.multiply(cost_test,features['labels']))/tf.reduce_sum(features['labels']) - tf.reduce_sum(tf.multiply(cost_test,1.0-features['labels']))/tf.reduce_sum(1.0-features['labels'])
  
  loss_op = log_likelihood
  log_image = tf.log(tf.clip_by_value(tf.reshape(cost_test, shape=[-1, 200, 200, 1]),0,0.1))

  tf.summary.image('images', tf.reshape(features['images'], shape=[-1, 200, 200, 1]))  
  tf.summary.image('rrt_in', tf.reshape(cost_test, shape=[-1, 200, 200, 1]))
  tf.summary.image('rrt_in_visible', tf.nn.l2_normalize(log_image,0))
  tf.summary.image('rrt_out', tf.reshape(logits_test, shape=[-1, 200, 200, 1]))   
  tf.summary.image('label', tf.reshape(features['labels'], shape=[-1, 200, 200, 1])) 

  tf.summary.scalar('dist_between_paths', dist_paths_rrt)
  tf.summary.scalar('dist_min', dist_min)
  tf.summary.scalar('dist_max', dist_max)
  tf.summary.scalar('mse_rrt', mse_rrt)
  tf.summary.scalar('mse_cost', mse_cost)
  tf.summary.scalar('log_likelihood', log_likelihood)

  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step()) 

  if mode == tf.estimator.ModeKeys.PREDICT:
    #~ estim_specs = tf.estimator.EstimatorSpec(mode, predictions=cost_test)
    estim_specs = tf.estimator.EstimatorSpec(mode, predictions=logits_test)
  elif mode == tf.estimator.ModeKeys.EVAL:
    metrics = {'dist_between_paths':tf.metrics.mean(dist_paths_rrt), 'dist_min': tf.metrics.mean(dist_min), 'dist_max': tf.metrics.mean(dist_max)}
    estim_specs = tf.estimator.EstimatorSpec( mode, loss=loss_op, eval_metric_ops=metrics)
  else:
    estim_specs = tf.estimator.EstimatorSpec(
	  mode=mode,
	  predictions=logits_test,
	  loss=loss_op,
	  train_op=train_op)
  return estim_specs





if __name__ == '__main__':

  normalize_img_data = True
  use_six_categories = True
  #~ create_npy_files = True
  #~ load_npy_files = False
  create_npy_files = False
  load_npy_files = True
  dataset_dir = '/home/sanson/catkin_ws/src/upo_fcn_learning/data/real_traj_dataset/set1/'
  testset_dir = '/home/sanson/catkin_ws/src/upo_fcn_learning/data/real_traj_dataset/test_set1/'
  save_dir1 = '/home/sanson/catkin_ws/src/upo_fcn_learning/data/real_traj_dataset/set1/'
  save_dir2 = '/home/sanson/catkin_ws/src/upo_fcn_learning/data/real_traj_dataset/test_set1/'
   
  path_save_net = "estimator"

  
  data_node = TestCNN(dataset_dir, save_dir1)
  data_node.load_data(load_npy=load_npy_files, save_npy=create_npy_files, norm=normalize_img_data, six=use_six_categories)
  data_node.remove_objective()
 
  test_node = TestCNN(testset_dir, save_dir2)
  test_node.load_data(load_npy=load_npy_files, save_npy=create_npy_files, norm=normalize_img_data, six=use_six_categories) 
  test_node.remove_objective()
 
      
  # Build the Estimator
  model = tf.estimator.Estimator(training, model_dir=path_save_net,config=tf.contrib.learn.RunConfig(save_checkpoints_steps=5, save_checkpoints_secs=None, save_summary_steps=1))
  
  oldinit = tf.Session.__init__
  def myinit(session_object, target='', graph=None, config=None):
    print("Intercepted!")
    optimizer_options = tf.OptimizerOptions()
    config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
    config.gpu_options.allow_growth = True 
    return oldinit(session_object, target, graph, config)
  tf.Session.__init__ = myinit


  predict_input_fn = tf.estimator.inputs.numpy_input_fn( x={'images': test_node.data[:,:,:,0],'images_obj': test_node.data_obj[:,:,:,0], 'labels': test_node.labels[:,:,:,0]}, num_epochs=1, shuffle=False)

  
  predictions = model.predict(input_fn=predict_input_fn)
  result1 = array(list(predictions))

  for i in range(test_node.data.shape[0]):
    scipy.misc.imsave('result/F_'+str(i)+'.map.jpg', test_node.data[i,:,:,0])
    scipy.misc.imsave('result/F_'+str(i)+'.label.jpg', test_node.labels[i,:,:,0])
    scipy.misc.imsave('result/F_'+str(i)+'.rrt_learned.jpg', result1[i,:,:])  
    scipy.misc.imsave('result/F_'+str(i)+'.rrt_and_map.jpg', (result1[i,:,:]*0.3+test_node.data[i,:,:,0]+0.0*test_node.labels[i,:,:,0])*255.0 ) 
    scipy.misc.imsave('rrt_out/file_'+str(i)+".jpg", result1[i,:,:])
    np.savetxt('rrt_out/file_'+str(i)+".csv", result1[i,:,:], delimiter=",")
    scipy.misc.imsave('labels/file_'+str(i)+'.jpg', data_node_big.test_labels[i,:,:,0])    
    np.savetxt("labels/file_"+str(i)+".csv", data_node_big.test_labels[i,:,:,0], delimiter=",")    
    
