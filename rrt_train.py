#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy import array
import os
from os import listdir
from os.path import isfile, join
import sys, shutil, random, math, pygame, csv, time
from math import sqrt,cos,sin,atan2
from keras.preprocessing.image import load_img, img_to_array, array_to_img #, flip_axis
import scipy.misc

import tensorflow as tf
#from tensorflow.contrib.image import rotate 
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.estimator.export import export
from tensorflow.python.saved_model import builder as saved_model_builder
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

import database_management as db_management
import net_archs

learning_rate = 0.000002
num_steps = 1000000
batch_size = 20
N_REP_NET = 2
sum_eps = 0.00
dropout = 0.2

train = 1
evaluate = 0
predict = 0

normalize_img_data = True
use_six_categories = True
create_npy_files = True
load_npy_files = False
#~ create_npy_files = False
#~ load_npy_files = True

dataset_dir = './data/Learning/'
testset_dir = './data/Test/'
save_dir1 = './data/Learning/'
save_dir2 = './data/Test/'  


tf.logging.set_verbosity(tf.logging.INFO)

sess = tf.Session()

rrt_star_module = tf.load_op_library('./bin/rrt_star.so')
metric_path_module = tf.load_op_library('./bin/metric_path.so')

@ops.RegisterGradient("RRTStar")
def _rrt_star_grad(op, grad):
  label = op.inputs[2]
  result = op.outputs[0]
  grad_in = label - result +sum_eps
  return grad, grad*grad_in, grad, 0

@ops.RegisterGradient("MetricPath")
def _metric_path_grad(op, grad):
  return op.inputs[0]*grad,op.inputs[1]*grad,op.inputs[2]*grad


def training(features, labels, mode): 
      
  dist_min = 9999999.0
  dist_max = 0.0  
  dist = []
    
  
  cost_train = net_archs.conv_net(features, labels, dropout, reuse=False, is_training=True)
  cost_test = net_archs.conv_net(features, labels, dropout, reuse=True, is_training=False)

  logits_test = rrt_star_module.rrt_star(features['images'],cost_train,features['labels'], 0) 
  
  dist_paths_rrt, dissimilarity =  metric_path_module.metric_path(logits_test,features['labels'], features['images'],0)
  mse_rrt = tf.losses.mean_squared_error(logits_test,features['labels'])
  dist.append(dist_paths_rrt)   
     
  dist_min = tf.cond(dist_paths_rrt<dist_min,lambda: dist_paths_rrt, lambda: dist_min )
  dist_max = tf.cond(dist_paths_rrt>dist_max,lambda: dist_paths_rrt, lambda: dist_max )
       
  for i in range(N_REP_NET-1):
    logits_test_i = rrt_star_module.rrt_star(features['images'],cost_train,features['labels'], i+1)   
    dist_paths_rrt_aux, dissimilarity_aux = metric_path_module.metric_path(logits_test_i,features['labels'],features['images'],0)
    mse_rrt += tf.losses.mean_squared_error(logits_test_i,features['labels'])
    logits_test += logits_test_i     
    dissimilarity += dissimilarity_aux
    dist.append(dist_paths_rrt_aux)
    dist_paths_rrt += dist_paths_rrt_aux
    dist_min = tf.cond(dist_paths_rrt_aux<dist_min,lambda: dist_paths_rrt_aux, lambda: dist_min )
    dist_max = tf.cond(dist_paths_rrt_aux>dist_max,lambda: dist_paths_rrt_aux, lambda: dist_max )

  logits_test /= N_REP_NET
  dist_paths_rrt /= N_REP_NET
  dissimilarity /= N_REP_NET
  mse_rrt /= N_REP_NET
  
  tf_dist = tf.stack(dist)
  
  size_dist = float(N_REP_NET)

  stddev = tf.sqrt(tf.reduce_sum(tf.pow(tf_dist - dist_paths_rrt, 2))/size_dist)
  stderror = stddev * tf.rsqrt(size_dist);

  mse_cost = tf.losses.mean_squared_error(cost_test,1.0-features['labels'])
  log_likelihood = tf.reduce_sum(tf.multiply(cost_test,features['labels']-logits_test+sum_eps))

  loss_op = log_likelihood
  log_image = tf.log(tf.clip_by_value(tf.reshape(cost_test, shape=[-1, 200, 200, 1]),0,0.1))


  tf.summary.image('images', tf.reshape(features['images'], shape=[-1, 200, 200, 1]))  
  tf.summary.image('rrt_in', tf.reshape(cost_test, shape=[-1, 200, 200, 1]))
  tf.summary.image('rrt_out', tf.reshape(logits_test, shape=[-1, 200, 200, 1]))   
  tf.summary.image('label', tf.reshape(features['labels'], shape=[-1, 200, 200, 1])) 

  #~ tf.summary.scalar('dist_between_paths', dist_paths_rrt)
  #~ tf.summary.scalar('dist_min', dist_min)
  #~ tf.summary.scalar('dist_max', dist_max)
  #~ tf.summary.scalar('dissimilarity', dissimilarity)
  tf.summary.scalar('mse_rrt', mse_rrt)
  tf.summary.scalar('mse_cost', mse_cost)
  tf.summary.scalar('log_likelihood', log_likelihood)
  #~ tf.summary.scalar('stderror', stderror)

  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(mse_rrt, global_step=tf.train.get_global_step()) 



  if mode == tf.estimator.ModeKeys.PREDICT:
    #estim_specs = tf.estimator.EstimatorSpec(mode, predictions=cost_test)
    estim_specs = tf.estimator.EstimatorSpec(mode, predictions=logits_test)
  elif mode == tf.estimator.ModeKeys.EVAL:
    metrics = {'mse_rrt': tf.metrics.mean(mse_rrt), 'mse_cost': tf.metrics.mean(mse_cost), 'log_likelihood': tf.metrics.mean(log_likelihood)}
    estim_specs = tf.estimator.EstimatorSpec( mode, loss=loss_op, eval_metric_ops=metrics)
  else:
    estim_specs = tf.estimator.EstimatorSpec(
	  mode=mode,
	  predictions=logits_test,
	  loss=mse_rrt,
	  train_op=train_op)
  return estim_specs





if __name__ == '__main__':

  path_save_net = net_archs.conv_net_name()

  
  data = db_management.TestCNN(dataset_dir, save_dir1)
  data.load_data(load_npy=load_npy_files, save_npy=create_npy_files, norm=normalize_img_data, six=use_six_categories)

  data_node_data, eval_node_data, data_node_labels, eval_node_labels = train_test_split(data.data, data.labels, test_size=0.05, random_state=0)  
 
  test_node = db_management.TestCNN(testset_dir, save_dir2)
  test_node.load_data(load_npy=load_npy_files, save_npy=create_npy_files, norm=normalize_img_data, six=use_six_categories) 
          
  # Build the Estimator
  model = tf.estimator.Estimator(training, model_dir=path_save_net,config=tf.contrib.learn.RunConfig(save_checkpoints_steps=100, save_checkpoints_secs=None, save_summary_steps=5))

  oldinit = tf.Session.__init__
  def myinit(session_object, target='', graph=None, config=None):
    print("Intercepted!")
    optimizer_options = tf.OptimizerOptions()
    config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=optimizer_options))
    config.gpu_options.allow_growth = True 
    return oldinit(session_object, target, graph, config)
  tf.Session.__init__ = myinit

  
  # Define the input function for training
  input_fn = tf.estimator.inputs.numpy_input_fn( x={'images': data_node_data[:,:,:,0],'images_obj': db_management.remove_objective(data_node_data[:,:,:,0]), 'labels': data_node_labels[:,:,:,0]}, y = data_node_labels[:,:,:,0], batch_size=batch_size, num_epochs=None, shuffle=True)
  eval_fn = tf.estimator.inputs.numpy_input_fn( x={'images': eval_node_data[:,:,:,0],'images_obj': db_management.remove_objective(eval_node_data[:,:,:,0]), 'labels': eval_node_labels[:,:,:,0]}, y = eval_node_labels[:,:,:,0], batch_size=batch_size, num_epochs=None, shuffle=True)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn( x={'images': test_node.data[:,:,:,0],'images_obj': db_management.remove_objective(test_node.data[:,:,:,0]), 'labels': test_node.labels[:,:,:,0]}, num_epochs=1, shuffle=False)

  train_spec = tf.estimator.TrainSpec(input_fn = input_fn, max_steps = num_steps)
  eval_spec = tf.estimator.EvalSpec(  input_fn = eval_fn,  throttle_secs=60*20, start_delay_secs=60*20)

  # Train the Model
  if train == 1:
    tf.estimator.train_and_evaluate( model, train_spec, eval_spec)
  
  if evaluate == 1:
    metrics = model.evaluate(input_fn=eval_fn)
    w = csv.writer(open(path_save_net+".csv","w"))
    for key, val in metrics.items():
      w.writerow([key,val])  

  if predict == 1:    
    predictions = model.predict(input_fn=predict_input_fn)
    result1 = array(list(predictions))

    for i in range(test_node.data.shape[0]):
      scipy.misc.imsave('result/F_'+str(i)+'.map.jpg', test_node.data[i,:,:,0])
      scipy.misc.imsave('result/F_'+str(i)+'.label.jpg', test_node.labels[i,:,:,0])
      scipy.misc.imsave('result/F_'+str(i)+'.label_and_map.jpg', test_node.data[i,:,:,0]+0.3*test_node.labels[i,:,:,0])      
      scipy.misc.imsave('result/F_'+str(i)+'.cost_and_map.jpg', result1[i,:,:]*255.0/3+test_node.data[i,:,:,0]+0.0*test_node.labels[i,:,:,0])  
      scipy.misc.imsave('result/F_'+str(i)+'.cost_learned.jpg', result1[i,:,:])  
      #np.savetxt("csv_files/file_"+str(i)+".csv", result1[i,:,:], delimiter=",")
      
      scipy.misc.imsave('result/F_'+str(i)+'.rrt_learned.jpg', result1[i,:,:])  
      scipy.misc.imsave('result/F_'+str(i)+'.rrt_and_map.jpg', (result1[i,:,:]*0.3+test_node.data[i,:,:,0]+0.0*test_node.labels[i,:,:,0])*255.0 ) 
