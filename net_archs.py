#!/usr/bin/env python

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
from keras.applications.vgg19 import VGG19


def normalize(x):
	max_x = tf.reshape(tf.math.reduce_max(x,axis=[1,2]),[-1,1,1,1])
	min_x = tf.reshape(tf.math.reduce_min(x,axis=[1,2]),[-1,1,1,1])
	return ( x - min_x ) / ( max_x - min_x ) + 1e-10




def conv_net_name():
	return "conv_net_train"
	
def conv_net(x_dict, y_dict, dropout, reuse, is_training):
  
  with tf.variable_scope('ConvNet', reuse=reuse):

      img = tf.reshape(x_dict['images'], shape=[-1, 200, 200, 1])
      img_obj = tf.reshape(x_dict['images_obj'], shape=[-1, 200, 200, 1])
            
      x = tf.layers.conv2d(img_obj, 16, 3, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
      
      x = tf.layers.conv2d(x, 16, 5, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

      x = tf.layers.conv2d(x, 16, 7, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.layers.conv2d(x, 16, 7, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)     
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')       
      
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training) 
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training) 
            
      x = tf.image.resize_nearest_neighbor(x, (200,200))
      x = tf.concat([x,img_obj],3)
      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)      
      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')  
      x = tf.layers.dropout(x, rate=dropout, training=is_training)  
      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)      
      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')  
      x = tf.layers.dropout(x, rate=dropout, training=is_training)
      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)      
      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')  
      x = tf.layers.dropout(x, rate=dropout, training=is_training)                 
      x = tf.layers.conv2d(x, 1, 1, activation=tf.nn.relu, padding='same')
      
      x = normalize(x)
      #~ x = tf.minimum( x + 1e-10, 1e+7)
      #~ x = tf.nn.l2_normalize(x,[1,2]) + 1e-10
      cost_label = tf.reshape(x, [-1,200, 200])
  return cost_label

def tunel_net_name():
	return "tunel_net_estimator"
	
def tunel_net(x_dict, y_dict, dropout, reuse, is_training):
  
  with tf.variable_scope('ConvNet', reuse=reuse):

      img = tf.reshape(x_dict['images'], shape=[-1, 200, 200, 1])
            
      x1 = tf.layers.conv2d(img, 16, 9, strides=(3, 3), activation=tf.nn.relu, padding='SAME')
      x1 = tf.layers.dropout(x1, rate=dropout, training=is_training)   
      
      x2 = tf.layers.conv2d(x1, 16, 7, strides=(3, 3), activation=tf.nn.relu, padding='SAME')
      x2 = tf.layers.dropout(x2, rate=dropout, training=is_training)   
   
      x3 = tf.layers.conv2d(x2, 16, 5, strides=(3, 3), activation=tf.nn.relu, padding='SAME')
      x3 = tf.layers.dropout(x3, rate=dropout, training=is_training)   
      
      x4 = tf.layers.conv2d(x3, 16, 3, strides=(3, 3), activation=tf.nn.relu, padding='SAME')
      x4 = tf.layers.dropout(x4, rate=dropout, training=is_training)   
                 
      x5 = tf.layers.conv2d(x4, 1, 1, strides=(3, 3), activation=tf.nn.relu, padding='SAME')
      x_media = tf.layers.dropout(x5, rate=dropout, training=is_training)   
      
      #~ x6 =  tf.layers.conv2d_transpose(x_media, 16, 3, strides=(3, 3), activation=tf.nn.relu, padding='SAME')
      #~ x6 = tf.layers.dropout(x6, rate=dropout, training=is_training)  

      #~ x6 = tf.concat([x6,x4],3) 
       
      #~ x7 =  tf.layers.conv2d_transpose(x6, 16, 5, strides=(3, 3), activation=tf.nn.relu, padding='SAME')
      #~ x7 = tf.layers.dropout(x7, rate=dropout, training=is_training)  
          
      #~ x8 = tf.concat([x7,x3],3) 
            
      #~ x8 =  tf.layers.conv2d_transpose(x8, 16, 7, strides=(3, 3), activation=tf.nn.relu, padding='SAME')
      #~ x8 = tf.layers.dropout(x8, rate=dropout, training=is_training)  
      
      #~ x9 = tf.concat([x8,x2],3) 
      
      #~ x9 =  tf.layers.conv2d_transpose(x9, 16, 9, strides=(3, 3), activation=tf.nn.relu, padding='SAME')
      #~ x9 = tf.layers.dropout(x9, rate=dropout, training=is_training)  
          
      #~ x10 = tf.concat([x9,x2],3) 
          
      
      x = tf.layers.conv2d(x_media, 16, 9, activation=tf.nn.relu, padding='SAME')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='SAME')
      x = tf.layers.dropout(x, rate=dropout, training=is_training) 
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='SAME')
      x = tf.layers.dropout(x, rate=dropout, training=is_training) 
            
            
      x = tf.image.resize_nearest_neighbor(x, (200,200))
      x = tf.concat([x,img],3)

      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='SAME')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)      
      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='SAME')  
      x = tf.layers.dropout(x, rate=dropout, training=is_training)                 
      x = tf.layers.conv2d(x, 1, 1, activation=tf.nn.relu, padding='SAME')
      
      x = normalize(x)
      #~ x = tf.minimum( x + 1e-10, 1e+7)
      #~ x = tf.nn.l2_normalize(x,[1,2]) + 1e-10
      cost_label = tf.reshape(x, [-1,200, 200])
  return cost_label



def test_1_name():
	return "test_1_train"
	
def test_1_net(x_dict, y_dict, dropout, reuse, is_training):
  
  with tf.variable_scope('Test1', reuse=reuse):

      img = tf.reshape(x_dict['images'], shape=[-1, 200, 200, 1])
      img_obj = tf.reshape(x_dict['images_obj'], shape=[-1, 200, 200, 1])
            
      x = tf.layers.conv2d(img_obj, 16, 3, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
      
      x = tf.layers.conv2d(x, 16, 5, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

      x = tf.layers.conv2d(x, 16, 7, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.layers.conv2d(x, 16, 7, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)     
      #~ x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')       
     
      x = tf.image.resize_nearest_neighbor(x, (200,200))
     
      x = tf.layers.dropout(x, rate=dropout, training=is_training)                 
      x = tf.layers.conv2d(x, 1, 1, activation=tf.nn.relu, padding='same')      
      x = normalize(x)
      #~ x = tf.minimum( x + 1e-10, 1e+7)
      #~ x = tf.nn.l2_normalize(x,[1,2]) + 1e-10
      cost_label = tf.reshape(x, [-1,200, 200])
  return cost_label

  
def VGG19_name():
	return "VGG19_train"
	
def VGG19_net(x_dict, y_dict, dropout, reuse, is_training):
  
  with tf.variable_scope('VGG19', reuse=reuse):

      img = tf.reshape(x_dict['images'], shape=[-1, 200, 200, 1])
      img_obj = tf.reshape(x_dict['images_obj'], shape=[-1, 200, 200, 1])

      x = img_obj

      vgg_model = VGG19(weights = None, include_top=False, input_shape = (200, 200, 1))
      #~ x = tf.image.resize_nearest_neighbor(x, (224,224))
      #~ x = tf.concat([x,x,x],3)
            
      x = vgg_model(x)       
      
      #~ x = vgg_model.get_layer('block5_pool').output

      #~ x = Flatten()(x)
      #~ x = Dense(200*200, activation='relu')(x)
 
      x = tf.image.resize_nearest_neighbor(x, (200,200))

      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='SAME')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)      
      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='SAME')  
      x = tf.layers.dropout(x, rate=dropout, training=is_training)                 
      x = tf.layers.conv2d(x, 1, 1, activation=tf.nn.relu, padding='SAME')      
      
      x = normalize(x)
      cost_label = tf.reshape(x, [-1,200, 200])
  return cost_label



def big_conv_net_name():
	return "big_conv_net_train"
	
def big_conv_net(x_dict, y_dict, dropout, reuse, is_training):
  
  with tf.variable_scope('ConvNet', reuse=reuse):

      img = tf.reshape(x_dict['images'], shape=[-1, 200, 200, 1])
      img_obj = tf.reshape(x_dict['images_obj'], shape=[-1, 200, 200, 1])
            
      x = tf.layers.conv2d(img_obj, 16, 3, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
      
      x = tf.layers.conv2d(x, 16, 5, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

      x = tf.layers.conv2d(x, 16, 7, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.layers.conv2d(x, 16, 7, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)     
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')       
      
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training) 
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training) 
            
      x = tf.image.resize_nearest_neighbor(x, (200,200))
      x = tf.concat([x,img_obj],3)

      x = tf.layers.conv2d(img_obj, 16, 3, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
      
      x = tf.layers.conv2d(x, 16, 5, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

      x = tf.layers.conv2d(x, 16, 7, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.layers.conv2d(x, 16, 7, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)     
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')       
      
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training) 
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training) 
            
      x = tf.image.resize_nearest_neighbor(x, (200,200))
      x = tf.concat([x,img_obj],3)
      
      x = tf.layers.conv2d(img_obj, 16, 3, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
      
      x = tf.layers.conv2d(x, 16, 5, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

      x = tf.layers.conv2d(x, 16, 7, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.layers.conv2d(x, 16, 7, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)     
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')       
      
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training) 
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training) 
            
      x = tf.image.resize_nearest_neighbor(x, (200,200))
      x = tf.concat([x,img_obj],3)
      
      x = tf.layers.conv2d(img_obj, 16, 3, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
      
      x = tf.layers.conv2d(x, 16, 5, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

      x = tf.layers.conv2d(x, 16, 7, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.layers.conv2d(x, 16, 7, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)     
      x = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')       
      
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)   
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training) 
      x = tf.layers.conv2d(x, 16, 9, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training) 
            
      x = tf.image.resize_nearest_neighbor(x, (200,200))
      x = tf.concat([x,img_obj],3)            
      
      
      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)      
      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')  
      x = tf.layers.dropout(x, rate=dropout, training=is_training)  
      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)      
      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')  
      x = tf.layers.dropout(x, rate=dropout, training=is_training)
      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')
      x = tf.layers.dropout(x, rate=dropout, training=is_training)      
      x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu, padding='same')  
      x = tf.layers.dropout(x, rate=dropout, training=is_training)                 
      x = tf.layers.conv2d(x, 1, 1, activation=tf.nn.relu, padding='same')
      
      x = normalize(x)
      #~ x = tf.minimum( x + 1e-10, 1e+7)
      #~ x = tf.nn.l2_normalize(x,[1,2]) + 1e-10
      cost_label = tf.reshape(x, [-1,200, 200])
  return cost_label














