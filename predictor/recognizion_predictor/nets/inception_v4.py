#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:21:55 2019

@author: klwang
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# from nets import inception_utils

slim = tf.contrib.slim

class Inception_v4():
  def __init__(self,inputs, num_classes=3755, is_training=True, reuse=False,
               dropout_keep_prob=0.8, spatial_squeeze=True, scope='InceptionV4',
               weight_decay = 0.0004, use_batch_norm=True, batch_norm_decay=0.9997,
               batch_norm_epsilon=0.001, create_aux_logits=True):
      self.inputs = inputs
      self.num_classes = num_classes
      self.is_training = is_training
      self.reuse = reuse
      self.dropout_keep_prob = dropout_keep_prob
      self.spatial_squeeze = spatial_squeeze
      self.scope = scope 
      self.weight_decay=weight_decay
      self.batch_norm_decay = batch_norm_decay
      self.batch_norm_epsilon = batch_norm_epsilon
      self.use_batch_norm = use_batch_norm
      self.create_aux_logits= create_aux_logits
      batch_norm_params = {'decay': self.batch_norm_decay,
                           'epsilon': self.batch_norm_epsilon,
                           'updates_collections': tf.GraphKeys.UPDATE_OPS}
      if self.use_batch_norm:
        self.normalizer_fn = slim.batch_norm
        self.normalizer_params = batch_norm_params
      else:
        self.normalizer_fn = None
        self.normalizer_params = {}
      self.create()
      
  def create(self):
    def block_inception_a(inputs, scope=None, reuse=None):
      with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                          stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockInceptionA', [inputs], reuse=reuse):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs, 96, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
            branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 96, [1, 1], scope='Conv2d_0b_1x1')
          return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
  
    def block_reduction_a(inputs, scope=None, reuse=None):
      with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                          stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockReductionA', [inputs], reuse=reuse):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs, 384, [3, 3], stride=2, padding='VALID',
                                   scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
            branch_1 = slim.conv2d(branch_1, 256, [3, 3], stride=2,
                                   padding='VALID', scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID',
                                       scope='MaxPool_1a_3x3')
          return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
    
    def block_inception_b(inputs, scope=None, reuse=None):
      with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                          stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockInceptionB', [inputs], reuse=reuse):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 224, [1, 7], scope='Conv2d_0b_1x7')
            branch_1 = slim.conv2d(branch_1, 256, [7, 1], scope='Conv2d_0c_7x1')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
            branch_2 = slim.conv2d(branch_2, 224, [1, 7], scope='Conv2d_0c_1x7')
            branch_2 = slim.conv2d(branch_2, 224, [7, 1], scope='Conv2d_0d_7x1')
            branch_2 = slim.conv2d(branch_2, 256, [1, 7], scope='Conv2d_0e_1x7')
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
          return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
    
    def block_reduction_b(inputs, scope=None, reuse=None):
      with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                          stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockReductionB', [inputs], reuse=reuse):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
            branch_0 = slim.conv2d(branch_0, 192, [3, 3], stride=2,
                                   padding='VALID', scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = slim.conv2d(branch_1, 256, [1, 7], scope='Conv2d_0b_1x7')
            branch_1 = slim.conv2d(branch_1, 320, [7, 1], scope='Conv2d_0c_7x1')
            branch_1 = slim.conv2d(branch_1, 320, [3, 3], stride=2,
                                   padding='VALID', scope='Conv2d_1a_3x3')
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID',
                                       scope='MaxPool_1a_3x3')
          return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
    
    
    def block_inception_c(inputs, scope=None, reuse=None):
      with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                          stride=1, padding='SAME'):
        with tf.variable_scope(scope, 'BlockInceptionC', [inputs], reuse=reuse):
          with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
          with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = tf.concat(axis=3, values=[
                slim.conv2d(branch_1, 256, [1, 3], scope='Conv2d_0b_1x3'),
                slim.conv2d(branch_1, 256, [3, 1], scope='Conv2d_0c_3x1')])
          with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 448, [3, 1], scope='Conv2d_0b_3x1')
            branch_2 = slim.conv2d(branch_2, 512, [1, 3], scope='Conv2d_0c_1x3')
            branch_2 = tf.concat(axis=3, values=[
                slim.conv2d(branch_2, 256, [1, 3], scope='Conv2d_0d_1x3'),
                slim.conv2d(branch_2, 256, [3, 1], scope='Conv2d_0e_3x1')])
          with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 256, [1, 1], scope='Conv2d_0b_1x1')
          return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
        
    def inception_v4_base(inputs, final_endpoint='Mixed_7d', scope=None):
      end_points = {}
    
      def add_and_check_final(name, net):
        end_points[name] = net
        return name == final_endpoint
    
      with tf.variable_scope(scope, 'InceptionV4', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
          net = slim.conv2d(inputs, 32, [3, 3], stride=2,
                            padding='VALID', scope='Conv2d_1a_3x3')
          if add_and_check_final('Conv2d_1a_3x3', net): return net, end_points
          net = slim.conv2d(net, 32, [3, 3], padding='VALID',
                            scope='Conv2d_2a_3x3')
          if add_and_check_final('Conv2d_2a_3x3', net): return net, end_points
          net = slim.conv2d(net, 64, [3, 3], scope='Conv2d_2b_3x3')
          if add_and_check_final('Conv2d_2b_3x3', net): return net, end_points
          with tf.variable_scope('Mixed_3a'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                         scope='MaxPool_0a_3x3')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(net, 96, [3, 3], stride=2, padding='VALID',
                                     scope='Conv2d_0a_3x3')
            net = tf.concat(axis=3, values=[branch_0, branch_1])
            if add_and_check_final('Mixed_3a', net): return net, end_points
          with tf.variable_scope('Mixed_4a'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
              branch_0 = slim.conv2d(branch_0, 96, [3, 3], padding='VALID',
                                     scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, 64, [1, 7], scope='Conv2d_0b_1x7')
              branch_1 = slim.conv2d(branch_1, 64, [7, 1], scope='Conv2d_0c_7x1')
              branch_1 = slim.conv2d(branch_1, 96, [3, 3], padding='VALID',
                                     scope='Conv2d_1a_3x3')
            net = tf.concat(axis=3, values=[branch_0, branch_1])
            if add_and_check_final('Mixed_4a', net): return net, end_points
          with tf.variable_scope('Mixed_5a'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(net, 192, [3, 3], stride=2, padding='VALID',
                                     scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                         scope='MaxPool_1a_3x3')
            net = tf.concat(axis=3, values=[branch_0, branch_1])
            if add_and_check_final('Mixed_5a', net): return net, end_points
          for idx in range(4):
            block_scope = 'Mixed_5' + chr(ord('b') + idx)
            net = block_inception_a(net, block_scope)
            if add_and_check_final(block_scope, net): return net, end_points
          net = block_reduction_a(net, 'Mixed_6a')
          if add_and_check_final('Mixed_6a', net): return net, end_points
          for idx in range(7):
            block_scope = 'Mixed_6' + chr(ord('b') + idx)
            net = block_inception_b(net, block_scope)
            if add_and_check_final(block_scope, net): return net, end_points
          net = block_reduction_b(net, 'Mixed_7a')
          if add_and_check_final('Mixed_7a', net): return net, end_points
          for idx in range(3):
            block_scope = 'Mixed_7' + chr(ord('b') + idx)
            net = block_inception_c(net, block_scope)
            if add_and_check_final(block_scope, net): return net, end_points
      raise ValueError('Unknown final endpoint %s' % final_endpoint)
      
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(self.weight_decay)):
      with slim.arg_scope([slim.conv2d],
                          weights_initializer=slim.variance_scaling_initializer(),
                          activation_fn=tf.nn.relu,
                          normalizer_fn=self.normalizer_fn,
                          normalizer_params=self.normalizer_params):
        end_points = {}
        with tf.variable_scope(self.scope, 'InceptionV4', [self.inputs], reuse=self.reuse) \
                              as scope:
          with slim.arg_scope([slim.batch_norm, slim.dropout],is_training=self.is_training):
            net, end_points = inception_v4_base(self.inputs, scope=scope)
            self.base = net
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, 
                                padding='SAME'):
              if self.create_aux_logits:
                with tf.variable_scope('AuxLogits'):
                  aux_logits = end_points['Mixed_6h']
                  aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3,
                                               padding='VALID',
                                               scope='AvgPool_1a_5x5')
                  aux_logits = slim.conv2d(aux_logits, 128, [1, 1],
                                           scope='Conv2d_1b_1x1')
                  aux_logits = slim.conv2d(aux_logits, 768,
                                           aux_logits.get_shape()[1:3],
                                           padding='VALID', scope='Conv2d_2a')
                  aux_logits = slim.flatten(aux_logits)
                  aux_logits = slim.fully_connected(aux_logits, self.num_classes,
                                                    activation_fn=None,
                                                    scope='Aux_logits')
                  end_points['AuxLogits'] = aux_logits
                              
              with tf.variable_scope('Logits'):
                net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                      scope='AvgPool_1a')
                self.logits_net = net
                net = slim.dropout(net, self.dropout_keep_prob, scope='Dropout_1b')
                self.net_flatten = slim.flatten(net, scope='PreLogitsFlatten')
                end_points['PreLogitsFlatten'] = self.net_flatten
                logits = slim.fully_connected(self.net_flatten, self.num_classes, activation_fn=None,
                                              scope='Logits')
                end_points['Logits'] = logits
                end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
                self.end_points = end_points
                self.full_net = logits
#  return logits, end_points
      

    
