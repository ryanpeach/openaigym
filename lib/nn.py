import numpy as np
import tensorflow as tf
from collections import deque
import os, pdb, random
from copy import deepcopy

class RollingMemory(object):
    """ A class used for memory which pops its oldest element after
        reaching a certain size """
    def __init__(self, size):
        self.Mem = deque(maxlen = int(size))
        self.SIZE = int(size)
        self.N = 0
        
    def __getitem__(self, i):
        return self.Mem[i]
    
    def __setitem__(self, i, v):
        self.Mem[i] = v
        
    def add(self, i):
        if len(self.Mem) == self.SIZE:
            out = self.Mem.popleft()
        else:
            out = None
        self.Mem.append(i)
        self.N += 1
        return out
        
    def copy(self):
        return deepcopy(self.Mem)
    
    def full(self):
        return len(self.Mem) == self.SIZE
        
    def __iter__(self):
        return iter(self.Mem)
    
    def __len__(self):
        return len(self.Mem)
        
def hotone(index, L):
    out = np.zeros(L)
    out[index] = 1
    return out
    
def softmax(x, axis = 0):
    """Compute softmax values for each sets of scores in x.
       REF: http://stackoverflow.com/questions/34968722/softmax-function-python """
    return np.exp(x) / np.sum(np.exp(x), axis=axis)

def variable_summaries(var, name):
    """SOURCE: https://www.tensorflow.org/versions/r0.8/how_tos/summaries_and_tensorboard/index.html"""
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def in_layer(Nx = 84, Ny = 84, Ch = 4):
    # Input and output classes
    x = tf.placeholder("float", shape=[None, Nx, Ny, Ch])
    
    input_layer = tf.placeholder("float", [None, Nx, Ny, Ch])
    
    return input_layer

def visualNetwork2D(x, Nx = 84, Ny = 84, Ch = 4, Na = 12, 
                        c1s = 8, c1f = 16, c2s = 4, c2f = 32, m1s = 4, m2s = 2):
    """ Returns a shape of 1600, given [None, 80, 80, 1] """
    assert(Nx/m1s == int(Nx/m1s) and (Nx/m1s/m2s == int(Nx/m1s/m2s)))
    assert(Ny/m1s == int(Ny/m1s) and (Ny/m1s/m2s == int(Ny/m1s/m2s)))
    
    # Layer shape is [None, 80, 80, 1]
    with tf.name_scope("Convolution1"):
        with tf.name_scope("weights"):
            conv_w_1 = tf.Variable(tf.truncated_normal([c1s,c1s,Ch,c1f], stddev=0.01))
            variable_summaries(conv_w_1, "Convolution1" + '/weights')
        with tf.name_scope("biases"):
            conv_b_1 = tf.Variable(tf.truncated_normal([c1f], stddev=0.01))
            variable_summaries(conv_b_1, "Convolution1" + '/biases')
        with tf.name_scope('Wx_plus_b'):
            conv1_act = tf.nn.conv2d(x, conv_w_1, strides=[1, 1, 1, 1], padding='SAME') + conv_b_1
            tf.histogram_summary("Convolution1" + '/activations', conv1_act)
        conv1 = tf.nn.relu(conv1_act, 'relu')
        tf.histogram_summary("Convolution1" + '/activations_relu', conv1)
        
    with tf.name_scope("Max1"):
        max1 = tf.nn.max_pool(conv1, ksize=[1, m1s, m1s, 1], strides=[1, m1s, m1s, 1], padding='SAME')
        tf.histogram_summary("Max1", max1)
        
    # Layer shape is [None, 20, 20, 32]
    with tf.name_scope("Convolution2"):
        with tf.name_scope("weights"):
            conv_w_2 = tf.Variable(tf.truncated_normal([c2s,c2s,c1f,c2f], stddev=0.01))
            variable_summaries(conv_w_2, "Convolution2" + '/weights')
        with tf.name_scope("biases"):
            conv_b_2 = tf.Variable(tf.truncated_normal([c2f], stddev=0.01))
            variable_summaries(conv_b_2, "Convolution2" + '/biases')
        with tf.name_scope('Wx_plus_b'):
            conv2_act = tf.nn.conv2d(max1, conv_w_2, strides=[1, 1, 1, 1], padding='SAME') + conv_b_2
            tf.histogram_summary("Convolution2" + '/activations', conv2_act)
        conv2 = tf.nn.relu(conv2_act, 'relu')
        tf.histogram_summary("Convolution2" + '/activations_relu', conv2)

    with tf.name_scope("Max2"):
        max2 = tf.nn.max_pool(conv2, ksize=[1, m2s, m2s, 1], strides=[1, m2s, m2s, 1], padding='SAME')
        tf.histogram_summary("Max2", max2)
        
    return tf.reshape(max2, [-1, int((Nx/m1s/m2s)*(Ny/m1s/m2s)*c2f)]) # Layer shape [None, 5, 5, 64] 1600 Total
    
def ffNetwork(x, No = 512, Na = 12):
    """ Copied from http://www.danielslater.net/2016/03/deep-q-learning-pong-with-tensorflow.html, making modifications"""
    
    with tf.name_scope('FF1'):
        with tf.name_scope('weights'):
            ff_w_1 = tf.Variable(tf.truncated_normal([x.get_shape().as_list()[1], No], stddev=0.01))
            variable_summaries(ff_w_1, "FF1" + '/weights')
        with tf.name_scope('biases'):
            ff_b_1 = tf.Variable(tf.constant(0.01, shape=[No]))
            variable_summaries(ff_b_1, "FF1" + '/biases')
        with tf.name_scope('Wx_plus_b'):
            ff1_act = tf.matmul(x, ff_w_1) + ff_b_1
            tf.histogram_summary("FF1" + '/activations', ff1_act)
        ff1 = tf.nn.relu(ff1_act, 'relu')
        tf.histogram_summary("FF1" + '/activations_relu', ff1)
        
    with tf.name_scope('FF2'):
        with tf.name_scope('weights'):
            ff_w_2 = tf.Variable(tf.truncated_normal([No, Na], stddev=0.01))
            variable_summaries(ff_w_2, "FF2" + '/weights')
        with tf.name_scope('biases'):
            ff_b_2 = tf.Variable(tf.constant(0.01, shape=[Na]))
            variable_summaries(ff_b_2, "FF2" + '/biases')
        out = tf.matmul(ff1, ff_w_2) + ff_b_2
        tf.histogram_summary("FF2" + '/activations_relu', out)
    
    return out

def tensor_shape(x):
    """ Returns the shape of a tensor as a normal list. """
    return [d._value for d in x.get_shape()._dims]
    
def ffLayer(in_layer, size):
    """ Returns one feed forward layer connected to in_layer of the given size.
        Also returns a tuple containing the weight and bias variables. """
    size_in = tensor_shape(in_layer)[1]
    W = tf.Variable(tf.random_uniform([size_in, size], -1.0, 1.0))
    B = tf.Variable(tf.zeros([size]))
    Y = tf.matmul(in_layer, W) + B
    return Y, (W, B)
    
def ffBranch(input_layer, sizes):
    """ Returns a series of feed forward layers branching originally from input_layer.
        Outputs the last layer and a list [(Y0,W0,B0), (Y1,W1,B1), ...] each at the given size in sizes. """
    prev_layer = input_layer
    Y, W, B = [], [], []
    for size in sizes:
        y_temp, wb_temp = ffLayer(prev_layer, size)
        Y.append(y_temp)
        W.append(wb_temp[0])
        B.append(wb_temp[1])
    return Y[-1], zip(Y, W, B)
    