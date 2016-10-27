import numpy as np
import tensorflow as tf
from lib.nn import *

def q_ff_network(input_layer, state_length, action_length, learn_rate = .5):
    ## FF Layer 1
    W0 = tf.Variable(tf.random_uniform([state_length, 512], -1.0, 1.0))
    b0 = tf.Variable(tf.zeros([512]))
    y0 = W0 * input_layer + b0
    
    ## FF Layer 2
    W0 = tf.Variable(tf.random_uniform([512, action_length], -1.0, 1.0))
    b0 = tf.Variable(tf.zeros([action_length]))
    y1 = W1 * y0 + b1
    
    return y1