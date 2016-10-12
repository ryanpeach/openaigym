# Map state to button presses

import numpy as np
import tensorflow as tf
from PLC import *
from nntools import *

def combine(data1, dtype1, data2, dtype2, prefix1, prefix2):
    # Initialize Variables
    d1_in_size, d1_out_size = len(data1.monitor), len(data1.monitor)      # The size of bit in and out
    d2_in_size, d2_out_size = len(data2.monitor), len(data2.monitor)  # The size of word in and out
    d1_N, d1_stddev, d1_constants = [], 0.01, 0.01                             # The shape, stddev, and const vals for the bit ff layers
    d2_N, d2_stddev, d2_constants = [], 0.01, 0.01                          # The shape, stddev, and const vals for the word ff layers
    
    # Initialize bits input
    in_layer1 = tf.placeholder(dtype1, shape=[None, d1_in_size, 1, 1], name=prefix1+"_in_layer1")   # The placeholder for the input of bit
    in_layer1_cast = tf.cast(in_layer1, tf.float32, name=prefix1+"_in_layer1_cast")
    ff1, _ = ff_layers(in_layer1, N=d1_N, stddev=d1_stddev, constants=d1_constants, prefix=prefix1+"_FF")                                      # The first input layers of bit, this might be thought of as "significance selection"
    
    # Initialize words input
    in_layer2 = tf.placeholder(dtype2, shape=[None, d2_in_size, 1, 1], name=prefix2+"_in_layer2")        # The placeholder for the input of bit
    in_layer_cast2 = tf.cast(word_in_layer, tf.float32, name=prefix2+"_in_layer_cast")
    ff2, _ = ff_layers(in_layer2, N=d2_N, stddev=d2_stddev, constants=d2_constants, prefix=prefix2+"_FF")                                   # The first input layers of bit, this might be thought of as "significance selection"
    
    # Combine bit_ff and word_ff
    comb_in_layer = tf.concat(0, [ff1, ff2], name = "{}_{}_concat".format(prefix1,prefix2))
    return comb_in_layer
    
def main(bit_data, word_data, No, N = 3):
    # The output of this network is going to be one-hot with No buttons
    # High confidence will determine whether or not the button should be pressed.
    
    # Initialize Variables
    comb_N, comb_stddev, comb_constants = [], 0.01, 0.01             # The shape, stddev, and const vals for the word ff layers
    branch_N, branch_stddev, branch_constants = [No], 0.01, 0.01
    
    # Combine the data types
    comb_in_layer = combine(bit_data, tf.bool, word_data, tf.uint16, 'bit', 'word')
    
    # Create the root
    master, roots = ff_layers(comb_in_layer, N=comb_N, stddev=comb_stddev, constants=comb_constants, prefix="root_FF") # The first input layers of bit, this might be thought of as "significance selection"

    # Create branches off of the closest root to master    
    branches = []
    for n in range(N):
        branch, _ = ff_layers(roots[-1], N=branch_N[n], stddev=branch_stddev[n], constants=branch_constants[n], prefix="branch{}_FF".format(n))
        branches.append(branch)
    
    return master, branches