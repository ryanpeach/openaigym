import numpy as np
import tensorflow as tf

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

def ff_layer(Xin, Nout, stddev = 0.01, const = 0.01, name = "FF"):
    with tf.name_scope(name):
        with tf.name_scope('weights'):
            ff_w = tf.Variable(tf.truncated_normal([Xin.get_shape().as_list()[1], Nout], stddev=stddev))
            variable_summaries(ff_w, name + "/weights")
        with tf.name_scope('biases'):
            ff_b = tf.Variable(tf.constant(const, shape=[Nout]))
            variable_summaries(ff_b, name + "/biases")
        with tf.name_scope('Wx_plus_b'):
            ff_act = tf.matmul(Xin, ff_w) + ff_b
            tf.histogram_summary(name + "/activations", ff_act)
        ff = tf.nn.relu(ff_act, 'relu')
        tf.histogram_summary(name + '/activations_relu', ff)
    return ff

def ff_layers(Xin, N = [512], stddev = 0.01, constants = 0.01, prefix='FF'):
    """ If stddev or constant are floats, that value will be used for each layer. """
    # Initialize Varialbes
    if isinstance(stddev, [float, int, long]): stddev = [float(stddev) for i in range(len(N))]
    if isinstance(constants, [float, int, long]): constants = [float(constants) for i in range(len(N))]
    
    # First layer connected to Xin
    prev, name = [], '{}_0'.format(prefix)
    ff_last = ff_layer(Xin, N[0], stddev=stddev[0], const=constants[0], name=name)
    
    # Subsequent Layers
    if len(N) > 1:
        for i, n in enumerate(N[1:]):
            i, name = i + 1, '{}_{}'.format(prefix,i)
            prev.append(ff_last)
            ff_last = ff_layer(ff_last, N[i], stddev=stddev[i], const=constants[i], name=name)
            
            
    # Return Final Layer
    return ff_last, prev