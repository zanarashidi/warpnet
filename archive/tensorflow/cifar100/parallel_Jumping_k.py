# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
'''
This is the resnet structure
'''
import numpy as np
from hyper_parameters import *
import time
import tensorflow as tf

BN_EPSILON = 0.001

def activation_summary(x):
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def create_variables(name, shape, initializer=tf.contrib.layers.variance_scaling_initializer(), is_fc_layer=False):
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables


def output_layer(input_layer, num_labels):
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h

def batch_normalization_layer(input_layer, dimension):
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
    return bn_layer

def batch_normalization_layer_grad(input_layer, dimension, input_grad):
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    dev = input_grad - mean
    multiplier = gamma/tf.sqrt(variance+BN_EPSILON) #gamma divided by sigma, [C]
    grad_bn = multiplier*((input_grad - tf.reduce_mean(input_grad,axis=0)) - dev/variance*tf.reduce_mean(dev*input_grad,axis=0))
    return grad_bn

def conv_bn_relu_layer(input_layer, filter_shape, stride):
    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)
    output = tf.nn.relu(bn_layer)
    return output

def bn_conv_layer_grad(filter_shape,input_grad):
    filter = create_variables(name='conv', shape=filter_shape)
    grad_bn = input_grad
    flipped_filter = tf.reverse(filter,axis=[1,2])
    grad_conv = tf.nn.conv2d(grad_bn, flipped_filter, strides=[1, 1, 1, 1], padding='SAME')
    return grad_conv

def bn_conv_layer(input_layer, filter_shape):
    filter = create_variables(name='conv', shape=filter_shape)
    in_channel = input_layer.get_shape().as_list()[-1]
    bn_layer = batch_normalization_layer(input_layer, in_channel)
    conv_layer = tf.nn.conv2d(bn_layer, filter, strides=[1, 1, 1, 1], padding='SAME')
    return conv_layer

def bn_relu_conv_layer_grad(filter_shape,input_grad):
    grad_bn = input_grad
    grad_relu = tf.to_float(grad_bn > 0)
    filter = create_variables(name='conv', shape=filter_shape)
    flipped_filter = tf.reverse(filter,axis=[1,2])
    grad_conv = tf.nn.conv2d(grad_relu, flipped_filter, strides=[1, 1, 1, 1], padding='SAME')
    return grad_conv 

def bn_relu_conv_layer(input_layer, filter_shape):
    in_channel = input_layer.get_shape().as_list()[-1]
    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)
    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, 1, 1, 1], padding='SAME')
    return conv_layer 

def F_block_grad(input_layer, output_channel, input_grad):
    input_channel = input_layer.get_shape().as_list()[-1]
    with tf.variable_scope('conv1_in_block'):
        grad_conv1 = bn_conv_layer_grad([3, 3, input_channel, output_channel], input_grad)
    with tf.variable_scope('conv2_in_block'):
        grad_conv2 = bn_relu_conv_layer_grad([3, 3, output_channel, output_channel], grad_conv1)
    with tf.variable_scope('last_bn_in_block'):
        #grad_res = batch_normalization_layer_grad(input_layer, output_channel, grad_conv2)
        grad_res = grad_conv2
    return grad_res

def F_block(input_layer, output_channel):
    input_channel = input_layer.get_shape().as_list()[-1]
    with tf.variable_scope('conv1_in_block'):
        conv1 = bn_conv_layer(input_layer, [3, 3, input_channel, output_channel])
    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel])
    with tf.variable_scope('last_bn_in_block'):
        bn_layer2 = batch_normalization_layer(conv2, output_channel)
    return bn_layer2

def F_block_stoch_width(input, output_channel, survival_rate):
    survival_roll = tf.random_uniform(shape=[], minval=0.0, maxval=1.0, name='survival')
    survive = tf.less(survival_roll, survival_rate)
    return tf.cond(survive, lambda: F_block(input, output_channel), lambda: tf.zeros(input.get_shape()))

def F_block_grad_stoch_width(input, output_channel, input_grad, survival_rate):
    survival_roll = tf.random_uniform(shape=[], minval=0.0, maxval=1.0, name='survival')
    survive = tf.less(survival_roll, survival_rate)
    return tf.cond(survive, lambda: F_block_grad(input, output_channel, input_grad), lambda: tf.zeros(input.get_shape()))

def residual_block(input,output_channel):
    #Residual unit, x2 = x1 + F1(x1,W)
    F1 = F_block(input, output_channel)
    out = input + F1
    return out

def block_jump(input,output_channel, survival_rate):
    with tf.device('/gpu:0'):
        with tf.variable_scope('F1'):
            F1 = F_block_stoch_width(input, output_channel, survival_rate) 
    with tf.device('/gpu:1'):
        with tf.variable_scope('F2') as F2_scope:
            F2 = F_block_stoch_width(input, output_channel, survival_rate)
    with tf.device('/gpu:2'):
        with tf.variable_scope('F2') as F2_scope:
            F2_scope.reuse_variables()
            grad_F2 = F_block_grad_stoch_width(input, output_channel, input, survival_rate)
    out = input + F1 + F2 + grad_F2
    return out

def block_jump3(input, output_channel, survival_rate):
    with tf.device('/gpu:0'):
        with tf.variable_scope('F1'):
            F1 = F_block_stoch_width(input, output_channel, survival_rate)
    with tf.device('/gpu:1'):
        with tf.variable_scope('F2') as F2_scope:
            F2 = F_block_stoch_width(input, output_channel, survival_rate)
    with tf.device('/gpu:2'):
        with tf.variable_scope('F3') as F3_scope:
            F3 = F_block_stoch_width(input, output_channel, survival_rate) 
    with tf.device('/gpu:3'):
        with tf.variable_scope('F2') as F2_scope:
            F2_scope.reuse_variables()
            grad_F2 = F_block_grad_stoch_width(input, output_channel, input, survival_rate) 
        with tf.variable_scope('F3') as F3_scope:    
            F3_scope.reuse_variables()
            F1_grad_F3 = F_block_grad_stoch_width(input, output_channel, input, survival_rate)
            # F2_grad_F3 = F_block_grad_stoch_width(input, output_channel, input, survival_rate)
    out = input + F1 + F2 + F3 + grad_F2 + F1_grad_F3
    return out

def block_jump4(input, output_channel):
    #4 jumps, keeping only 2 Fs 
    with tf.variable_scope('F1'):
        F1,_ = F_block(input,output_channel, input)
    with tf.variable_scope('F2') as F2_scope:
        F2,_ = F_block(input,output_channel, input)
        F2_scope.reuse_variables()
        _, grad_F2 = F_block(input,output_channel, F1)

    with tf.variable_scope('F3') as F3_scope:
        F3,_ = F_block(input,output_channel, input)
        F3_scope.reuse_variables()
        _, F1_grad_F3 = F_block(input,output_channel, F1)
        _, F2_grad_F3 = F_block(input,output_channel, F2)

    with tf.variable_scope('F4') as F4_scope:
        F4,_ = F_block(input,output_channel, input)
        F4_scope.reuse_variables()
        _, F1_grad_F4 = F_block(input,output_channel, F1)
        _, F2_grad_F4 = F_block(input,output_channel, F2)
        _, F3_grad_F4 = F_block(input,output_channel, F3)
    
    out = F1 + F2 + F3 + F4 +grad_F2 + F1_grad_F3 + F1_grad_F4 + F2_grad_F3 + F2_grad_F4 + F3_grad_F4 
    return out

def num_param():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print ('total number of parameters:')
    print (total_parameters)

    return total_parameters
 
def inference(input_tensor_batch, n, reuse, is_training):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''
    n_warp = 2
    dr0 = 1/((2^n_warp)-1)
    dr1 = 1/(2^((2^n_warp)-1))
    dr2 = 1/2
    dr3 = 0

    if(is_training):
        sr = 1-dr3
    else:
        sr = 1

    k = 4 
    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
        list = []
        for ii in range(0,k):
            list.append(conv0)
        out = tf.concat(list,axis=3)
        activation_summary(out)
        layers.append(out)

    print (out.get_shape())
    for i in range(n):
        #DEFINE JUMP BLOCKS AND PUT THEM HERE
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            #SET JUMP VARIABLE SCOPE HERE
            #out = residual_block(layers[-1],16*k,layers[-1])
            out = block_jump3(layers[-1],16*k, sr)
            activation_summary(out)
            layers.append(out)
    
    avg_pool1 = tf.nn.pool(layers[-1],[2,2],'AVG','VALID',strides=[2,2])
    list = []
    for ii in range(0,2):
        list.append(avg_pool1)
    input_stage2 = tf.concat(list,axis=3)
    layers.append(input_stage2)

    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            #conv2 = residual_block(layers[-1],32*k,layers[-1])
            conv2 = block_jump3(layers[-1],32*k, sr)
            activation_summary(conv2)
            layers.append(conv2)

    avg_pool2 = tf.nn.pool(layers[-1],[2,2],'AVG','VALID',strides=[2,2])
    list = []
    for ii in range(0,2):
        list.append(avg_pool2)
    input_stage3 = tf.concat(list,axis=3)
    layers.append(input_stage3)

    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            #conv3 = residual_block(layers[-1],64*k,layers[-1])
            conv3 = block_jump3(layers[-1],64*k, sr)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [8, 8, 64*k]

    with tf.variable_scope('final_pooling', reuse=reuse):
        bn = batch_normalization_layer(layers[-1],64*k)
        relu = tf.nn.relu(bn)
        avg_pool = tf.reduce_mean(relu, [1, 2]) 
        out = output_layer(avg_pool, 100)
        layers.append(out)
    
    params = num_param()
    print (params)
    return layers[-1]

def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    result = inference(input_tensor, 2, reuse=False, is_training=0)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

batch_size = 128
W = 32
H = 32
C = 3
n = 2 
reuse = False
data = np.random.randn(batch_size,W,H,C).astype(np.float32)
input_tensor_batch = tf.stack(data) 
is_training = 1

if __name__ == "__main__":
    with tf.Session() as sess: 
        y = inference(input_tensor_batch, n, reuse, is_training)
        init = tf.global_variables_initializer()
        sess.run(init)
        sess.run(y)
        start = time.time()
        sess.run(y)
        end = time.time()
        forward_time = end-start
        print ("Forward Prop time!  ", (end - start))

        gradient = tf.gradients(y,input_tensor_batch, colocate_gradients_with_ops=True)
        sess.run(gradient) 
        start = time.time() 
        sess.run(gradient)
        end = time.time()
        grad_time = end-start
        print ("gradient evalution time!  ", (end - start))
        print ("Back Prop time!  ", grad_time - forward_time)

