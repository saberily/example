#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from datetime import datetime
import math
import time
import tensorflow as tf


#input_op输入tensor
#name卷积层名称
#kh，kw卷积核高宽，n_out卷积核输出通道数
#dh，dw卷积时步长的高宽
#p参数列表
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    #获取输入的通道数
    #每个卷积核通道数和输入通道数保持一致
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        #卷积w定义[h, w, n_in, n_out]
        #kernel为卷积核
        kernel = tf.get_variable(scope+"w",
            shape=[kh, kw, n_in, n_out], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer_conv2d())

        #stride[batch, h, w, channel]
        #batch, channel 默认为1
        strides = [1, dh, dw, 1]
        conv = tf.nn.conv2d(input_op, kernel, strides, padding='SAME')

        #输出有几个通道bias的shape就是几
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')

        #z为得到的卷积层
        z = tf.nn.bias_add(conv, biases)
        #relu只做激活操作
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]

        return activation


#神经网络只有全连接层组成
#vgg最后需要通过全连接层来做矩阵转换
#input_op输入tensor
#name全连接层名
#n_out输出tensor
#p参数列表
def fc_op(input_op, name, n_out, p):
    #获取输入的通道数
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
            shape=[n_in, n_out], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out],
                                         dtype=tf.float32), name='b')
        #relu_layer先做矩阵相乘，然后加上biases，之后做激活操作
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation


#input_op输入tensor
#name池化层名
#kh，kw池化尺寸
#dh，dw池化步长
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)


#vgg的前向传播
def inference_op(input_op, keep_prob):
    p = []

    #input_op为224*224*3
    #第一层卷积网络
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dh=2, dw=2)

    #pool1为112*112*64
    #第二层卷积网络
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    #poo2为56*56*128
    #第三层卷积网络
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    #poo3为28*28*256
    #第四层卷积网络
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    #pool4为14*14*512
    #第五层卷积网络
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dh=2, dw=2)

    #pool5为7*7*512
    shp = pool5.get_shape()
    #扁平化pool5准备做全连接
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    print(shp[1].value, shp[2].value, shp[3].value)
    #h自适应(这里为1)，w为flattened_shape的一维矩阵
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    #全连接层
    #输入resh1 shape = [1, flattened_shape]
    #输出fc6 shape = [1, 4096]
    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    #dropout根据keep_prob随机使fc6中4096个节点数中的一部分节点不生效
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")

    #输出fc8 shape = [1, 1000]
    fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, p


def time_tensorflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squraed = 0.0

    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                        (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squraed += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squraed / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
            (datetime.now(), info_string, num_batches, mn, sd))


def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size, 3],
                                               dtype=tf.float32,
                                               stddev=1e-1))
        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, p = inference_op(images, keep_prob)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        time_tensorflow_run(sess, predictions, {keep_prob:1.0}, "Forward")
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        time_tensorflow_run(sess, grad, {keep_prob:0.5}, "Forward-backward")


batch_size = 32
num_batches = 100
run_benchmark()

