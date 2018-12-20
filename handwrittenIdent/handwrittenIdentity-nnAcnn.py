#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf


'''
input -> l1 -> l2 -> ... -> out
'''


n_input    = 784
n_hidden_1 = 256
n_hidden_2 = 128
n_layerslist = [n_hidden_1, n_hidden_2]
n_classes    = 10
training_epochs = 10
batch_size      = 100
learning_rate   = 0.001
display_step    = 1
stddev = 0.1


#每个隐层的操作
#input_op输入tensor
#name此隐层名
#n_out输出维度
#p参数列表
def hidden_op(input_op, name, n_out, p):
    #获取输入的通道数
    #每个卷积核通道数和输入通道数保持一致
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        #w定义
        w = tf.get_variable(scope+"w",
                            shape=[n_in, n_out], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=stddev))
        #b定义
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        b = tf.Variable(bias_init_val, trainable=True, name='b')

        #hidden层操作
        p += [w, b]
        return tf.nn.sigmoid(tf.add(tf.matmul(input_op, w), b))


def multilayer_perceptron(_X, _layerslist, _classes):
    p = []
    step = 1
    hidden = _X
    for i in _layerslist:
        hidden = hidden_op(hidden, "hidden_" + str(i), i, p)
        step += 1
    return hidden_op(hidden, "output", _classes, p)


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
            initializer=tf.random_normal_initializer(stddev=0.1))
            #initializer=tf.contrib.layers.xavier_initializer_conv2d())

        #stride[batch, h, w, channel]
        #batch, channel 默认为1
        strides = [1, dh, dw, 1]
        conv = tf.nn.conv2d(input_op, kernel, strides, padding='SAME')

        #输出有几个通道bias的shape就是几
        bias_init_val = tf.constant(n_out, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')

        #z为得到的卷积层
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.sigmoid(z, name=scope)
        #relu只做激活操作
        #activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]

        return activation


#神经网络只有全连接层组成
#vgg最后需要通过全连接层来做矩阵转换
#input_op输入tensor
#name全连接层名
#n_out输出tensor
#p参数列表
def fc_op(input_op, name, n_out, p):
    #全连接层需先做扁平化处理
    shp = input_op.get_shape()
    #shp[0] -- 样本数使用-1自适应即可
    #shp[1] -- h
    #shp[2] -- w
    #shp[3] -- c
    n_in = 1
    step = 0
    for shp_i in shp:
        if 0 == step:
            pass
        else:
            n_in *= shp_i.value
        step += 1

    #-1为样本数自适应
    resh = tf.reshape(input_op, [-1, n_in], name=name+"resh")

    with tf.name_scope(name) as scope:
        #由[-1, n_in] * [n_in, n_out] 得到 [-1, n_out]
        kernel = tf.get_variable(scope+"w",
            shape=[n_in, n_out], dtype=tf.float32,
            #initializer=tf.contrib.layers.xavier_initializer())
            initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.Variable(tf.constant(0.0, shape=[n_out],
                                         dtype=tf.float32), name='b')
        #relu_layer先做矩阵相乘，然后加上biases，之后做激活操作
        activation_drop = tf.nn.sigmoid(tf.add(tf.matmul(resh, kernel), biases))
        #使用relu_layer不能收敛？？
        #activation_drop = tf.nn.relu(tf.add(tf.matmul(resh, kernel), biases))
        #activation_drop = tf.nn.relu_layer(resh, kernel, biases, name=scope)
        #activation_drop = tf.nn.dropout(activation_drop, keep_prob, name=scope)
        p += [kernel, biases]

        return activation_drop


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


LAYER_TYPE_CONV = 0
LAYER_TYPE_POOL = 1
LAYER_TYPE_FC   = 2
#定义conv，pool，fc层基本处理
#_input_op输入tensor
#layerlist网络结构
def cnn_op(_input_op, layerlist):
    p = []
    input_tmp = _input_op
    for layer in layerlist:
        layertype = layer[0]
        input_op = input_tmp
        if layertype == LAYER_TYPE_CONV:
            name = layer[1]
            kh = layer[2]
            kw = layer[3]
            n_out = layer[4]
            dh = layer[5]
            dw = layer[6]
            input_tmp = conv_op(input_op, name, kh, kw, n_out, dh, dw, p)
        elif layertype == LAYER_TYPE_POOL:
            name = layer[1]
            kh = layer[2]
            kw = layer[3]
            dh = layer[4]
            dw = layer[5]
            input_tmp = mpool_op(input_op, name, kh, kw, dh, dw)
        elif layertype == LAYER_TYPE_FC:
            name = layer[1]
            n_out = layer[2]
            input_tmp = fc_op(input_op, name, n_out, p)
        else:
            pass
    return input_tmp, p


x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

#定义一个验证码识别cnn网络
def crack_captcha_cnn():
    x_ = tf.reshape(x, shape=[-1, 28, 28, 1])

    #定义网络结构
    #conv1-pool1-conv2-pool2-conv3-pool3-fc1(fc内接drop_out)
    layerlist = []
    layerlist.append([LAYER_TYPE_CONV, "conv1", 3, 3, 128, 1, 1])
    layerlist.append([LAYER_TYPE_POOL, "pool1", 2, 2, 2, 2])
    layerlist.append([LAYER_TYPE_CONV, "conv2", 3, 3, 256, 1, 1])
    layerlist.append([LAYER_TYPE_POOL, "pool2", 2, 2, 2, 2])
    #layerlist.append([LAYER_TYPE_CONV, "conv3", 3, 3, 64, 1, 1])
    #layerlist.append([LAYER_TYPE_POOL, "pool3", 2, 2, 2, 2])
    layerlist.append([LAYER_TYPE_FC,   "fc1",   256])
    #layerlist.append([LAYER_TYPE_FC,   "fc2",   128])
    layerlist.append([LAYER_TYPE_FC,   "fc3",   10])
    output, p = cnn_op(x_, layerlist)
    return output

#cnn
pred1 = crack_captcha_cnn()
cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred1, labels=y))
#使用每个batch来优化w，b参数
optm1 = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost1)
#tf.argmax()获取[0,0,1,0,0,0]中1的下标
corr1 = tf.equal(tf.argmax(pred1, 1), tf.argmax(y, 1))
#求整个batch的准确率
accr1 = tf.reduce_mean(tf.cast(corr1, "float"))

#pred可以理解为一个batch处理完之后的返回的集合
#集合的大小即batch size的大小
#理解pred为一个batch size的集合即可
pred = multilayer_perceptron(x, n_layerslist, n_classes)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
#使用每个batch来优化w，b参数
optm = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
#tf.argmax()获取[0,0,1,0,0,0]中1的下标
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#求整个batch的准确率
accr = tf.reduce_mean(tf.cast(corr, "float"))


#csv数据处理
def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    #默认第1列为label
    defaults = [['null']]
    for i in range(n_input):
        #解析为整数
        defaults.append([1])
    tmp = tf.decode_csv(value, defaults)
    tmp1 = tmp[0:1]
    tmp2 = tmp[1:]

    example = tf.stack(list(tmp2))
    #label = tf.stack(list(tmp1))
    #将0~9转换成1维矩阵
    label = tf.case({
        tf.equal(tmp1[0], tf.constant('0')): lambda: tf.constant([1,0,0,0,0,0,0,0,0,0]),
        tf.equal(tmp1[0], tf.constant('1')): lambda: tf.constant([0,1,0,0,0,0,0,0,0,0]),
        tf.equal(tmp1[0], tf.constant('2')): lambda: tf.constant([0,0,1,0,0,0,0,0,0,0]),
        tf.equal(tmp1[0], tf.constant('3')): lambda: tf.constant([0,0,0,1,0,0,0,0,0,0]),
        tf.equal(tmp1[0], tf.constant('4')): lambda: tf.constant([0,0,0,0,1,0,0,0,0,0]),
        tf.equal(tmp1[0], tf.constant('5')): lambda: tf.constant([0,0,0,0,0,1,0,0,0,0]),
        tf.equal(tmp1[0], tf.constant('6')): lambda: tf.constant([0,0,0,0,0,0,1,0,0,0]),
        tf.equal(tmp1[0], tf.constant('7')): lambda: tf.constant([0,0,0,0,0,0,0,1,0,0]),
        tf.equal(tmp1[0], tf.constant('8')): lambda: tf.constant([0,0,0,0,0,0,0,0,1,0]),
        tf.equal(tmp1[0], tf.constant('9')): lambda: tf.constant([0,0,0,0,0,0,0,0,0,1]),
    }, lambda: tf.constant([0,0,0,0,0,0,0,0,0,0]), exclusive=True)

    return example, label


def create_pipeline(filename, batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    example, label = read_data(file_queue)

    min_after_dequeue = 1000
    capacity = min_after_dequeue + batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    return example_batch, label_batch


filesize = 60 * 1000
x_train_batch, y_train_batch = create_pipeline('MNIST Train 60k 28x28 dense.csv', batch_size, num_epochs=training_epochs)
#这里直接使用一个batch一次性计算所有测试集的准确率所以这里batch_size为测试集大小
x_test, y_test = create_pipeline('MNIST Test 10k 28x28 dense.csv', 10 * 1000)


init_op = tf.global_variables_initializer()
#local variables like epoch_num, batch_size
local_init_op = tf.local_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for epoch in range(training_epochs):
        avg_cost = 0
        avg_cost1 = 0
        total_batch = int(filesize/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = sess.run([x_train_batch, y_train_batch])
            feeds = {x : batch_xs, y : batch_ys}
            sess.run(optm, feed_dict=feeds)
            sess.run(optm1, feed_dict=feeds)
            avg_cost += sess.run(cost, feed_dict=feeds)
            avg_cost1 += sess.run(cost1, feed_dict=feeds)
        avg_cost = avg_cost / total_batch
        avg_cost1 = avg_cost1 / total_batch

        if (epoch + 1) % display_step == 0:
            print("Epoch:%03d/%03d cost:%.9f"     % (epoch, training_epochs, avg_cost))
            print("Epoch:%03d/%03d cost:%.9f -1-" % (epoch, training_epochs, avg_cost1))
            feeds = {x : batch_xs, y : batch_ys}
            train_acc = sess.run(accr, feed_dict=feeds)
            train_acc1 = sess.run(accr1, feed_dict=feeds)
            print("Train Accuracy:%.3f" % (train_acc))
            print("Train Accuracy:%.3f -1-" % (train_acc1))

            batch_xs, batch_ys = sess.run([x_test, y_test])
            feeds = {x : batch_xs, y : batch_ys}
            test_acc = sess.run(accr, feed_dict=feeds)
            test_acc1 = sess.run(accr1, feed_dict=feeds)
            print("Test Accuracy:%.3f" % (test_acc))
            print("Test Accuracy:%.3f -1-" % (test_acc1))

    coord.request_stop()
    coord.join(threads)

    saver.save(sess, "./model/crack_handwritten.model")
print("Optimization Finished")