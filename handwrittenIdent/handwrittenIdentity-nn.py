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
learning_rate   = 0.01
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


x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
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
        total_batch = int(filesize/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = sess.run([x_train_batch, y_train_batch])
            feeds = {x : batch_xs, y : batch_ys}
            sess.run(optm, feed_dict=feeds)
            avg_cost += sess.run(cost, feed_dict=feeds)
        avg_cost = avg_cost / total_batch

        if (epoch + 1) % display_step == 0:
            print("Epoch:%03d/%03d cost:%.9f" % (epoch, training_epochs, avg_cost))
            feeds = {x : batch_xs, y : batch_ys}
            train_acc = sess.run(accr, feed_dict=feeds)
            print("Train Accuracy:%.3f" % (train_acc))

            batch_xs, batch_ys = sess.run([x_test, y_test])
            feeds = {x : batch_xs, y : batch_ys}
            test_acc = sess.run(accr, feed_dict=feeds)
            print("Test Accuracy:%.3f" % (test_acc))

    coord.request_stop()
    coord.join(threads)

    saver.save(sess, "./model/crack_handwritten.model")
print("Optimization Finished")