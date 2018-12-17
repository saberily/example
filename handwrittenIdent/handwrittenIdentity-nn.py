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

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


def multilayer_perceptron(_X, _input, _layerslist, _classes):
    wlist = []
    blist = []
    step = 1
    for i in _layerslist:
        #第一层权重
        if step == 1:
            wlist.append(tf.Variable(tf.random_normal([_input, i], stddev=stddev)))
        else:
            wlist.append(tf.Variable(tf.random_normal([prenodenum, i], stddev=stddev)))
        blist.append(tf.Variable(tf.random_normal([i])))
        prenodenum = i
        step += 1
    #out层权重
    wlist.append(tf.Variable(tf.random_normal([prenodenum, _classes], stddev=stddev)))
    blist.append(tf.Variable(tf.random_normal([_classes])))

    layer_tmp = _X
    for w, b in zip(wlist, blist):
        layer_tmp = tf.nn.sigmoid(tf.add(tf.matmul(layer_tmp, w), b))
    return (layer_tmp)

pred = multilayer_perceptron(x, n_input, n_layerslist, n_classes)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optm = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
corr = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(corr, "float"))


#git上传不了很大的文件，此数据集经过处理很小，所以不会很准
filenames = ['MNIST Train 60k 28x28.csv']
filesize = 60 * 1000
filename_queue = tf.train.string_input_producer(filenames, num_epochs=training_epochs, shuffle=False)
#定义Reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
#定义Decoder
record_defaults = []
for i in range(n_input + n_classes):
    #解析为整数
    record_defaults.append([1])
tmp = tf.decode_csv(value,record_defaults=record_defaults)
tmp1 = tmp[0:n_classes]
tmp2 = tmp[n_classes:]
features = tf.stack(list(tmp2))
label = tf.stack(list(tmp1))
example_batch, label_batch = tf.train.shuffle_batch([features,label], batch_size=batch_size, capacity=2*batch_size, min_after_dequeue=batch_size, num_threads=2)


#git上传不了很大的文件，此数据集经过处理很小，所以不会很准
test_filenames = ['MNIST Test 10k 28x28.csv']
test_filesize = 10 * 1000
test_filename_queue = tf.train.string_input_producer(test_filenames, shuffle=False)
#定义Reader
test_reader = tf.TextLineReader()
test_key, test_value = test_reader.read(test_filename_queue)
#定义Decoder
test_record_defaults = []
for i in range(n_input + n_classes):
    #解析为整数
    test_record_defaults.append([1])
test_tmp = tf.decode_csv(test_value,record_defaults=test_record_defaults)
test_tmp1 = test_tmp[0:n_classes]
test_tmp2 = test_tmp[n_classes:]
test_features = tf.stack(list(test_tmp2))
test_label = tf.stack(list(test_tmp1))
test_example_batch, test_label_batch = tf.train.shuffle_batch([test_features,test_label], batch_size=test_filesize, capacity=test_filesize, min_after_dequeue=0, num_threads=2)



init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    sess.run(local_init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(filesize/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = sess.run([example_batch, label_batch])
            feeds = {x : batch_xs, y : batch_ys}
            sess.run(optm, feed_dict=feeds)
            avg_cost += sess.run(cost, feed_dict=feeds)
        avg_cost = avg_cost / total_batch

        if (epoch + 1) % display_step == 0:
            print("Epoch:%03d/%03d cost:%.9f" % (epoch, training_epochs, avg_cost))
            feeds = {x : batch_xs, y : batch_ys}
            train_acc = sess.run(accr, feed_dict=feeds)
            print("Train Accuracy:%.3f" % (train_acc))

            batch_xs, batch_ys = sess.run([test_example_batch, test_label_batch])
            feeds = {x : batch_xs, y : batch_ys}
            test_acc = sess.run(accr, feed_dict=feeds)
            print("Test Accuracy:%.3f" % (test_acc))
    coord.request_stop()
    coord.join(threads)

    saver.save(sess, "./model/crack_handwritten.model")

print("Optimization Finished")