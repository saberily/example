#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from captcha.image import ImageCaptcha
from PIL import Image
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


number   = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j',
            'k','l','m','n','o','p','q','r','s','t',
            'u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J',
            'K','L','M','N','O','P','Q','R','S','T',
            'U','V','W','X','Y','Z']


#暂时只是用数字生成简单的1位验证码否则训练时间太长
MAX_CAPTCHA = 1
CHAR_SET = number
CHAR_SET_LEN = len(CHAR_SET)
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
#最后将rgb转换成1维的灰度图
IMAGE_CHANNEL = 1
#图片变化系数
IMAGE_SCALE = 1


#生成一个随机4位验文本证码（生成label）
def random_captcha_text(char_set=number+alphabet+ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


#获取验证码的text（label）和图片（feature）
def gen_captcha_text_and_image():
    image = ImageCaptcha()

    #暂时只是用数字生成简单的1位验证码否则训练时间太长
    captcha_text = random_captcha_text(CHAR_SET, MAX_CAPTCHA)
    #将列表转换成字符串
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)
    #可以将生成的验证码存储成图片
    #因为只使用number生成1位的验证码所以这里最多只会存储10张验证码
    image.write(captcha_text, 'captchas/' + captcha_text + '.jpg')

    captcha_image = Image.open(captcha)
    #将图片转换成ndarray
    captcha_image = np.array(captcha_image)
    #这里实际调试的时候注意一下shape
    #print(captcha_image.shape)
    return captcha_text, captcha_image


#rgb图片转换成灰度图，颜色对数字识别没有影响
#这里的img是一个ndarray
#rgb的shape为(60, 160, 3)
def rgb2gray(img):
    if len(img.shape) > 2:
        #求最后一个axis的均值，这里就是rgb的均值
        #axis=0计算第0层[]中所有以[]为一个数据单位的均值
        #axis=1计算第1层[]中所有以[]为一个数据单位的均值
        #axis=-1计算最后一层这里就是axis=2第2层中所有以[]为一个数据单位的均值，
        #没有[]括住的数据以逗号或空格为单位表示的是最内层的数据了
        gray = np.mean(img, axis=-1)
        #上面的转法较快，正规转法如下  
        #r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def text2vec(text):
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长 MAX_CAPTCHA 个字符')

    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)

    #i=index
    #c=char
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + int(c)
        #print('idx=', idx)
        vector[idx] = 1
    #每个验证码位是一个1维长度为CHAR_SET_LEN的向量
    return vector


# 向量转回文本
def vec2text(vec):
    text=[]
    #返回非0的数字下标
    char_pos = vec.nonzero()[0]
    #c为数组下标
    for i, c in enumerate(char_pos):
        number = c % 10
        text.append(str(number))

    return ''.join(text)


#验证码测试函数
def captcha_test():
    captcha_text, captcha_image = gen_captcha_text_and_image()
    print(captcha_text)
    print(text2vec(captcha_text))
    print(vec2text(text2vec(captcha_text)))

    f = plt.figure()
    #整个figure只有一个图像
    #将整个figure按3行，1列划分，将ax绘制在第1个划分的位置处
    ax = f.add_subplot(3, 1, 1)
    #增加captcha_text显示在图片0.1，0.9处，
    ax.text(0.1, 0.9, captcha_text, ha='center', va='center', transform=ax.transAxes)
    #增加图像显示
    #在当前subplot上绘制图片
    plt.imshow(captcha_image)

    #将整个figure按3行，1列划分，将ax绘制在第2个划分的位置处
    ax = f.add_subplot(3, 1, 2)
    image = rgb2gray(captcha_image)
    plt.imshow(image)
    plt.show()

    #numpy的flatten将数组转换成一维，并除以255做数据预处理
    image = image.flatten() / IMAGE_SCALE
    print(image)


# 生成一个训练batch
def get_next_batch(batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])

    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            #有时生成图像大小不是(60, 160, 3)
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = rgb2gray(image)

        #为第1维index=i处赋值
        batch_x[i,:] = image.flatten() / IMAGE_SCALE
        batch_y[i,:] = text2vec(text)
    return batch_x, batch_y


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
        biases = tf.Variable(bias_init_val, trainable=True, name=scope+"b")

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
    #全连接层需先做扁平化处理
    shp = input_op.get_shape()
    #shp[0] -- 样本数使用-1自适应即可
    #shp[1] -- h
    #shp[2] -- w
    #shp[3] -- c
    n_in = shp[1].value * shp[2].value * shp[3].value
    #-1为样本数自适应
    resh = tf.reshape(input_op, [-1, n_in], name=name+"resh")

    with tf.name_scope(name) as scope:
        #由[-1, n_in] * [n_in, n_out] 得到 [-1, n_out]
        kernel = tf.get_variable(scope+"w",
            shape=[n_in, n_out], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out],
                                         dtype=tf.float32), name=scope+"b")
        #relu_layer先做矩阵相乘，然后加上biases，之后做激活操作
        activation = tf.nn.relu_layer(resh, kernel, biases, name=scope)
        activation_drop = tf.nn.dropout(activation, keep_prob, name=scope+"_drop")
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


#the begining
#输入定义
#IMAGE_HEIGHT = 60 IMAGE_WIDTH = 160
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
#dropout
keep_prob = tf.placeholder(tf.float32)
learning_rate = 0.001


#定义一个验证码识别cnn网络
def crack_captcha_cnn():
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    #定义网络结构
    #conv1-pool1-conv2-pool2-conv3-pool3-fc1(fc内接drop_out)
    layerlist = []
    layerlist.append([LAYER_TYPE_CONV, "conv1", 3, 3, 32, 1, 1])
    layerlist.append([LAYER_TYPE_POOL, "pool1", 2, 2, 2, 2])
    layerlist.append([LAYER_TYPE_CONV, "conv2", 3, 3, 64, 1, 1])
    layerlist.append([LAYER_TYPE_POOL, "pool2", 2, 2, 2, 2])
    layerlist.append([LAYER_TYPE_CONV, "conv3", 3, 3, 64, 1, 1])
    layerlist.append([LAYER_TYPE_POOL, "pool4", 2, 2, 2, 2])
    layerlist.append([LAYER_TYPE_FC,   "fc1",   10])
    output, p = cnn_op(x, layerlist)
    return output, p


#训练一个验证码识别的cnn网络
def train_crack_captcha_cnn():
    output, p = crack_captcha_cnn()
    #print(output.get_shape())
    #print(Y.get_shape())
    #计算loss
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    #定义优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    #计算预测值
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    #获取预测值向量被置一的下标
    max_idx_p = tf.argmax(predict, 2)
    print(max_idx_p)
    #获取真实值向量被置一的下标
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    print(max_idx_l)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    print(correct_pred)
    #计算一个batch的准确率, axis=None所有维度统计最终计算出一个数
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #保存模型
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(128)
            _, loss_, max_idx_p_, max_idx_l_, correct_pred_ = sess.run([optimizer, loss, max_idx_p, max_idx_l, correct_pred],
                                                                        feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.})
            acc_ = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.})
            print("loss:", step, loss_, acc_)
            #print(max_idx_p_)
            #print(max_idx_l_)
            #print(correct_pred_)


            #每10 step计算一次准确率  
            if step % 10 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print("acc:", step, acc)
                #如果准确率大于50%,保存模型,完成训练
                #越精确需要训练时间越长
                if acc > 0.50:
                    #saver.save(sess, "./model/crack_capcha.model", global_step=step)
                    saver.save(sess, "./model/crack_capcha.model")
                    break
            step += 1


#测试验证码识别网络训练的模型
def crack_captcha():
    text, image = gen_captcha_text_and_image()
    image = rgb2gray(image)
    image = image.flatten() / IMAGE_SCALE
    output, p = crack_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model/crack_capcha.model")
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [image], keep_prob: 1})
        print(text_list)
        print(text_list.shape)
        predict_text = text_list[0].tolist()
        print("正确: {}  预测: {}".format(text, predict_text))


if __name__ == '__main__':
    flag = 1

    #验证码生成测试
    if flag == 0:
        captcha_test()
    #模型训练
    elif flag == 1:
        train_crack_captcha_cnn()
    #模型测试
    elif flag == 2:
        crack_captcha()
    else:
        pass