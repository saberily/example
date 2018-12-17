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

    #暂时只是用数字生层简单的验证码
    char_set = number
    captcha_text = random_captcha_text(char_set)
    #将列表转换成字符串
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)
    #可以将生成的验证码存储成图片
    #image.write(captcha_text, 'captchas/' + captcha_text + '.jpg')

    captcha_image = Image.open(captcha)
    #将图片转换成ndarray
    captcha_image = np.array(captcha_image)
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
        raise ValueError('验证码最长4个字符')

    #MAX_CAPTCHA=4
    #CHAR_SET_LEN=10（0-9个数字）
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)

    #i=index
    #c=char
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + int(c)
        vector[idx] = 1
    #[[1][2][3][4]] 每个num 10位
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

    return "".join(text)


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

        batch_x[i,:] = image.flatten() / 255
        batch_y[i,:] = text2vec(text)

    return batch_x, batch_y


#定义CNN
#w_alpha和b_alpha用来控制初始化w，b参数的大小，可以不用
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    #3 conv layer
    #filter大小[3*3*1]  生成32个特征图
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    #Fully connected layer
    w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]))
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_CAPTCHA*CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_CAPTCHA*CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out


#训练
def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)  
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while True:
            batch_x, batch_y = get_next_batch(64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print(step, loss_)

            #每10 step计算一次准确率  
            if step % 10 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                #如果准确率大于10%,保存模型,完成训练
                #越精确需要训练时间越长
                if acc > 0.10:
                    #saver.save(sess, "./model/crack_capcha.model", global_step=step)
                    saver.save(sess, "./model/crack_capcha.model")
                    break
            step += 1


def crack_captcha(captcha_image):
    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model/crack_capcha.model")
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
        text = text_list[0].tolist()
        return text


if __name__ == '__main__':
    #是否训练
    train = 1
    if train == 1:
        text, image = gen_captcha_text_and_image()
        print("验证码图像shape:", image.shape)  #(60, 160, 3)
        #图像大小
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160
        MAX_CAPTCHA = len(text)
        print("验证码文本最长字符数", MAX_CAPTCHA)
        #文本转向量
        char_set = number
        CHAR_SET_LEN = len(char_set)
        print(text2vec("9876"))
        print(vec2text(text2vec("9876")))

        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
        #dropout
        keep_prob = tf.placeholder(tf.float32)
        train_crack_captcha_cnn()

    if train == 0:
        text, image = gen_captcha_text_and_image()
        IMAGE_HEIGHT = 60
        IMAGE_WIDTH = 160
        MAX_CAPTCHA = len(text)
        char_set = number
        CHAR_SET_LEN = len(char_set)

        f = plt.figure()
        ax = f.add_subplot(1, 1, 1)
        ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
        plt.imshow(image)
        plt.show()

        image = rgb2gray(image)
        #plt.imshow(image)
        #plt.show()
        print(image)
        #numpy的flatten将数组转换成一维，并除以255做数据预处理
        image = image.flatten() / 255
        print(image)

        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH])
        Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA*CHAR_SET_LEN])
        #dropout
        keep_prob = tf.placeholder(tf.float32)

        predict_text = crack_captcha(image)
        print("正确: {}  预测: {}".format(text, predict_text))


        text, image = gen_captcha_text_and_image()

        f = plt.figure()
        #整个figure只有一个图像
        #将整个figure按3行，1列划分，将ax绘制在低2个划分的位置处
        ax = f.add_subplot(3, 1, 2)
        #增加text，0.5行列的比例位置，
        ax.text(0.5, 0.5, text, ha='center', va='center', transform=ax.transAxes)
        #增加图像显示
        #在当前subplot上绘制图片
        plt.imshow(image)
        plt.show()
