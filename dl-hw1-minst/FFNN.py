# -*- coding=utf-8 -*-
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#some constants
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_x_minmax = mnist.train.images
train_y_data = mnist.train.labels
test_x_minmax = mnist.test.images
test_y_data = mnist.test.labels
eval_tensorflow = True
max_epoch = 1000
learning_rate = 10e-2

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


if eval_tensorflow:
    print "Start evaluating softmax regression model by tensorflow..."
    # reformat y into one-hot encoding style
    lb = preprocessing.LabelBinarizer()
    lb.fit(train_y_data)
    train_y_data_trans = lb.transform(train_y_data)
    test_y_data_trans = lb.transform(test_y_data)

    x = tf.placeholder(tf.float32, [None, 784])
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10]))
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]))
        variable_summaries(b)
    with tf.name_scope('Wx_plus_b'):
        V = tf.matmul(x, W) + b
        tf.summary.histogram('pre_activations', V)
    with tf.name_scope('softmax'):
        y = tf.nn.softmax(V)
        tf.summary.histogram('activations', y)

    y_ = tf.placeholder(tf.float32, [None, 10])

    with tf.name_scope('cross_entropy'):
        loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        tf.summary.scalar('cross_entropy', loss)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    with tf.name_scope('evaluate'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('log/train', sess.graph)
    test_writer = tf.summary.FileWriter('log/test')

    for step in range(max_epoch):
        if step % 10 == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict={x: test_x_minmax, y_: test_y_data_trans})
            test_writer.add_summary(summary, step)
            print "step %d accuracy=%.4f" % (step, acc)
        else:
            sample_index = np.random.choice(train_x_minmax.shape[0], 100)
            batch_xs = train_x_minmax[sample_index, :]
            batch_ys = train_y_data_trans[sample_index, :]
            summary, _ = sess.run([merged, train], feed_dict={x: batch_xs, y_: batch_ys})
            train_writer.add_summary(summary, step)

    print "Accuracy of test set: %f" % sess.run(accuracy, feed_dict={x: test_x_minmax, y_: test_y_data_trans})
