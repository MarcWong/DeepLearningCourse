# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tflearn.datasets.oxflower17 as oxflower17

X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

# Sequential implementation of AlexNet
network = input_data(shape=[None, 227, 227, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.2)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.2)
network = fully_connected(network, 17, activation='softmax')

# you may try momentum, sgd, adam, and the learning_rate should be correspondently adjusted
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.0002)

# Training
# set the checkpoint_path correspondently
model = tflearn.DNN(network, checkpoint_path='checkpoints/model_alexnet-momentum',
                    max_checkpoints=100, tensorboard_verbose=0,tensorboard_dir='logs/')

# Load model
model.load('checkpoints/model_alexnet-momentum-6000')

model.fit(X, Y, n_epoch=500, validation_set=0.2, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=100,
          snapshot_epoch=False, run_id='alexnet_oxflowers17-momentum',
           )

# Save model
# model.save('model_alexnet')
