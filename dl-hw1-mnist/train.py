# coding=utf-8
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.callbacks import TensorBoard
import numpy


model = Sequential()
model.add(Dense(input_dim=28*28,units=500)) # input_layer，28*28=784
model.add(Activation('sigmoid')) # activation function:sigmoid/tanh/relu, sigmoid seems better
model.add(Dropout(0.3))

model.add(Dense(units=500)) # 500
model.add(Activation('sigmoid'))
model.add(Dropout(0.3))

model.add(Dense(units=10)) # 10 classes, so the output unit number is 10
model.add(Activation('softmax')) # softmax as the last layer of classification

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # set hyperparameters
model.compile(loss='categorical_crossentropy', optimizer=sgd) # set categorical_crossentropy as loss function

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
Y_train = (numpy.arange(10) == y_train[:, None]).astype(int)
Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)

model.fit(X_train, Y_train, batch_size=200, epochs=100, shuffle=True, verbose=1, validation_split=0.3, callbacks=[TensorBoard(log_dir='./logs')])

print ('test set')
model.evaluate(X_test, Y_test, batch_size=200, verbose=1)
