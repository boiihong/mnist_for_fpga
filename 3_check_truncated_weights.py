import tensorflow as tf
import random
# import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

PATH_MODEL = 'model/'

def saveParam(npArray, filename):
	if len(npArray.shape) == 1:
		npArray = np.array([npArray]) # convert to 2d array
	shape = npArray.shape
	f = open(filename, 'w')
	f.write(str(shape[0]) + ' ' + str(shape[1]) + '\n')
	f.close()

	f = open(filename, 'ab')
	np.savetxt(f, npArray)
	f.close()

def my_loadtxt(filename, row, col):
	returnarr = np.zeros((row, col),dtype=int)
	f = open(filename)
	for i in range(0, row):
	  x = f.readline()
	  y = x.strip().split(',')
	  for j in range(0, col):
	    returnarr[i][j] = y[j]
	return returnarr

# load mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset


tf.reset_default_graph()

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
W1 = tf.placeholder(tf.float32, [784, 1024])
W2 = tf.placeholder(tf.float32, [1024, 1024])
W3 = tf.placeholder(tf.float32, [1024, 784])
W4 = tf.placeholder(tf.float32, [784, 512])
W5 = tf.placeholder(tf.float32, [512, 10])

# network definition
L1 = tf.nn.tanh(tf.matmul(X, W1))
L2 = tf.nn.tanh(tf.matmul(L1 , W2))
L3 = tf.nn.tanh(tf.matmul(L2 , W3))
L4 = tf.nn.tanh(tf.matmul(L3 , W4))
hypothesis = tf.matmul(L4 , W5)

# read weights
w1 = my_loadtxt('model/W1.csv', 784, 1024)
w2 = my_loadtxt('model/W2.csv', 1024, 1024)
w3 = my_loadtxt('model/W3.csv', 1024, 784)
w4 = my_loadtxt('model/W4.csv', 784, 512)
w5 = my_loadtxt('model/W5.csv', 512, 10)

# check min, max..
print("w1 min, max " , w1.min(), w1.max())
print("w2 min, max " , w2.min(), w2.max())
print("w3 min, max " , w3.min(), w3.max())
print("w4 min, max " , w4.min(), w4.max())
print("w5 min, max " , w5.min(), w5.max())

# session
sess = tf.Session()

# test model
keep_prob = tf.placeholder(tf.float32)
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels, 
      W1:w1, W2:w2, W3:w3, W4:w4, W5:w5, keep_prob: 1}))


# dump..
testx = mnist.test.images[0:1]

saveParam(sess.run(L1, feed_dict={X: testx,W1:w1, W2:w2, W3:w3, W4:w4, W5:w5, keep_prob:1}), PATH_MODEL + 'l1.param')
saveParam(sess.run(L2, feed_dict={X: testx,W1:w1, W2:w2, W3:w3, W4:w4, W5:w5, keep_prob:1}), PATH_MODEL + 'l2.param')
saveParam(sess.run(L3, feed_dict={X: testx,W1:w1, W2:w2, W3:w3, W4:w4, W5:w5, keep_prob:1}), PATH_MODEL + 'l3.param')
saveParam(sess.run(L4, feed_dict={X: testx,W1:w1, W2:w2, W3:w3, W4:w4, W5:w5, keep_prob:1}), PATH_MODEL + 'l4.param')
saveParam(sess.run(hypothesis, feed_dict={X: testx, W1:w1, W2:w2, W3:w3, W4:w4, W5:w5,keep_prob:1}), PATH_MODEL + 'l5.param')

