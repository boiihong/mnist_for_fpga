import tensorflow as tf
import random
# import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# load mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset


tf.reset_default_graph()

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# create weight variables
W1 = tf.get_variable("W1", shape= [784,1024])
W2 = tf.get_variable("W2", shape= [1024,1024])
W3 = tf.get_variable("W3", shape= [1024,784])
W4 = tf.get_variable("W4", shape= [784,512])
W5 = tf.get_variable("W5", shape= [512,10])


# do quantization.. 
#W1 = tf.quantize(W1, -1.0,1.0 ,tf.qint16, "SCALED")
#W2 = tf.quantize(W2, -1.0,1.0 ,tf.qint16, "SCALED")
#W3 = tf.quantize(W3, -1.0,1.0 ,tf.qint16, "SCALED")
#W4 = tf.quantize(W4, -1.0,1.0 ,tf.qint16, "SCALED")
#W5 = tf.quantize(W5, -1.0,1.0 ,tf.qint16, "SCALED")



# network definition
L1 = tf.nn.tanh(tf.matmul(X, W1))
L2 = tf.nn.tanh(tf.matmul(L1 , W2))
L3 = tf.nn.tanh(tf.matmul(L2 , W3))
L4 = tf.nn.tanh(tf.matmul(L3 , W4))
hypothesis = tf.matmul(L4 , W5)


# restore ckpt
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, 'model/model.ckpt')
print("model restored")

# Test model and check accuracy
keep_prob = tf.placeholder(tf.float32)
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))


np.savetxt("model/W1.csv", (np.around(W1.eval(session=sess) * 1e2, 0)).astype(int), delimiter=',', fmt='%d')
np.savetxt("model/W2.csv", (np.around(W2.eval(session=sess) * 1e2, 0)).astype(int), delimiter=',', fmt='%d')
np.savetxt("model/W3.csv", (np.around(W3.eval(session=sess) * 1e2, 0)).astype(int), delimiter=',', fmt='%d')
np.savetxt("model/W4.csv", (np.around(W4.eval(session=sess) * 1e2, 0)).astype(int), delimiter=',', fmt='%d')
np.savetxt("model/W5.csv", (np.around(W5.eval(session=sess) * 1e2, 0)).astype(int), delimiter=',', fmt='%d')
