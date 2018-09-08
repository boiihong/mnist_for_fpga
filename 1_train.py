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

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

# parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# weights & bias for nn layers
# http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
W1 = tf.get_variable("W1", shape=[784, 1024],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = tf.nn.tanh(tf.matmul(X, W1) )#+ b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[1024, 1024],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = tf.nn.tanh(tf.matmul(L1, W2))# + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[1024, 784],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = tf.nn.tanh(tf.matmul(L2, W3))# + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[784, 512],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = tf.nn.tanh(tf.matmul(L3, W4) )#+ b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[512, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) #+ b5

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())



# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')
save_path = saver.save(sess, "model/model.ckpt")
print('model saved in path: ',  save_path)

# do quantization.. 
#W1 = tf.quantize(W1, , ,tf.qint16, "SCALED")



# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))



# Get one and predict
#r = random.randint(0, mnist.test.num_examples - 1)
#print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
#print("Prediction: ", sess.run(
 #   tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))

# Save weight and bias
#saveParam(W1.eval(sess), PATH_MODEL + 'w1.param')
#saveParam(W2.eval(sess), PATH_MODEL + 'w2.param')
#saveParam(W3.eval(sess), PATH_MODEL + 'w3.param')
#saveParam(W4.eval(sess), PATH_MODEL + 'w4.param')
#saveParam(W5.eval(sess), PATH_MODEL + 'w5.param')

#saveParam(b1.eval(sess), PATH_MODEL + 'b1.param')
#saveParam(b2.eval(sess), PATH_MODEL + 'b2.param')
#saveParam(b3.eval(sess), PATH_MODEL + 'b3.param')
#saveParam(b4.eval(sess), PATH_MODEL + 'b4.param')
#saveParam(b5.eval(sess), PATH_MODEL + 'b5.param')

#testx = mnist.test.images[0:1]

#saveParam(sess.run(L1, feed_dict={X: testx, keep_prob:1}), PATH_MODEL + 'l1.param')
#saveParam(sess.run(L2, feed_dict={X: testx, keep_prob:1}), PATH_MODEL + 'l2.param')
#saveParam(sess.run(L3, feed_dict={X: testx, keep_prob:1}), PATH_MODEL + 'l3.param')
#saveParam(sess.run(L4, feed_dict={X: testx, keep_prob:1}), PATH_MODEL + 'l4.param')
#saveParam(sess.run(hypothesis, feed_dict={X: testx, keep_prob:1}), PATH_MODEL + 'l5.param')

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()
