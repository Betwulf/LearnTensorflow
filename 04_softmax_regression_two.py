import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import png
import numpy as np


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()


data_x = tf.placeholder(tf.float32, [None, 784])
data_y = tf.placeholder(tf.float32, [None, 10])  # 10 classes


# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

y = tf.matmul(data_x, W) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=data_y, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={data_x: batch[0], data_y: batch[1]})


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(data_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={data_x: mnist.test.images, data_y: mnist.test.labels}))
