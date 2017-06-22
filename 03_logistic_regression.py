import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import png
import numpy as np


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Generate image files to see what the data looks like
train = mnist.train
v = train.next_batch(10)
i = 0
for img in v[0]:
    pixels = [x*255 for x in img]
    pixel_matrix = np.array(pixels)
    pixel_matrix = pixel_matrix.reshape(28, 28).tolist()
    png.from_array(pixel_matrix, 'L').save('img_{}.png'.format(i))
    i += 1


# parameters for learning stuff
learning_rate = 0.1
epochs = 50
batch_size = 100
display_step = 1

data_x = tf.placeholder(tf.float32, [None, 784])
data_y = tf.placeholder(tf.float32, [None, 10])  # 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# model
softmax_regression = tf.nn.softmax(tf.matmul(data_x, W) + b)


# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(data_y * tf.log(softmax_regression), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            _, c = sess.run([optimizer, cost], feed_dict={data_x: batch_xs,
                                                          data_y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(softmax_regression, 1), tf.argmax(data_y, 1))
    # Calculate accuracy for 3000 examples
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({data_x: mnist.test.images[:3000], data_y: mnist.test.labels[:3000]}))

    # Test model (John style)
    out = sess.run(softmax_regression, feed_dict={data_x: v[0]})
    print([np.argmax(x) for x in out])
    print(out)
