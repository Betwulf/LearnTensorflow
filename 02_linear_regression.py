
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

epochs = 500
data_x = np.array([1,3,5,7,9,11])
data_y = data_x * 1.5 + 4 + (np.random.random()*8 - 4)
population = data_x.shape[0]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

M = tf.Variable(np.random.random(), name="weight")
b = tf.Variable(np.random.random(), name="offset")


lin = tf.add(tf.multiply(M, X), b)

delta = lin - Y
mean_square_error = tf.reduce_sum(tf.pow(delta, 2)) / (population)
# mse = tf.reduce_sum(tf.pow(tf.subtract(lin, Y), 2)) / (2*population)

trainer = tf.train.GradientDescentOptimizer(0.13).minimize(mean_square_error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for (x, y) in zip(data_x, data_y):
            # d = sess.run(delta, feed_dict={X: x, Y: y})
            m = sess.run(mean_square_error, feed_dict={X: x, Y: y})
            c = sess.run(trainer, feed_dict={X: x, Y: y})
        if epoch % 50 == 0:
            print("epoch: ", epoch, " Error: ", m)
            print(" M: ", sess.run(M), " b: ", sess.run(b))
            # print("delta: ", d, " mean_square_error: ", m)

    result_M = sess.run(M)
    result_b = sess.run(b)
    print("Done. f(x) = ", result_M, "*x + ", result_b)

    plt.plot(data_x, data_y, 'ro', label='Original data')
    plt.plot(data_x, sess.run(M) * data_x + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

