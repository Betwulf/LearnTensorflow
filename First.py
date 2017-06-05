import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print(node3)
print("sess:", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
aplusb = a + b

print(sess.run(a + b, {a: 3.5, b: 4.5}))
print(sess.run(aplusb, {a: [3, 2], b: [4, 5]}))

tripleab = 3 * aplusb

print(sess.run(tripleab, {a: [3, 2], b: [4, 5]}))

m = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = m*x + b

init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

y = tf.placeholder(tf.float32)

errorsquared = tf.reduce_sum(tf.square(linear_model - y))
print(sess.run(errorsquared, {x: [1, 2, 3, 4], y:[0, -1, -2, -3]}))
fixm = tf.assign(m, [-1])
fixb = tf.assign(b, [1])
sess.run([fixm, fixb])
print(sess.run(errorsquared, {x: [1, 2, 3, 4], y:[0, -1, -2, -3]}))

