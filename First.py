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

print(sess.run(a+b, {a: 3.5, b: 4.5}))
print(sess.run(a+b, {a: [3, 2], b: [4, 5]}))
