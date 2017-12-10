import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

adder_node = x * y

with tf.Session() as sess:
    fileWriter = tf.summary.FileWriter("./graph", sess.graph)
    print(sess.run(adder_node, {x: [123, 312], y: [4554, 121]}))
