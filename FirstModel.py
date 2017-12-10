import tensorflow as tf

W = tf.Variable([221.0], tf.float32)
b = tf.Variable([-9799.0], tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

model = W * x + b

loss = tf.reduce_sum(tf.square(model - y))

optimizer = tf.train.GradientDescentOptimizer(0.001)

train = optimizer.minimize(loss)

with tf.Session() as sess:
    fileWriter = tf.summary.FileWriter("./graph", sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(90000):
        sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    print(sess.run([W, b]))
