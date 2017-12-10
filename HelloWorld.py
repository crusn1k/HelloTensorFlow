import tensorflow as tf

hello = tf.constant('Hello World!', tf.string)

with tf.Session() as sess:
    print(sess.run(hello))
