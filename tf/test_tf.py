import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)

# Placeholder

x = tf.placeholder(tf.int32)

with tf.Session() as sess:
    # TODO: Feed the x tensor 123
    output = sess.run(x,feed_dict={x:123})
    print(output)

# Math Operations

# TODO: Convert the following to TensorFlow:
x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x,y),tf.constant(1,dtype=tf.float64)) # or use tf.cast

# TODO: Print z from a session
with tf.Session() as sess:
    output = sess.run(z)
    print(output)
