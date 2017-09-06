import tensorflow as tf
import numpy as np

sess = tf.Session()

a = tf.constant([1, 2, 3], dtype=tf.float16)
b = tf.constant([1, 2, 3], dtype=tf.float16)

c = tf.tensordot(a, b, axes=1)
d = tf.norm(a, ord=2)
help(tf.norm)

print (sess.run(a))
print (sess.run(b))
print (sess.run(c))
print (sess.run(d))
