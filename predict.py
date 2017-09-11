import tensorflow as tf

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.save(sess, "./model/cnn.model")
