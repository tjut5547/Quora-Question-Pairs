import tensorflow as tf
import pandas as pd
import datetime

from cnn import *
from preexe import *
from tensorflow.contrib import learn


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement = True,
        log_device_placement = False
    )
    sess = tf.Session(config = session_conf)
    with sess.as_default():
        cnn = Cnn(sequence_length=100,
                  vocab_size=1000,
                  embedding_size=50,
                  filter_sizes=[1, 2, 3, 4],
                  num_filters=10)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        # global_step : 每一轮参数值的更新之后就会增加１

        sess.run(tf.global_variables_initializer())
        def train_step(a_batch, b_batch, label):
            feed_dict = {
                cnn.input_sentence_a : a_batch,
                cnn.input_sentence_b : b_batch,
                cnn.label : label
            }
            _, step, loss = sess.run([train_op, global_step, cnn.loss], feed_dict=feed_dict)

            # _ = sess.run([train_op], feed_dict=feed_dict)
            # _ = sess.run([global_step], feed_dict=feed_dict)
            # _ = sess.run([cnn.loss], feed_dict=feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))

        def dev_step(a_batch, b_batch, label):
            feed_dict = {
                cnn.input_sentence_a : a_batch,
                cnn.input_sentence_b : b_batch,
                cnn.label : label
            }
            step, loss, accuracy = sess.run([global_step, cnn.loss], feed_dict = feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))


        batches = get_batch(100)
        for data in batches:
            x_train, y_train = zip(*data)
            x_train_a = list(dict(x_train).keys())
            x_train_b = list(dict(x_train).values())
            y_train = [label[0] for label in y_train]

            print (x_train_a)
            print (x_train_b)
            print (y_train)

            train_step(x_train_a, x_train_b, y_train)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 100 == 0:
                dev_step(x_train[0], x_train[1], y_train)
