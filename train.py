import tensorflow as tf
import pandas as pd
import datetime
import numpy as np
import time

from cnn import *
from preexe import *
from tensorflow.contrib import learn

import matplotlib.pyplot as plt

global_loss = []
global_accuracy = []

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = Cnn(sequence_length=50,
                  vocab_size=24000,
                  embedding_size=128,
                  filter_sizes=[2, 3, 4, 5],
                  num_filters=128,
                  num_classes=2)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(0.0005).minimize(cnn.loss, global_step=global_step)
        # grads_and_vars = optimizer.compute_gradients(cnn.loss)
        # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        # global_step : 每一轮参数值的更新之后就会增加1

        sess.run(tf.global_variables_initializer())

        def train_step(a_batch, b_batch, label):
            feed_dict = {
                cnn.input_sentence_a: a_batch,
                cnn.input_sentence_b: b_batch,
                cnn.label : label
            }
            _, step, loss, accuracy = sess.run([optimizer, global_step, cnn.loss, cnn.accuracy], feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, accuracy {}".format(time_str, step, loss, accuracy))
            global_loss.append(loss)
            global_accuracy.append(accuracy)

        def dev_step(a_batch, b_batch, label):
            feed_dict = {
                cnn.input_sentence_a: a_batch,
                cnn.input_sentence_b: b_batch,
                cnn.label: label
            }
            step, loss = sess.run([global_step, cnn.loss], feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))

        batches = get_batch(500)
        for data in batches:
            print(datetime.datetime.now().isoformat())
            x_train, y_train = zip(*data)
            x_train_a = [a for a, b in x_train]
            x_train_b = [b for a, b in x_train]
            y_train = [((1 + label[0]) % 2, (0 + label[0]) % 2) for label in y_train]

            train_step(x_train_a, x_train_b, y_train)
            current_step = tf.train.global_step(sess, global_step)
            # if current_step % 100 == 0:
            #     print("\nEvaluation:")
            #     dev_step(x_train_a, x_train_b, y_train)#, writer=dev_summary_writer)
            #     print("")

        x = list(range(len(global_loss)))
        y = global_loss
        plt.plot(x, y, 'r', label="loss")
        plt.xlabel("batches")
        plt.ylabel("loss")
        plt.savefig("loss_modify.png")
        plt.close()

        plt.plot(x, global_accuracy, 'b', label="accuracy")
        plt.xlabel("batches")
        plt.ylabel("accuracy")
        plt.savefig("accuracy.png")
        plt.close()

