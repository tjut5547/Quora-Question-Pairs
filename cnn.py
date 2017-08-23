import tensorflow as tf

class Cnn(object):
    def __init__(self, sequence_length, vocab_size, embedding_size,
                 filter_sizes, num_filters):
        self.input_sentence_a = tf.placeholder(tf.int32, [None, sequence_length], name = "input_a")
        self.input_sentence_b = tf.placeholder(tf.int32, [None, sequence_length], name = "input_b")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)
        with tf.name_scope("embedding layer"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name = "W")
            self.embedded_chars_a = tf.nn.embedding_lookup(self.W, self.input_sentence_a)
            self.embedded_chars_b = tf.mm.embedding_lookup(self.w, self.input_sentence_b)
            self.embedded_chars_a_expand = tf.expand_dims(self.embedded_chars_a, -1)
            self.embedded_chars_b_expand = tf.expand_dims(self.embedded_chars_b, -1)


        with tf.name_scope("conv layer"):
            pooled_outputs_a = []
            pooled_outputs_b = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv_a = tf.nn.conv2d(self.embedded_chars_a_expanded, W,
                                        strides=[1, 1, 1, 1], padding="VALID")
                    conv_b = tf.nn.conv2d(self.embedded_chars_b_expanded, W,
                                          strides=[1, 1, 1, 1], padding="VALID")
                    relu_a = tf.nn.relu(tf.nn.bias_add(conv_a, b), name="relu_a")
                    relu_b = tf.nn.relu(tf.nn.bias_add(conv_b, b), name="relu_b")
                    pooled_a = tf.nn.max_pool(
                        relu_a,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")

                    pooled_b = tf.nn.max_pool(
                        relu_b,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs_a.append(pooled_a)
                    pooled_outputs_b.append(pooled_b)