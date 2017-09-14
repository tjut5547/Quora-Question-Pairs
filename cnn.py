import tensorflow as tf


class Cnn(object):
    def __init__(self, sequence_length, vocab_size, embedding_size,
                 filter_sizes, num_filters, num_classes, num_sample):
        self.label = tf.placeholder(tf.float32, [None], name="label")
        self.input_sentence_a = tf.placeholder(tf.int32, [None, sequence_length], name="input_a")
        self.input_sentence_b = tf.placeholder(tf.int32, [None, sequence_length], name="input_b")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)
        with tf.name_scope("embedding_layer"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name = "W")
            self.embedded_chars_a = tf.nn.embedding_lookup(self.W, self.input_sentence_a)
            self.embedded_chars_b = tf.nn.embedding_lookup(self.W, self.input_sentence_b)
            self.embedded_chars_a_expand = tf.expand_dims(self.embedded_chars_a, -1)
            self.embedded_chars_b_expand = tf.expand_dims(self.embedded_chars_b, -1)

        pooled_outputs_a = []
        pooled_outputs_b = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                '''
                filter_size * embedding_size * 1 : 长 * 宽 * channel
                num_filters : 卷积核个数
                '''
                # Wa = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Wa")
                Wb = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="Wb")
                ba = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="ba")
                bb = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="bb")
                conv_a = tf.nn.conv2d(self.embedded_chars_a_expand, Wb,
                                      strides=[1, 1, 1, 1], padding="VALID")
                conv_b = tf.nn.conv2d(self.embedded_chars_b_expand, Wb,
                                      strides=[1, 1, 1, 1], padding="VALID")
                relu_a = tf.nn.relu(tf.nn.bias_add(conv_a, ba), name="relu_a")
                relu_b = tf.nn.relu(tf.nn.bias_add(conv_b, bb), name="relu_b")
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

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_a = tf.concat(pooled_outputs_a, 3)
        self.h_pool_b = tf.concat(pooled_outputs_b, 3)
        self.h_pool_flat_a = tf.reshape(self.h_pool_a, [-1, num_filters_total])
        self.h_pool_flat_b = tf.reshape(self.h_pool_b, [-1, num_filters_total])

        with tf.name_scope("output"):
            self.norm_a = tf.nn.l2_normalize(self.h_pool_flat_a, dim=1)
            self.norm_b = tf.nn.l2_normalize(self.h_pool_flat_b, dim=1)
            self.const = tf.constant(num_sample * [0.999999], dtype=tf.float32)
            self.result = tf.add(tf.reduce_sum(tf.multiply(self.norm_a, self.norm_b), 1), self.const)
            self.predictions = tf.cast(tf.cast(self.result, tf.int32), tf.float32)

        with tf.name_scope("loss"):
            self.loss = tf.losses.log_loss(predictions=self.result, labels=self.label)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.label)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

cnn = Cnn(sequence_length=35,
          vocab_size=1000,
          embedding_size=50,
          filter_sizes=[1, 2, 3, 4, 5],
          num_filters=10,
          num_classes=2,
          num_sample=10)
