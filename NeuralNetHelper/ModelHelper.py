import tensorflow as tf
# from keras.constraints import max_norm
import tensorflow.contrib.slim as slim


class Model(object):
    def __init__(self, input_batch):
        self.scale = None
        self.cb_size = None
        self.train = tf.placeholder(tf.bool, name="is_train")
        self.inference = []
        self.inference_learned = []
        self.wo_soft = []
        self.input_batch = input_batch

    def build_model(self, scale_softmax, codebook_size):
        self.scale = scale_softmax
        self.cb_size = codebook_size
        """
        Define your Network here
        :param input_batch: feature batch
        :return y: output network
        """

        # Start here to define your network
        num_neurons = 512
        # self.input_batch = tf.Print(self.input_batch, [self.input_batch])

        # fc1 = slim.layers.fully_connected(self.input_batch, 512)
        fc1 = tf.layers.dense(self.input_batch, num_neurons, activation=tf.nn.relu)
        fc1_bn = tf.layers.batch_normalization(fc1, training=self.train, center=False, scale=False)
        # fc1_dropout = tf.layers.dropout(fc1_bn, rate=0.3, training=self.train)
        fc2 = tf.layers.dense(fc1_bn, num_neurons, activation=tf.nn.relu)
        fc2_bn = tf.layers.batch_normalization(fc2, training=self.train, center=False, scale=False)
        # fc2_dropout = tf.layers.dropout(fc2_bn, rate=0.3, training=self.train)
        # fc3 = tf.layers.dense(fc2_bn, num_neurons, activation=tf.nn.relu)
        # fc3_bn = tf.layers.batch_normalization(fc3, training=self.train, center=False, scale=False)
        # #
        # fc4 = tf.layers.dense(fc3_bn, num_neurons, activation=tf.nn.relu)
        # fc4_bn = tf.layers.batch_normalization(fc4, training=self.train, center=False, scale=False)
        # #
        # fc5 = tf.layers.dense(fc4_bn, num_neurons, activation=tf.nn.sigmoid)
        # fc5_bn = tf.layers.batch_normalization(fc5, training=self.train)
        #
        # fc6 = tf.layers.dense(fc5_bn, self.cb_size, activation=tf.nn.sigmoid)
        # fc6_bn = tf.layers.batch_normalization(fc6, training=self.train)
        # fc3_dropout = tf.layers.dropout(fc3_bn, rate=0.3, training=self.train)
        # fc4 = tf.layers.dense(fc3_bn, self.cb_size, activation=None, trainable=False,
        #                       kernel_initializer=tf.initializers.identity,
        #                       bias_initializer=tf.initializers.zeros)
        out = tf.layers.dense(fc2_bn, self.cb_size, activation=tf.nn.sigmoid)
        out_scaled = tf.scalar_mul(self.scale, out)
        self.wo_soft = out_scaled
        self.inference = tf.nn.softmax(out_scaled, name='nn_output')
        self.inference_learned = tf.layers.dense(self.inference, 127, activation=tf.nn.sigmoid)
        # self.inference_learned = tf.Print(self.inference_learned, [self.inference_learned], summarize=127)

        # end of definition of network

    def loss(self, phoneme_batch, log_cn_pr):
        output_nn = self.inference
        phoneme_batch = tf.cast(phoneme_batch, dtype=tf.int32)  # cast to int and put them in [[alignments]]
        batch_size = tf.cast(tf.shape(phoneme_batch)[0], dtype=tf.float32)

        # gather all data --> s~(t) for eq 5.50
        gather_data = tf.gather_nd(log_cn_pr, phoneme_batch)

        # s2 (summand 2)
        labels_loss = tf.multiply(gather_data, output_nn)
        labels_loss = tf.expand_dims(tf.reduce_sum(labels_loss, axis=1), 1)
        # s2 = tf.tile(s2, [1, tf.shape(output_soft)[1]])

        labels_loss *= output_nn

        # s2 = tf.Print(s2, [s2])
        # s2 = gather_data - s2
        # s2 = scale_soft / bat_size * tf.losses.mean_squared_error(gather_data, s2)

        # s2 = -self.scale / batch_size * tf.losses.sigmoid_cross_entropy(labels_loss, output_nn * gather_data)
        # s2 = -self.scale / batch_size * tf.losses.sigmoid_cross_entropy(labels_loss, output_nn * gather_data)
        # s2 = -tf.losses.softmax_cross_entropy(tf.one_hot(tf.squeeze(phoneme_batch), 41, axis=1),
        #                                      tf.tensordot(self.inference, tf.transpose(log_cn_pr), axes=1))
        # s2 = -tf.losses.sigmoid_cross_entropy(tf.one_hot(tf.squeeze(phoneme_batch), 41, axis=1),
        #                                                 tf.tensordot(self.inference, tf.transpose(log_cn_pr), axes=1))

        # s2 = -tf.reduce_mean(-tf.reduce_sum(tf.one_hot(tf.squeeze(phoneme_batch), 41, axis=1) *
        #                                    tf.log(tf.tensordot(self.inference, tf.transpose(log_cn_pr), axes=1)),
        #                                    reduction_indices=[1]))

        s2 = tf.losses.softmax_cross_entropy(labels_loss, self.wo_soft)

        # s2 = tf.losses.softmax_cross_entropy(tf.tensordot(self.inference, tf.transpose(log_cn_pr), axes=1), self.wo_soft)

        return s2

    def new_loss(self, phoneme_batch, new_cond):
        phoneme_batch = tf.cast(phoneme_batch, dtype=tf.int32)  # cast to int and put them in [[alignments]]

        # gather_data = tf.gather_nd(tf.transpose(new_cond), phoneme_batch)

        # one_hot = tf.one_hot(tf.squeeze(phoneme_batch), 41, axis=1)
        # labels_new = tf.tensordot(one_hot, new_cond, axes=1)
        # labels_new = tf.Print(labels_new, [labels_new])
        # loss = tf.losses.softmax_cross_entropy(tf.tensordot(tf.one_hot(tf.squeeze(phoneme_batch), 41, axis=1), new_cond,
        #                                        axes=1), self.wo_soft)
        # ----
        # works with normal cond_prob
        # loss = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(tf.squeeze(phoneme_batch), 127, axis=1) *
        #                                    tf.log(tf.tensordot(self.inference, tf.transpose(new_cond), axes=1)),
        #                                    reduction_indices=[1]))
        # ---

        # --- loss test with learned output layer
        # loss = tf.losses.mean_squared_error(tf.one_hot(tf.squeeze(phoneme_batch), 127, axis=1),
        #                                     self.inference_learned)
        loss = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(tf.squeeze(phoneme_batch), 127, axis=1) *
                                           tf.log(self.inference_learned),
                                           reduction_indices=[1]))


        return loss


