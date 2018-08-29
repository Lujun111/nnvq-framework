import tensorflow as tf


class Model(object):
    """
    Creates a model object
    """
    def __init__(self, input_batch, scale_softmax, codebook_size, restore):
        """
        Init the Model

        :param input_batch: size of batch
        """
        self.scale = scale_softmax
        self.cb_size = codebook_size
        self.restore = restore
        self.train = tf.placeholder(tf.bool, name="is_train")
        # self.train_out = tf.placeholder(tf.bool, name="train_out_layer")
        self.inference = []
        self.inference_learned = []
        self.wo_soft = []
        self.input_batch = input_batch

        # create model
        # if self.restore:
        #     self._build_restored_model()
        # else:
        #     self._build_model()

        self._build_model()

    def _build_model(self):
        """
        Build your model for training. All of the architecture
        is defined here.

        :param scale_softmax: Scaling for the WTA-layer (Winner-Takes-All)
        :param codebook_size: Size of the codebook
        """

        # Start here to define your network
        # ------------------------------------------------------------------
        num_neurons = 512

        fc1 = tf.layers.dense(self.input_batch, num_neurons, activation=tf.nn.relu)
        fc1_bn = tf.layers.batch_normalization(fc1, training=self.train, center=False, scale=False)

        # fc1_dropout = tf.layers.dropout(fc1_bn, rate=0.3, training=self.train)
        fc2 = tf.layers.dense(fc1_bn, num_neurons, activation=tf.nn.relu)
        fc2_bn = tf.layers.batch_normalization(fc2, training=self.train, center=False, scale=False)
        # fc2_dropout = tf.layers.dropout(fc2_bn, rate=0.3, training=self.train)

        # WTA-layer starts here
        out = tf.layers.dense(fc2_bn, self.cb_size, activation=tf.nn.sigmoid)
        out_scaled = tf.scalar_mul(self.scale, out)
        # output without softmax
        self.wo_soft = out_scaled
        # output with soft, be aware use a name 'nn_output' for the output node!
        self.inference = tf.nn.softmax(out_scaled, name='nn_output')
        # learn the mapping
        self.inference_learned = tf.layers.dense(self.inference, 127, activation=tf.nn.sigmoid)

        # ------------------------------------------------------------------
        # end of definition of network

    def build_restored_model(self):

        old_output = tf.get_default_graph().get_tensor_by_name('nn_output:0')
        # old_output = tf.Print(old_output, [old_output], summarize=400)

        # added layer
        # self.inference = tf.layers.dense(old_output, 127, activation=tf.nn.sigmoid, name='mapping_layer')
        self.inference = old_output

    def loss(self, phoneme_batch, new_cond=None):
        phoneme_batch = tf.cast(phoneme_batch, dtype=tf.int32)  # cast to int and put them in [[alignments]]

        # ----
        # train output layer to create P(s_k|m_j)
        if self.restore:
            # used_loss = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(tf.squeeze(phoneme_batch), 127, axis=1) *
            #                                           tf.log(self.inference), reduction_indices=[1]))
            used_loss = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(tf.squeeze(phoneme_batch), 127, axis=1) *
                                                      tf.log(tf.tensordot(self.inference, tf.transpose(new_cond),
                                                                          axes=1)), reduction_indices=[1]))

        else:
            # TODO no tf.assert for None
            # used hand-made P(s_k|m_j)
            used_loss = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(tf.squeeze(phoneme_batch), 127, axis=1) *
                                                      tf.log(tf.tensordot(self.inference, tf.transpose(new_cond),
                                                                          axes=1)), reduction_indices=[1]))

        # ---

        return used_loss

    def _old_loss(self, phoneme_batch, log_cn_pr):
        """
        deprecated!

        :param phoneme_batch:
        :param log_cn_pr:
        :return:
        """
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

        s2 = -self.scale / batch_size * tf.losses.sigmoid_cross_entropy(labels_loss, output_nn * gather_data)

        return s2

