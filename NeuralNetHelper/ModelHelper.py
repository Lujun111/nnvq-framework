import tensorflow as tf
# import tensorflow_probability as tfp


class Model(object):
    """
    Creates a model object
    """
    def __init__(self, settings):
        """
        Init the Model

        :param input_batch: size of batch
        """
        # set settings
        self.settings = settings

        # input placeholders
        self.train = tf.placeholder(tf.bool, name="is_train")
        self.features = tf.placeholder(tf.float32, shape=[None, self.settings.dim_features], name='ph_features')

        # output of model
        self.inference = []
        self.inference_learned = []
        self.wo_soft = []

        # create model
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

        fc1 = tf.layers.dense(self.features, num_neurons, activation=tf.nn.relu)
        fc1_bn = tf.layers.batch_normalization(fc1, training=self.train, center=False, scale=False)

        # fc1_dropout = tf.layers.dropout(fc1_bn, rate=0.3, training=self.train)
        fc2 = tf.layers.dense(fc1_bn, num_neurons, activation=tf.nn.relu)
        fc2_bn = tf.layers.batch_normalization(fc2, training=self.train, center=False, scale=False)
        # fc2_dropout = tf.layers.dropout(fc2_bn, rate=0.3, training=self.train)

        # WTA-layer starts here
        out = tf.layers.dense(fc2_bn, self.settings.codebook_size, activation=tf.nn.sigmoid)
        out_scaled = tf.scalar_mul(self.settings.scale_soft, out)
        # output without softmax
        # self.wo_soft = out_scaled
        # output with soft, be aware use a name 'nn_output' for the output node!
        self.inference = tf.nn.softmax(out_scaled, name='nn_output')

        # learn the mapping
        if self.settings.train_prob:
            self.wo_soft = tf.layers.dense(self.inference, 127, activation=None)
            self.inference_learned = tf.nn.softmax(self.wo_soft, name='new_nn_output')

        # ------------------------------------------------------------------
        # end of definition of network

    def loss(self, phoneme_batch, new_cond=None):
        phoneme_batch = tf.cast(phoneme_batch, dtype=tf.int32)  # cast to int and put them in [[alignments]]

        # ----
        # train output layer to create P(s_k|m_j)
        if self.settings.train_prob:
            used_loss = tf.losses.softmax_cross_entropy(tf.one_hot(tf.squeeze(phoneme_batch), 127, axis=1),
                                                        self.wo_soft)

            # used_loss = tf.losses.mean_squared_error(tf.one_hot(tf.squeeze(phoneme_batch), 127, axis=1),
            #                                          self.inference_learned)

        else:
            # used hand-made P(s_k|m_j)
            if new_cond is not None:
                used_loss = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(tf.squeeze(phoneme_batch), 127, axis=1) *
                                                          tf.log(tf.tensordot(self.inference, tf.transpose(new_cond),
                                                                              axes=1)), reduction_indices=[1]))
            else:
                raise ValueError("new_cond is None, please insert value")

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

