import tensorflow as tf
from NeuralNetHelper.MiscNNHelper import MiscNN


class Model(object):
    """
    Creates a model object
    """
    def __init__(self, ph_train, ph_features, settings, ph_output=None):
        """
        Init the Model

        :param input_batch: size of batch
        """
        # set settings
        self._settings = settings

        # input placeholders
        # self.train = tf.placeholder(tf.bool, name="is_train")
        # self.features = tf.placeholder(tf.float32, shape=[None, self.settings.dim_features], name='ph_features')
        self.train = ph_train
        self.features = ph_features
        self._misc = MiscNN(settings)
        self.train_output = ph_output

        # output of model
        self.inference = []
        self.inference_learned = []
        self.logits = []
        self.alpha = None
        self.scale = None

        # create model
        if self._settings.identifier == 'nnvq':
            self._build_model_vq()
        elif self._settings.identifier == 'vanilla':
            self._build_model_vanilla()
        elif self._settings.identifier == 'combination':
            self._build_combination()
        # elif self.settings.identifier == 'restore':
        #     self._build_combination()

        self._input_tensor = self._misc.set_probabilities('model_checkpoint/vq_graph/p_s_m.mat')

    def _build_model_vq(self):
        """
        Build your model for training. All of the architecture
        is defined here.

        :param scale_softmax: Scaling for the WTA-layer (Winner-Takes-All)
        :param codebook_size: Size of the codebook
        """

        # Start here to define your network
        # ------------------------------------------------------------------
        num_neurons = 512

        with tf.variable_scope('base_network'):
            fc1 = tf.layers.dense(self.features, num_neurons, activation=tf.nn.relu)
            fc1_bn = tf.layers.batch_normalization(fc1, training=self.train, center=False, scale=False)
            fc1_dropout = tf.layers.dropout(fc1_bn, rate=0.25, training=self.train)

            fc2 = tf.layers.dense(fc1_dropout, num_neurons, activation=tf.nn.relu)
            fc2_bn = tf.layers.batch_normalization(fc2, training=self.train, center=False, scale=False)
            fc2_dropout = tf.layers.dropout(fc2_bn, rate=0.25, training=self.train)

            # fc3 = tf.layers.dense(fc2_bn, num_neurons, activation=tf.nn.relu)
            # fc3_bn = tf.layers.batch_normalization(fc3, training=self.train, center=False, scale=False)
            #
            # fc4 = tf.layers.dense(fc3_bn, num_neurons, activation=tf.nn.relu)
            # fc4_bn = tf.layers.batch_normalization(fc4, training=self.train, center=False, scale=False)

            # WTA-layer starts here
            out = tf.layers.dense(fc2_dropout, self._settings.codebook_size, activation=tf.nn.sigmoid)
            out_scaled = tf.scalar_mul(self._settings.scale_soft, out)
            # output without softmax
            self.logits = out_scaled
            # output with soft, be aware use a name 'nn_output' for the output node!
            self.inference = tf.nn.softmax(self.logits, name='nn_output')

        # learn the mapping
        # with tf.variable_scope('added_network'):
        #     # self.logits_new = tf.layers.dense(self.inference, 127, activation=None, use_bias=False,
        #     #                                   kernel_constraint=lambda x: tf.clip_by_value(x, 0.01, 1000))
        #
        #     self.logits_new = tf.layers.dense(self.inference, 127, activation=None)
        #     # self.logits_new = tf.Print(self.logits_new, [tf.reduce_min(self.logits_new)])
        #     self.inference_learned = tf.nn.softmax(self.logits_new, name='nn_output')
            # self.inference_learned = tf.divide(self.logits_new, tf.reduce_sum(self.logits_new, axis=1, keepdims=True),
            #                                    name='nn_output')
            # self.inference_learned = tf.Print(self.inference_learned, [tf.reduce_sum(self.inference_learned)])

        # ------------------------------------------------------------------
        # end of definition of network

    def _build_model_vanilla(self):
        num_neurons = 512

        with tf.variable_scope('vanilla_network'):
            fc1 = tf.layers.dense(self.features, num_neurons, activation=tf.nn.relu)
            fc1_bn = tf.layers.batch_normalization(fc1, training=self.train, center=False, scale=False)
            fc1_dropout = tf.layers.dropout(fc1_bn, rate=0.25, training=self.train)

            fc2 = tf.layers.dense(fc1_dropout, num_neurons, activation=tf.nn.relu)
            fc2_bn = tf.layers.batch_normalization(fc2, training=self.train, center=False, scale=False)
            fc2_dropout = tf.layers.dropout(fc2_bn, rate=0.25, training=self.train)
            #
            fc3 = tf.layers.dense(fc2_dropout, num_neurons, activation=tf.nn.relu)
            fc3_bn = tf.layers.batch_normalization(fc3, training=self.train, center=False, scale=False)
            fc3_dropout = tf.layers.dropout(fc3_bn, rate=0.25, training=self.train)

            fc4 = tf.layers.dense(fc3_dropout, num_neurons, activation=tf.nn.relu)
            fc4_bn = tf.layers.batch_normalization(fc4, training=self.train, center=False, scale=False)
            fc4_dropout = tf.layers.dropout(fc4_bn, rate=0.25, training=self.train)

            fc5 = tf.layers.dense(fc4_dropout, num_neurons, activation=tf.nn.relu)
            fc5_bn = tf.layers.batch_normalization(fc5, training=self.train, center=False, scale=False)
            fc5_dropout = tf.layers.dropout(fc5_bn, rate=0.25, training=self.train)

            self.logits = tf.layers.dense(fc5_dropout, 127, activation=None)
            # output with soft, be aware use a name 'nn_output' for the output node!
            self.inference = tf.nn.softmax(self.logits, name='nn_output')

    def _build_combination(self):
        # we combine the nnvq and vanilla network
        num_neurons = 512

        # with tf.variable_scope('scaling_network'):
        #     fc1_scale = tf.layers.dense(self.features, 1, activation=tf.nn.sigmoid)
        #     self.scale = 35 * fc1_scale
        #     # self.scale = tf.Print(self.scale, [tf.reduce_min(self.scale)])

        # first we build the nnvq network
        with tf.variable_scope('base_network'):
            fc1 = tf.layers.dense(self.features, num_neurons, activation=tf.nn.relu)
            fc1_bn = tf.layers.batch_normalization(fc1, training=self.train, center=False, scale=False)
            fc1_dropout = tf.layers.dropout(fc1_bn, rate=0.25, training=self.train)

            fc2 = tf.layers.dense(fc1_dropout, num_neurons, activation=tf.nn.relu)
            fc2_bn = tf.layers.batch_normalization(fc2, training=self.train, center=False, scale=False)
            fc2_dropout = tf.layers.dropout(fc2_bn, rate=0.25, training=self.train)

            # fc3 = tf.layers.dense(fc2_bn, num_neurons, activation=tf.nn.relu)
            # fc3_bn = tf.layers.batch_normalization(fc3, training=self.train, center=False, scale=False)
            #
            # fc4 = tf.layers.dense(fc3_bn, num_neurons, activation=tf.nn.relu)
            # fc4_bn = tf.layers.batch_normalization(fc4, training=self.train, center=False, scale=False)

            # WTA-layer starts here
            out = tf.layers.dense(fc2_dropout, self._settings.codebook_size, activation=tf.nn.sigmoid)
            out_scaled = tf.scalar_mul(self._settings.scale_soft, out)
            # out_scaled = tf.multiply(self.scale, out)
            # out_scaled = tf.Print(out_scaled, [out_scaled])
            # output without softmax
            self.logits_nnvq = out_scaled
            # output with soft, be aware use a name 'nn_output' for the output node!
            self.inference_nnvq = tf.nn.softmax(self.logits_nnvq, name='nn_output')

        with tf.variable_scope('vanilla_network'):
            fc1_van = tf.layers.dense(self.features, num_neurons, activation=tf.nn.relu)
            fc1_van_bn = tf.layers.batch_normalization(fc1_van, training=self.train, center=False, scale=False)
            fc1_van_dropout = tf.layers.dropout(fc1_van_bn, rate=0.25, training=self.train)

            fc2_van = tf.layers.dense(fc1_van_dropout, num_neurons, activation=tf.nn.relu)
            fc2_van_bn = tf.layers.batch_normalization(fc2_van, training=self.train, center=False, scale=False)
            fc2_van_dropout = tf.layers.dropout(fc2_van_bn, rate=0.25, training=self.train)
            #
            fc3_van = tf.layers.dense(fc2_van_dropout, num_neurons, activation=tf.nn.relu)
            fc3_van_bn = tf.layers.batch_normalization(fc3_van, training=self.train, center=False, scale=False)
            fc3_van_dropout = tf.layers.dropout(fc3_van_bn, rate=0.25, training=self.train)

            self.logits_vanilla = tf.layers.dense(fc3_van_dropout, 127, activation=None)
            # output with soft, be aware use a name 'nn_output' for the output node!
            self.inference_vanilla = tf.nn.softmax(self.logits_vanilla, name='nn_output')

        # we combine the models here
        with tf.variable_scope('combination_network'):
            # alpha = tf.layers.dense(self.features, 1, activation=tf.nn.sigmoid)
            # self.logits_combination = tf.layers.dense(tf.concat([self.logits_nnvq, self.logits_vanilla], 1), 127,
            #                                           activation=None)

            # test_tmp = tf.tensordot(self.inference_nnvq, self._input_tensor, axes=1)

            out_combi1 = tf.layers.dense(tf.concat([out, self.logits_vanilla], 1), num_neurons,
                                         activation=tf.nn.relu)
            out_combi1_bn = tf.layers.batch_normalization(out_combi1, training=self.train_output, center=False, scale=False)
            out_combi1_dropout = tf.layers.dropout(out_combi1_bn, rate=0.25, training=self.train_output)

            # out_combi2 = tf.layers.dense(out_combi1_dropout, num_neurons,
            #                              activation=tf.nn.relu)
            # out_combi2_bn = tf.layers.batch_normalization(out_combi2, training=self.train_output, center=False,
            #                                               scale=False)
            # out_combi2_dropout = tf.layers.dropout(out_combi2_bn, rate=0.25, training=self.train_output)

            self.logits_combination = tf.layers.dense(out_combi1_dropout, 127, activation=None)
            self.inference_combination = tf.nn.softmax(self.logits_combination, name='nn_output')
            #
            # out_combi1 = tf.layers.dense(tf.concat([out, self.logits_vanilla, self.features], 1),
            #                              num_neurons, activation=tf.nn.relu)
            # out_combi1 = tf.layers.dense(tf.concat([out, self.logits_vanilla], 1), num_neurons, activation=tf.nn.relu)
            # out_combi2 = tf.layers.dense(out_combi1, num_neurons, activation=tf.nn.relu)

            # self.alpha = tf.layers.dense(tf.concat([out, self.logits_vanilla], 1), 1, activation=tf.nn.sigmoid)
            # # alpha = tf.Print(alpha, [tf.reduce_min(alpha)])
            #
            # self.inference_combination = tf.multiply(tf.pow(self.inference_vanilla, self.alpha),
            #                              tf.pow(tf.tensordot(self.inference_nnvq, self._input_tensor, axes=1),
            #                                     (tf.ones(tf.shape(self.alpha)) - self.alpha)), name='nn_output')