import tensorflow as tf
# import tensorflow_probability as tfp


class Model(object):
    """
    Creates a model object
    """
    def __init__(self, ph_train, ph_features, settings):
        """
        Init the Model

        :param input_batch: size of batch
        """
        # set settings
        self.settings = settings

        # input placeholders
        # self.train = tf.placeholder(tf.bool, name="is_train")
        # self.features = tf.placeholder(tf.float32, shape=[None, self.settings.dim_features], name='ph_features')
        self.train = ph_train
        self.features = ph_features

        # output of model
        self.inference = []
        self.inference_learned = []
        self.logits = []

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

        with tf.variable_scope('base_network'):
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
            self.logits = out_scaled
            # output with soft, be aware use a name 'nn_output' for the output node!
            self.inference = tf.nn.softmax(self.logits, name='nn_output')

        # learn the mapping
        # with tf.variable_scope('added_network'):
        #     self.logits_new = tf.layers.dense(self.inference, 127, activation=None)
        #     self.inference_learned = tf.nn.softmax(self.logits_new, name='new_nn_output')

        # ------------------------------------------------------------------
        # end of definition of network

