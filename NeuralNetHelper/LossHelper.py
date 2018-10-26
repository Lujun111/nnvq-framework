import tensorflow as tf


class Loss(object):
    def __init__(self, logits, labels, cond_prob=None, identifier=None):
        """
        Create your individual loss for your model

        :param logits:      logits from the model (without softmax)
        :param labels:      labels for calculating the loss
        :param cond_prob:   P(s_k|m_j) probability for calculating the loss
        :param identifier:  define which loss is used
        """
        self._logits = logits
        self._labels = labels
        self._identifier = identifier
        self._cond_prob = cond_prob
        self.loss = self._get_loss()

    def _get_loss(self):
        """
        Depending on the used identifier (string) return the loss respectively
        We are defining our own loss here which is need for training

        :return:    Returning the loss for training the model
        """
        self._labels = tf.cast(self._labels, dtype=tf.int32)  # cast to int and put them in [[alignments]]

        if self._identifier is None:
            raise TypeError("Please define a proper identifier")
        # loss for training a vanilla network
        if self._identifier == 'vanilla':
            used_loss = tf.losses.softmax_cross_entropy(tf.one_hot(tf.squeeze(self._labels), 127, axis=1),
                                                        self._logits)
        # loss for training a nnvq network
        elif self._identifier == 'nnvq' and self._cond_prob is not None:
            used_loss = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(tf.squeeze(self._labels), 127, axis=1) *
                                                      tf.log(tf.tensordot(self._logits, tf.transpose(self._cond_prob),
                                                                          axes=1)), reduction_indices=[1]))
        elif self._identifier == 'own':
            used_loss = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(tf.squeeze(self._labels), 127, axis=1) *
                                                      tf.log(self._logits), reduction_indices=[1]))
        else:
            raise NotImplementedError("The used identifier does not exist, please define it!")

        return used_loss
