import tensorflow as tf


class Loss(object):
    def __init__(self, logits, labels, cond_prob=None, identifier=None):
        self._logits = logits
        self._labels = labels
        self._identifier = identifier
        self.loss = self._get_loss(cond_prob=cond_prob)

    def _get_loss(self, cond_prob=None):
        # creating a dict for storing all the losses which we need
        # add the loss which you need

        self._labels = tf.cast(self._labels, dtype=tf.int32)  # cast to int and put them in [[alignments]]
        # ----
        # train output layer to create P(s_k|m_j)

        if self._identifier == 'vanilla':
            used_loss = tf.losses.softmax_cross_entropy(tf.one_hot(tf.squeeze(self._labels), 127, axis=1),
                                                        self._logits)
        elif self._identifier == 'nnvq' and cond_prob is not None:
            used_loss = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(tf.squeeze(self._labels), 127, axis=1) *
                                          tf.log(tf.tensordot(self._logits, tf.transpose(cond_prob),
                                                              axes=1)), reduction_indices=[1]))



        # if cond_prob is not None:
        #     dict_losses['nnvq'] = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(tf.squeeze(self._labels), 127, axis=1) *
        #                                               tf.log(tf.tensordot(self._logits, tf.transpose(cond_prob),
        #                                                                   axes=1)), reduction_indices=[1]))

        # if cond_prob is None:
        #     used_loss = tf.losses.softmax_cross_entropy(tf.one_hot(tf.squeeze(tf.cast(self._labels, dtype=tf.int32)), 127, axis=1),
        #                                                 self._logits)
        #     #
        #     # used_loss = tf.losses.mean_squared_error(tf.one_hot(tf.squeeze(self._labels), 127, axis=1),
        #     #                                          self._logits)
        #
        # else:
        #     # used hand-made P(s_k|m_j)
        #     used_loss = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(tf.squeeze(self._labels), 127, axis=1) *
        #                                               tf.log(tf.tensordot(self._logits, tf.transpose(cond_prob),
        #                                                                   axes=1)), reduction_indices=[1]))

        # ---

        return used_loss