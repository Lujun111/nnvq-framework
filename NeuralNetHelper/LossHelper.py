import tensorflow as tf


class Loss(object):
    def __init__(self, logits, labels):
        self._logits = logits
        self._labels = labels

    def loss(self, cond_prob=None):
        self._labels = tf.cast(self._labels, dtype=tf.int32)  # cast to int and put them in [[alignments]]

        # ----
        # train output layer to create P(s_k|m_j)
        if cond_prob is None:
            used_loss = tf.losses.softmax_cross_entropy(tf.one_hot(tf.squeeze(self._labels), 127, axis=1),
                                                        self._logits)

            # used_loss = tf.losses.mean_squared_error(tf.one_hot(tf.squeeze(phoneme_batch), 127, axis=1),
            #                                          self.inference_learned)

        else:
            # used hand-made P(s_k|m_j)
            used_loss = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(tf.squeeze(self._labels), 127, axis=1) *
                                                      tf.log(tf.tensordot(self._logits, tf.transpose(cond_prob),
                                                                          axes=1)), reduction_indices=[1]))

        # ---
        return used_loss
