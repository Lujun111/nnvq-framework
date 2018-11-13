import tensorflow as tf
from NeuralNetHelper.MiscNNHelper import MiscNN


class Loss(object):
    def __init__(self, model, labels, settings, cond_prob=None):
        """
        Create your individual loss for your model

        :param logits:      logits from the model (without softmax)
        :param labels:      labels for calculating the loss
        :param cond_prob:   P(s_k|m_j) probability for calculating the loss
        :param identifier:  define which loss is used
        """
        self._model = model
        self._labels = labels
        self._settings = settings
        self._cond_prob = cond_prob
        self._misc = MiscNN(settings)
        self.loss = self._get_loss()

    def _get_loss(self):
        """
        Depending on the used identifier (string) return the loss respectively
        We are defining our own loss here which is need for training

        :return:    Returning the loss for training the model
        """
        self._labels = tf.cast(self._labels, dtype=tf.int32)  # cast to int and put them in [[alignments]]

        if self._settings.identifier is None:
            raise TypeError("Please define a proper identifier")
        # loss for training a vanilla network
        if self._settings.identifier == 'vanilla':
            used_loss = tf.losses.softmax_cross_entropy(tf.one_hot(tf.squeeze(self._labels), self._settings.num_labels,
                                                                   axis=1), self._model.logits)
        # loss for training a nnvq network
        elif self._settings.identifier == 'nnvq':
            cond_prob = self._misc.conditioned_probability(self._model.inference, self._labels,
                                                           discrete=self._settings.sampling_discrete)

            smoothed_labels = self._misc.label_smoothing(tf.one_hot(tf.squeeze(self._labels), self._settings.num_labels,
                                                                    axis=1), epsilon=0.1)
            used_loss = tf.reduce_mean(-tf.reduce_sum(smoothed_labels *
                                                      tf.log(tf.tensordot(self._model.inference, tf.transpose(cond_prob),
                                                                          axes=1)), reduction_indices=[1]))
        elif self._settings.identifier == 'nnvq_tri':
            # cond_prob = self._misc.conditioned_probability(self._model.inference, self._labels,
            #                                                discrete=self._settings.sampling_discrete)

            smoothed_labels = self._misc.label_smoothing(tf.one_hot(tf.squeeze(self._labels), self._settings.num_labels,
                                                                    axis=1), epsilon=0.1)
            used_loss = tf.reduce_mean(-tf.reduce_sum(smoothed_labels *
                                                      tf.log(tf.tensordot(self._model.inference,
                                                                          tf.transpose(self._cond_prob),
                                                                          axes=1)), reduction_indices=[1]))
            used_loss = tf.Print(used_loss, [used_loss])
        elif self._settings.identifier == 'combination':
            cond_prob = self._misc.conditioned_probability(self._model.logits, self._labels,
                                                           discrete=self._settings.sampling_discrete)

            l1_vanilla = tf.losses.softmax_cross_entropy(tf.one_hot(tf.squeeze(self._labels), self._settings.num_labels,
                                                                    axis=1), self._model.logits_vanilla)
            l2_vq = tf.reduce_mean(-tf.reduce_sum(tf.one_hot(tf.squeeze(self._labels), self._settings.num_labels, axis=1) *
                                                  tf.log(tf.tensordot(self._model.inference_nnvq, tf.transpose(cond_prob),
                                                                      axes=1)), reduction_indices=[1]))
            l3_combination = tf.losses.softmax_cross_entropy(tf.one_hot(tf.squeeze(self._labels),
                                                                        self._settings.num_labels, axis=1),
                                                             self._model.logits_combination)
            used_loss = 0.45 * l1_vanilla + 0.1 * l2_vq + 0.45 * l3_combination
        else:
            raise NotImplementedError("The used identifier does not exist, please define it!")

        return used_loss
