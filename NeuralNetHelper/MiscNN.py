import tensorflow as tf


class MiscNN(object):
    """
    MiscNN object contains auxiliary functions which are needed for
    logging and to compute functions in the graph
    """
    def __init__(self, codebook_size, num_labels):
        """
        Init MiscNN

        :param codebook_size: size of the codebook
        :param num_labels:    number of labels (e.g. for monophone states=127)
        """
        self.cb_size = codebook_size
        self.num_labels = num_labels

    def calculate_mi_tf(self, y_nn, labels):
        """
        Calculate the mutual information between the neural net output and
        the labels (phonemes/states)

        :param y_nn:    output of the neural net
        :param labels:  phonemes or states of the data (coming from the alignments of kaldi)
        :return:        mutual information
        """
        alpha = 1.0
        beta = -1.0

        # take the argmax of y_nn to get the class label determined by the
        # neural network
        y_labels = tf.argmax(y_nn, axis=1)

        # get p_w, p_y and p_w_y from helper
        p_w, p_y, p_w_y = self._helper_mi_tf(y_labels, labels)

        # normalize
        p_w /= tf.reduce_sum(p_w)
        p_y /= tf.reduce_sum(p_y)
        p_w_y = tf.divide(p_w_y,
                          tf.expand_dims(tf.clip_by_value(tf.reduce_sum(p_w_y, axis=1),
                                                          1e-8, 1e6), 1))
        # H(Y) on log2 base
        h_y = tf.multiply(p_y, tf.log(tf.clip_by_value(p_y, 1e-8, 1e6)) / tf.log(2.0))
        h_y = tf.reduce_sum(h_y)

        # H(W) on log2 base
        h_w = tf.multiply(p_w, tf.log(tf.clip_by_value(p_w, 1e-8, 1e6)) / tf.log(2.0))
        h_w = tf.reduce_sum(h_w)

        # H(W|Y) on log2 base
        h_w_y = p_w_y * tf.log(tf.clip_by_value(p_w_y, 1e-12, 1e6)) / tf.log(2.0)
        h_w_y = tf.reduce_sum(h_w_y, axis=0)
        h_w_y = tf.multiply(p_y, h_w_y)
        h_w_y = tf.reduce_sum(h_w_y)

        return -alpha * h_w - beta * h_w_y, -h_w, -h_y, -h_w_y

    def _helper_mi_tf(self, y_labels, labels):
        """
        Create P(w), P(y) and P(w|y) using the output labels of the neural network
        For deeper understanding how we create these probability, please check the
        TF-API

        :param y_labels:                output labels of the neural net
        :param labels:                  phonemes or states of the data (coming from the alignments of kaldi)
        :return pwtmp, pytmp, pw_y_tmp: return P(w), P(y) and P(w|y)
        """

        # define tf variables to use scatter_nd and scatter_nd_add
        pwtmp = tf.Variable(tf.zeros(self.num_labels), trainable=False, dtype=tf.float32)
        pytmp = tf.Variable(tf.zeros(self.cb_size), trainable=False, dtype=tf.float32)
        pw_y_tmp = tf.Variable(tf.zeros([self.num_labels, self.cb_size]), trainable=False, dtype=tf.float32)

        # create P(w)
        pwtmp = pwtmp.assign(tf.fill([self.num_labels], 0.0))  # reset Variable/floor
        pwtmp = tf.scatter_add(pwtmp, labels, tf.ones(tf.shape(labels)))

        # create P(y)
        pytmp = pytmp.assign(tf.fill([self.cb_size], 0.0))  # reset Variable/floor
        pytmp = tf.scatter_add(pytmp, y_labels, tf.ones(tf.shape(y_labels)))

        # create P(w|y)
        pw_y_tmp = pw_y_tmp.assign(tf.fill([self.num_labels, self.cb_size], 0.0))  # reset Variable/floor
        pw_y_tmp = tf.scatter_nd_add(pw_y_tmp,
                                     tf.concat([tf.cast(labels, dtype=tf.int64), tf.expand_dims(y_labels, 1)],
                                               axis=1), tf.ones(tf.shape(y_labels)))

        # adding to graph for visualisation in tensorboard
        tf.identity(pwtmp, 'P(w)')
        tf.identity(pytmp, 'P(y)')
        tf.identity(pw_y_tmp, 'P(w|y)')

        return pwtmp, pytmp, pw_y_tmp

    def conditioned_probability(self, y_nn, labels, discrete=False):
        """
        Create the conditioned probability P(s_k|m_j) using the output of the neural network
        and the target labels coming out of kaldi

        :param y_nn:        output of the neural net
        :param labels:      phonemes or states of the data (coming from the alignments of kaldi)
        :param discrete:    flag for creating P(s_k|m_j) in the discrete way (check dissertation
                            of Neukirchen, p.62 (5.51))
        :return:            return P(s_k|m_j)
        """
        # small delta for smoothing P(s_k|m_j), necessary if we log the probability
        eps = 0.01

        # cast labels into int32
        labels = tf.cast(labels, dtype=tf.int32)  # cast to int and put them in [[alignments]]

        # create variable in order to use scatter_nd_add and scatter_add (discrete creation of P(s_k|m_j))
        nominator = tf.Variable(tf.zeros([self.num_labels, self.cb_size]), trainable=False, dtype=tf.float32, name='nom_test')
        nominator = nominator.assign(tf.fill([self.num_labels, self.cb_size], eps))  # reset Variable/floor
        denominator = tf.Variable(tf.zeros([self.cb_size]), trainable=False, dtype=tf.float32, name='den_test')
        denominator = denominator.assign(tf.fill([self.cb_size], self.num_labels * eps))  # reset Variable/floor

        if discrete:
            # discretize the output of the neural network
            output_dis = tf.argmax(y_nn, axis=1)

            # get nominator
            nominator = tf.scatter_nd_add(nominator, tf.concat([tf.cast(labels, dtype=tf.int64),
                                                                tf.expand_dims(output_dis, 1)],
                                                               axis=1), tf.ones(tf.shape(output_dis)))
            # get denominator
            denominator = tf.scatter_add(denominator, output_dis, tf.ones(tf.shape(output_dis)[0]))

            # create P(s_k|m_j) in the discrete way
            conditioned_prob = tf.div(nominator, tf.expand_dims(denominator, 0))

        else:
            # get nominator
            nominator = tf.scatter_nd_add(nominator, labels, y_nn)

            # get denominator
            denominator = tf.reduce_sum(y_nn, axis=0)

            # smoothing the probability
            denominator += self.num_labels * eps

            # create P(s_k|m_j) in the continuous way
            conditioned_prob = tf.divide(nominator, denominator)

        return conditioned_prob

    def joint_probability(self, y_nn, labels):
        # TODO I don't know if it works properly because I only use it for logging
        """
        Create joint probability P(s_k, m_j)

        :param y_nn:    output of the neural net
        :param labels:  phonemes or states of the data (coming from the alignments of kaldi)
        :return:        return P(s_k, m_j)
        """

        # determine batch size
        batch_size = tf.cast(tf.shape(y_nn)[0], dtype=tf.float32)

        # create variable in order to use scatter_nd_add
        joint_prob = tf.Variable(tf.zeros([self.num_labels, self.cb_size]), trainable=False, dtype=tf.float32)
        joint_prob = joint_prob.assign(tf.fill([self.num_labels, self.cb_size], 0.0))  # reset Variable/floor

        # cast labels to int32
        labels = tf.cast(labels, dtype=tf.int32)

        # create P(s_k, m_j), (check dissertation of Neukirchen, p.61 (5.46))
        joint_prob = tf.scatter_nd_add(joint_prob, labels, y_nn)
        joint_prob = tf.div(joint_prob, batch_size)

        return joint_prob

    def vq_data(self, y_nn, labels, nominator, denominator, discrete=True):
        """
        Create the nominator and denominator for P(s_k|m_j)
        This function is used for using all the training data to create P(s_k|m_j)

        :param y_nn:        output of the neural net
        :param labels:      phonemes or states of the data (coming from the alignments of kaldi)
        :param nominator:   nominator for P(s_k|m_j)
        :param denominator: denominator for P(s_k|m_j)
        :param discrete:    flag for creating P(s_k|m_j) in a discrete way
        :return:            return nominator and denominator of creating P(s_k|m_j)
        """
        # cast labels to int32
        labels = tf.cast(labels, dtype=tf.int32)

        y_labels = tf.argmax(y_nn, axis=1)
        # labels_softmax = output_soft

        if discrete:
            # create nominator
            nominator = tf.scatter_nd_add(nominator, tf.concat([tf.cast(labels, dtype=tf.int32),
                                                                tf.expand_dims(y_labels, 1)],
                                                               axis=1), tf.ones(tf.shape(y_labels)))

            # create dominator
            denominator = tf.scatter_add(denominator, y_labels, tf.ones(tf.shape(y_labels)[0]))
        else:
            raise NotImplementedError("Not implemented!")

        return nominator, denominator

    def testing_stuff(self, output_nn, cond_prob, phonemes):
        """
        deprecated!

        :param output_nn:
        :param cond_prob:
        :param phonemes:
        :return:
        """
        # labels_softmax = output_nn
        phonemes = tf.cast(phonemes, dtype=tf.int32)
        cond_prob = tf.transpose(cond_prob)
        # cond_prob = tf.Print(cond_prob, [cond_prob])

        s_k = tf.Variable(tf.zeros([127]), trainable=False, dtype=tf.float32)
        s_k.assign(tf.fill([127], 0.0))

        out_nn = tf.Variable(tf.zeros([400]), trainable=False, dtype=tf.float32)
        out_nn.assign(tf.fill([400], 0.0))

        labels_softmax = tf.argmax(output_nn, axis=1)

        s_k = tf.scatter_add(s_k, phonemes, tf.ones(tf.shape(phonemes)))
        s_k = tf.clip_by_value(s_k, 1e-15, 1e6)
        s_k /= tf.reduce_sum(s_k)

        # out_nn = tf.scatter_add(out_nn, labels_softmax, tf.ones(tf.shape(phonemes)[0]))
        # out_nn = tf.clip_by_value(out_nn, 1e-15, 1e6)
        # out_nn /= tf.reduce_sum(out_nn)

        tmp_out = tf.reduce_sum(output_nn, axis=0)
        out_nn = tmp_out / tf.reduce_sum(tmp_out)

        # out_nn = tf.reduce_sum(output_nn, axis=0) / tf.cast(tf.shape(output_nn)[0], dtype=tf.float32)

        joint = tf.tile(tf.expand_dims(out_nn, 1), [1, 127]) * cond_prob

        tmp = joint / tf.reduce_sum(joint, axis=1, keepdims=True)

        return tmp



